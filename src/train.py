import os
from typing import Dict, Any
import torch

__all__ = ["ZeroSketch", "train_sketch"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZeroSketch:
    """Light-weight Count-Min style sketch with conservative update.

    The earlier implementation produced an erroneous tensor shape
    (n_keys, rows, rows) which triggered a RuntimeError during the
    comparison `vals > cur`.  The gather/scatter logic has been
    rewritten so that every intermediate has the expected
    (n_keys, rows) shape and the in-place update only touches the
    counters that actually need to grow.
    """

    def __init__(self, rows: int = 3, w: int = 32) -> None:
        self.w = w
        self.rows = rows
        # random 64-bit hashing parameters – fixed at construction for reproducibility
        self.hash_a = torch.randint(1, 2 ** 31, (rows,), device=device, dtype=torch.int64)
        # global fp16 accumulator (shared across all graph nodes)
        self.C = torch.zeros(rows, w, device=device, dtype=torch.float16)

    # ---------------------------------------------------------------------
    #                     Hashing / gather / scatter helpers
    # ---------------------------------------------------------------------
    def _indices(self, keys: torch.Tensor) -> torch.Tensor:
        """Return the (n_keys, rows) table indices for every key."""
        # Broadcasting: keys[:, None] –> (n_keys, 1) * (rows,) –> (n_keys, rows)
        return ((keys[:, None] * self.hash_a) % self.w).long()

    def _gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Fast gather so that the result is (n_keys, rows) not (n_keys, rows, rows)."""
        # C.gather expects the index to have the same first-dim (rows)
        # Layout: C        – (rows, w)
        #         idx.T    – (rows, n_keys)
        # Result           – (rows, n_keys) → transpose → (n_keys, rows)
        return self.C.gather(1, idx.T).T.contiguous()

    # ---------------------------------------------------------------------
    #                        Public sketch interface
    # ---------------------------------------------------------------------
    @torch.cuda.amp.autocast(dtype=torch.float16)
    def query(self, keys: torch.Tensor) -> torch.Tensor:
        """Return the conservative (min-row) estimate for a batch of keys."""
        idx = self._indices(keys)
        cur = self._gather(idx)  # (n_keys, rows)
        return cur.min(dim=1).values

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def update(self, keys: torch.Tensor, vals: torch.Tensor) -> None:
        """In-place conservative update.

        For every (key, row) pair we only touch the counter if the
        incoming value is larger than the current one.  We achieve an
        O(1) atomic update per key by masking the scatter targets with
        a 0/1 `need` tensor.
        """
        idx = self._indices(keys)                      # (n_keys, rows)
        cur = self._gather(idx)                        # (n_keys, rows)

        # --------------------------- mask of needed updates ---------------------------
        need = (vals.unsqueeze(1) > cur)               # (n_keys, rows) – boolean
        if not need.any():
            return  # nothing to do – early exit avoids needless scatter

        # Shapes compatible with scatter_(add) ➜  (rows, n_keys)
        need_t   = need.T.float()                      # (rows, n_keys)
        idx_t    = idx.T                               # (rows, n_keys)
        vals_t   = vals.unsqueeze(0).expand(self.rows, -1)  # (rows, n_keys)

        # ------------------------ atomic add (0 or val) -----------------------------
        # note: adding the whole value (instead of delta) would over-shoot, therefore
        # we first zero-out the source wherever the counter is already sufficient.
        self.C.scatter_add_(1,
                            idx_t,
                            need_t * (vals_t - self.C.gather(1, idx_t)))

    # ---------------------------------------------------------------------
    #                Convenience helpers (ckpt / restore)                   
    # ---------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"C": self.C, "hash_a": self.hash_a}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.C.copy_(state["C"])
        self.hash_a.copy_(state["hash_a"])


# ----------------------------------------------------------------------------------
#                              Training harness
# ----------------------------------------------------------------------------------

def train_sketch(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A minimal synthetic workload that stress-tests the ZeroSketch.

    The loop performs a single (update → query) pass over `n_keys`
    random key/value pairs and returns simple error metrics so that CI
    can assert numerical sanity.
    """
    torch.manual_seed(cfg.get("seed", 0))

    n_keys: int = cfg["sketch"].get("n_keys")
    max_val: float = cfg["sketch"].get("max_val")

    # -------------------------- deterministic synthetic data -------------------------
    keys = torch.arange(n_keys, device=device, dtype=torch.int64)
    vals = torch.rand(n_keys, device=device, dtype=torch.float16) * max_val

    sketch = ZeroSketch(rows=cfg["sketch"].get("rows", 3),
                        w=cfg["sketch"].get("w", 32))

    # one pass: update then query (mirrors sampler life-cycle)
    sketch.update(keys, vals)
    estimates = sketch.query(keys)

    # ------------------------------ error statistics --------------------------------
    over_estimation = (estimates - vals).clamp_min(0.0)
    mae      = float(over_estimation.mean().item())
    max_err  = float(over_estimation.max().item())

    metrics = {
        "n_keys"    : int(n_keys),
        "rows"      : int(sketch.rows),
        "width"     : int(sketch.w),
        "mae"       : mae,
        "max_error": max_err,
    }
    return metrics