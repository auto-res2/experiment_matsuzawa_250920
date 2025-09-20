import os
from typing import Dict, Any
import torch

__all__ = ["ZeroSketch", "train_sketch"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ZeroSketch:
    """Light-weight Count-Min style sketch with conservative update.

    The implementation keeps a *global* counter matrix `C` of shape
    (rows, w).  Any (key,val) pair is mapped to `rows` counters via a
    fast multiply-mod hash.  The *conservative* update writes **only**
    those counters whose current value is smaller than the incoming
    one.  This guarantees the usual Count-Min error bound while
    preventing write-amplification on heavily duplicated keys.

    IMPORTANT – dtype discipline:
      • `C` lives in *fp16* (sufficient for sketch errors we measure).
      • All temporaries must therefore stay in *fp16* when fed into
        `scatter_add_`.  A previous bug cast the update mask to fp32
        which triggered a `RuntimeError: scatter(): Expected self.dtype
        to be equal to src.dtype`.
    """

    def __init__(self, rows: int = 3, w: int = 32) -> None:
        self.w = w
        self.rows = rows
        # random 64-bit hashing parameters – fixed at construction for reproducibility
        self.hash_a = torch.randint(1, 2 ** 31, (rows,), device=device, dtype=torch.int64)
        # global fp16 accumulator (shared across all graph nodes)
        self.C = torch.zeros(rows, w, device=device, dtype=torch.float16)

    # ------------------------------------------------------------------
    #                     Hashing / gather / scatter helpers
    # ------------------------------------------------------------------
    def _indices(self, keys: torch.Tensor) -> torch.Tensor:
        """Return the (n_keys, rows) table indices for every key."""
        # Broadcasting: keys[:, None] –> (n_keys, 1) * (rows,) –> (n_keys, rows)
        return ((keys[:, None] * self.hash_a) % self.w).long()

    def _gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Efficient gather keeping a 2-D result (n_keys, rows)."""
        # C.gather expects the index to have the same first-dim (rows)
        # Layout: C     – (rows, w)
        #         idx.T – (rows, n_keys)
        # Result        – (rows, n_keys) → transpose → (n_keys, rows)
        return self.C.gather(1, idx.T).T.contiguous()

    # ------------------------------------------------------------------
    #                        Public sketch interface
    # ------------------------------------------------------------------
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
        incoming value is larger than the current one.  A *mask* `need`
        identifies those counters; multiplication by the mask turns the
        source of `scatter_add_` into 0 or `delta` (new−old), achieving
        an atomic O(1) update per counter.
        """
        idx = self._indices(keys)          # (n_keys, rows)
        cur = self._gather(idx)            # (n_keys, rows)

        # --------------------------- mask of needed updates ---------------------------
        need = vals.unsqueeze(1) > cur      # (n_keys, rows) – boolean
        if not need.any():
            return  # nothing to do – early exit avoids needless scatter

        # --------------------------- prepare fp16 tensors -----------------------------
        # Shapes compatible with scatter_(add)  →  (rows, n_keys)
        need_t   = need.T.to(dtype=torch.float16)                 # (rows, n_keys) fp16  {0,1}
        idx_t    = idx.T                                          # (rows, n_keys) int64
        cur_t    = cur.T                                          # (rows, n_keys) fp16 (view)
        vals_t   = vals.unsqueeze(0).expand(self.rows, -1)        # (rows, n_keys) fp16

        delta = vals_t - cur_t            # only positive where need_t==1 (but cheap to compute)
        src   = need_t * delta            # fp16 – same dtype as C

        # ------------------------ atomic add (0 or delta) -----------------------------
        # NOTE: src and self.C have identical dtype (fp16) – this avoids the scatter() dtype error.
        self.C.scatter_add_(1, idx_t, src)

    # ------------------------------------------------------------------
    #                Convenience helpers (ckpt / restore)               
    # ------------------------------------------------------------------
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

    n_keys: int  = cfg["sketch"].get("n_keys")
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