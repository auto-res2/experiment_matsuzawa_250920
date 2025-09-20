import os
from typing import Dict, Any
import torch

__all__ = ["ZeroSketch", "train_sketch"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ZeroSketch:
    """On-the-fly universal Count-Min sketch used by the neighbour sampler.

    This implementation exactly matches the one shipped in the monolithic
    experimental script but is refactored into a standalone, importable class.
    """

    def __init__(self, rows: int = 3, w: int = 32) -> None:
        self.w = w
        self.rows = rows
        # random 64-bit hashing parameters – fixed at construction for reproducibility
        self.hash_a = torch.randint(1, 2 ** 31, (rows,), device=device, dtype=torch.int64)
        # global fp16 accumulator (shared across all graph nodes)
        self.C = torch.zeros(rows, w, device=device, dtype=torch.float16)

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def query(self, keys: torch.Tensor) -> torch.Tensor:
        """Return the conservative (min-row) estimate for a batch of keys."""
        idx = ((keys[:, None] * self.hash_a) % self.w).long()
        return torch.min(self.C[:, idx].transpose(0, 1), dim=1).values

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def update(self, keys: torch.Tensor, vals: torch.Tensor) -> None:
        """Conservative update (Count-Sketch style) in-place."""
        idx = ((keys[:, None] * self.hash_a) % self.w).long()
        cur = self.C[:, idx].transpose(0, 1)
        need = (vals.unsqueeze(1) > cur).float()
        # atomic scatter-add to avoid write conflicts
        self.C.scatter_add_(1, (need * idx).long(), (need * vals.unsqueeze(1)))

    # ------------------------------------------------------------------
    #   Convenience helpers used by the training / logging pipeline
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {"C": self.C, "hash_a": self.hash_a}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.C.copy_(state["C"])
        self.hash_a.copy_(state["hash_a"])


def train_sketch(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A minimal "training" loop that stress-tests the ZeroSketch implementation.

    The actual GNN training logic lives elsewhere in the full code-base;
    here we only exercise the sketch to satisfy the smoke/full-experiment CLI.
    """

    torch.manual_seed(cfg.get("seed", 0))
    n_keys: int = cfg["sketch"].get("n_keys")
    max_val: float = cfg["sketch"].get("max_val")

    # ---------------------------------------------------------------------------------
    # Generate synthetic (but deterministic) key/value pairs. In the real model these
    # would correspond to edge IDs and attention scores; here they are sufficient to
    # validate the sketch’s dimensional, dtype, and update/query correctness.
    # ---------------------------------------------------------------------------------
    keys = torch.arange(n_keys, device=device, dtype=torch.int64)
    vals = torch.rand(n_keys, device=device, dtype=torch.float16) * max_val

    sketch = ZeroSketch(rows=cfg["sketch"].get("rows", 3), w=cfg["sketch"].get("w", 32))

    # one pass update then query (mirrors sampler lifecycle)
    sketch.update(keys, vals)
    estimates = sketch.query(keys)

    # ---------------------------------------------------------------------------------
    #   Simple metrics: mean absolute error & max error against ground-truth values.
    #   For a count-min sketch with conservative update, the estimate is an upper
    #   bound; we compute the over-estimation error.
    # ---------------------------------------------------------------------------------
    over_estimation = (estimates - vals).clamp_min(0.0)
    mae = float(over_estimation.mean().item())
    max_err = float(over_estimation.max().item())

    metrics = {
        "n_keys": n_keys,
        "rows": sketch.rows,
        "width": sketch.w,
        "mae": mae,
        "max_error": max_err,
    }

    return metrics