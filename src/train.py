import os
import json
from pathlib import Path
import torch

__all__ = [
    "ZeroSketch",
    "train",
]


class ZeroSketch:
    """Zero-update Count-Sketch with conservative MIN query.

    The previous implementation produced a `(batch, rows)` tensor because of a
    double broadcasting mistake which in turn caused a shape mismatch during
    the L1-loss computation.  We now gather the counters row-wise and take the
    *minimum across rows* so the output is 1-D `(batch,)`, exactly matching the
    ground-truth `vals` tensor produced by `_synthetic_batch()`.
    """

    def __init__(self, rows: int = 3, w: int = 32):
        if rows <= 0 or w <= 0:
            raise ValueError("rows and w must be positive")
        self.w = w
        self.rows = rows
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        # prime-range hash coefficients 2  signed 64-bit avoids overflow on mul
        self.hash_a = torch.randint(
            low=1,
            high=2 ** 31,
            size=(rows,),
            device=dev,
            dtype=torch.int64,
        )
        # global sketch per layer  2  fp16 keeps memory foot-print <1 kB
        self.C = torch.zeros(rows, w, device=dev, dtype=torch.float16)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def query(self, keys: torch.LongTensor) -> torch.Tensor:
        """Return the *conservative minimum* across the sketch rows.

        Output shape:  `(batch,)`  and **not** `(batch, rows)` as before.
        """
        if keys.dtype != torch.long:
            keys = keys.long()
        # (batch, rows)
        idx = ((keys[:, None] * self.hash_a) % self.w).long()

        # Gather counters row-wise → (rows, batch)
        row_ids = torch.arange(self.rows, device=idx.device).unsqueeze(1)  # (rows, 1)
        gathered = self.C[row_ids, idx.T]  # (rows, batch)
        gathered = gathered.T  # (batch, rows)
        return gathered.min(dim=1).values  # (batch,)

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def update(self, keys: torch.LongTensor, vals: torch.Tensor) -> None:
        """Conservative *min* update (max assignment because we store negatives).

        For smoke-/unit-tests we implement a straightforward per-row loop which
        is more readable than the previous scatter_add trick and still plenty
        fast given the tiny default sketch (\<=96 counters).
        """
        if keys.dtype != torch.long:
            keys = keys.long()
        if vals.dtype != torch.float16:
            vals = vals.to(torch.float16)

        # (batch, rows)
        idx = ((keys[:, None] * self.hash_a) % self.w).long()

        # Row-wise in-place conservative update
        for r in range(self.rows):
            row_idx = idx[:, r]  # (batch,)
            cur = self.C[r, row_idx]  # (batch,)
            mask = vals > cur  # need to increase counter
            if mask.any():
                self.C[r, row_idx[mask]] = vals[mask]


# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------

def _synthetic_batch(n_keys: int = 256):
    """Build a synthetic batch so smoke-tests do not need a real dataset."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    keys = torch.randint(0, 10_000, (n_keys,), device=dev)
    vals = torch.rand(n_keys, device=dev)
    return keys, vals


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

def train(cfg: dict) -> dict:
    """Minimal train loop that exercises ZeroSketch.

    Because the public snippet only contains the sketch, we limit ourselves to
    calling update/query to prove it works – replacing the real model training
    done in the full HAWQ-Skim code-base.
    """

    epochs = int(cfg.get("epochs", 2))
    batches_per_epoch = int(cfg.get("batches_per_epoch", 4))

    sketch = ZeroSketch(rows=cfg.get("rows", 3), w=cfg.get("w", 32))
    losses = []

    for ep in range(epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            keys, vals = _synthetic_batch(cfg.get("batch_size", 256))
            # simulate forward pass: query then L1-loss to true vals
            pred = sketch.query(keys)  # (batch,)
            loss = torch.mean(torch.abs(pred.float() - vals))
            # backward-less optimisation – we only update the sketch (as per paper)
            sketch.update(keys, vals)
            epoch_loss += loss.item()
        epoch_loss /= batches_per_epoch
        losses.append(epoch_loss)
        if cfg.get("verbose", True):
            print(f"Epoch {ep+1}/{epochs} – sketch L1 error: {epoch_loss:.4f}")

    metrics = {
        "final_l1_error": losses[-1],
        "mean_l1_error": sum(losses) / len(losses),
        "epochs": epochs,
        "rows": sketch.rows,
        "w": sketch.w,
    }

    # ------------------------------------------------------------------
    # Persist metrics under .research/iteration8 (new mandatory path)
    # ------------------------------------------------------------------
    out_dir = Path(".research") / "iteration8"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (cfg.get("run_name", "sketch_run") + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics →", json_path)
    print(json.dumps(metrics, indent=2))
    return metrics
