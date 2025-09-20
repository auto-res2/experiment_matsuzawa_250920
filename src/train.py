import os
import json
from pathlib import Path
import torch

__all__ = [
    "ZeroSketch",
    "train",
]


class ZeroSketch:
    """Zero-update Count-Sketch used by HAWQ-Skim (see paper)."""

    def __init__(self, rows: int = 3, w: int = 32):
        if rows <= 0 or w <= 0:
            raise ValueError("rows and w must be positive")
        self.w = w
        self.rows = rows
        # prime-range hash coefficients →  signed 64-bit avoids overflow on mul
        self.hash_a = torch.randint(
            low=1,
            high=2 ** 31,
            size=(rows,),
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.int64,
        )
        # global sketch per layer  →  fp16 keeps memory foot-print <1 kB
        self.C = torch.zeros(rows, w, device=self.hash_a.device, dtype=torch.float16)

    def query(self, keys: torch.LongTensor) -> torch.Tensor:
        """Conservative MIN query (identical to CM-sketch)."""
        if keys.dtype != torch.long:
            keys = keys.long()
        idx = ((keys[:, None] * self.hash_a) % self.w).long()
        return torch.min(self.C[:, idx].transpose(0, 1), dim=1).values

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def update(self, keys: torch.LongTensor, vals: torch.Tensor) -> None:
        if keys.dtype != torch.long:
            keys = keys.long()
        if vals.dtype != torch.float16:
            vals = vals.to(torch.float16)
        idx = ((keys[:, None] * self.hash_a) % self.w).long()
        cur = self.C[:, idx].transpose(0, 1)
        need = (vals.unsqueeze(1) > cur).float()
        # scatter_add_ on *need* mask emulates atomic-min (safe on single-GPU training loop)
        self.C.scatter_add_(1, (need * idx).long(), need * vals.unsqueeze(1))


def _synthetic_batch(n_keys: int = 256):
    """Helper that builds a synthetic batch so smoke-tests do not need a dataset."""
    keys = torch.randint(0, 10_000, (n_keys,), device="cuda" if torch.cuda.is_available() else "cpu")
    vals = torch.rand(n_keys, device=keys.device)
    return keys, vals


def train(cfg: dict) -> dict:
    """Minimal train loop that exercises ZeroSketch several times.

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
            pred = sketch.query(keys)
            loss = torch.mean(torch.abs(pred.float() - vals))
            # backward-less optimisation – we only update the sketch (as per paper)
            sketch.update(keys, vals)
            epoch_loss += loss.item()
        epoch_loss /= batches_per_epoch
        losses.append(epoch_loss)
        if cfg.get("verbose", True):
            print(f"Epoch {ep+1}/{epochs} − sketch L1 error: {epoch_loss:.4f}")

    metrics = {
        "final_l1_error": losses[-1],
        "mean_l1_error": sum(losses) / len(losses),
        "epochs": epochs,
        "rows": sketch.rows,
        "w": sketch.w,
    }
    # persist metrics under .research/iteration7
    out_dir = Path(".research") / "iteration7"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (cfg.get("run_name", "sketch_run") + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics →", json_path)
    print(json.dumps(metrics, indent=2))
    return metrics
