import json
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Callable, Dict

import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize

__all__ = [
    'evaluate_stream',
]


@torch.no_grad()
def evaluate_stream(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    *,
    result_dir: Path,
    tag: str,
):
    """Run streaming evaluation (batch = 1) and save JSON metrics."""
    model.to(device)
    top1_correct = 0
    n_seen = 0
    latencies_ms = []

    softmax = torch.nn.Softmax(dim=1)

    for sample in dataloader:
        img = sample["image"].to(device, non_blocking=True)
        label = sample["label"].to(device)

        t0 = perf_counter_ns()
        logits = model(img)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = perf_counter_ns()
        latencies_ms.append((t1 - t0) / 1e6)

        pred = logits.argmax(1)
        top1_correct += (pred == label).sum().item()
        n_seen += img.size(0)

    acc = 100.0 * top1_correct / max(1, n_seen)
    metrics: Dict[str, float] = {
        "accuracy_top1": acc,
        "num_samples": n_seen,
        "latency_mean_ms": sum(latencies_ms) / len(latencies_ms),
    }

    # ------------------------------------------------------------------
    # persist & echo
    # ------------------------------------------------------------------
    result_file = result_dir / f"{tag}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"[Evaluation ‑ {tag}]", json.dumps(metrics, indent=2))
