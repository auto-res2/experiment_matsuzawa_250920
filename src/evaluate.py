import json
import os
from typing import Dict, Any

import torch

__all__ = ["evaluate_sketch"]


def evaluate_sketch(metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics to STDOUT and save a JSON copy for later inspection."""

    # ---------------------------- STDOUT summary -----------------------------
    print("\n===== ZeroSketch Evaluation =====")
    for k, v in metrics.items():
        print(f"{k:>12}: {v}")
    print("================================\n")

    # ----------------------- persist results for CI run ----------------------
    out_dir = os.path.join(".research", "iteration3")
    os.makedirs(out_dir, exist_ok=True)
    file_name = f"result_sketch_{metrics['n_keys']}.json"
    path = os.path.join(out_dir, file_name)

    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
    except IOError as e:
        print(f"[WARN] Could not write metrics JSON → {e}")