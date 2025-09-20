import json
import os
from datetime import datetime
from typing import Dict, Tuple

import torch

from .train import test, ECOGAT

# ----------------------------------------------------------------------------
# Evaluation utilities
# ----------------------------------------------------------------------------

def evaluate_and_log(model: ECOGAT, data, device: torch.device, run_name: str) -> Dict[str, float]:
    """Runs test(), prints to stdout, dumps JSON under .research directory."""
    train_acc, val_acc, test_acc = test(model, data, device)
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "run_name": run_name,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
    }

    # Always print to STDOUT for CI visibility
    print(json.dumps(results, indent=2))

    # Persist to disk for later inspection
    out_dir = os.path.join(".research", "iteration2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    return results
