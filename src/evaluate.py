import json
import os
from typing import Dict, Any

import torch

__all__ = ["evaluate_sketch"]


def evaluate_sketch(metrics: Dict[str, Any]) -> None:
    """Pretty-print metrics and persist them under .research/iteration5/.

    All CI artefacts *must* be written to `.research/iteration5/` – the
    grading harness asserts the path.  The JSON file is re-opened and
    printed back to STDOUT for immediate visual confirmation.
    """

    # ---------------------------- STDOUT summary -----------------------------
    print("\n===== ZeroSketch Evaluation =====")
    for k, v in metrics.items():
        print(f"{k:>12}: {v}")
    print("================================\n")

    # ----------------------- persist results for CI run ----------------------
    out_dir = os.path.join(".research", "iteration5")
    os.makedirs(out_dir, exist_ok=True)
    file_name = f"result_sketch_{metrics['n_keys']}.json"
    path = os.path.join(out_dir, file_name)

    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
    except IOError as e:
        raise RuntimeError(f"Could not write metrics JSON → {e}") from e

    # ---------- re-open & print the JSON so the runner can verify -------------
    with open(path, "r", encoding="utf-8") as fp:
        loaded = json.load(fp)
    print("Persisted metrics:")
    print(json.dumps(loaded, indent=2))