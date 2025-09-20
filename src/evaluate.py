import json
from pathlib import Path

__all__ = ["evaluate"]


def evaluate(metrics: dict, cfg: dict) -> None:
    """Simple evaluation that echoes metrics and marks pass/fail."""

    thresh = cfg.get("target_l1", 0.25)
    status = "PASS" if metrics["final_l1_error"] <= thresh else "FAIL"

    report = {
        "status": status,
        "threshold": thresh,
        "final_l1_error": metrics["final_l1_error"],
    }

    # save side-car report – mandatory .research/iteration8 path
    out_dir = Path(".research") / "iteration8"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (cfg.get("run_name", "sketch_run") + "_eval.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # always print to stdout as requested
    print("Evaluation report →", path)
    print(json.dumps(report, indent=2))
