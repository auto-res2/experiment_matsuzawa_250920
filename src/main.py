import argparse
import os
from pathlib import Path
from typing import Dict

import torch
import yaml

from .train import load_backbone
from .preprocess import load_stream
from .evaluate import evaluate_stream

# -------------------------------------------------------------------------
# config loader
# -------------------------------------------------------------------------

def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


# -------------------------------------------------------------------------
# orchestrator
# -------------------------------------------------------------------------

def run_experiment(cfg: Dict, *, tag: str, smoke: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_backbone(cfg["model"], pretrained=True, device=device)

    dataloader = load_stream(
        dataset=cfg["dataset"],
        root=cfg["data_root"],
        batch_size=1,
        num_workers=cfg.get("num_workers", 4),
        smoke=smoke,
    )

    result_dir = Path(".research") / tag / "results"
    evaluate_stream(model, dataloader, device, result_dir=result_dir, tag=tag)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("UniCORN Experiment Runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Quick validation run")
    group.add_argument("--full-experiment", action="store_true", help="Full scale experiment")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    cfg_dir = base_dir / "config"

    if args.smoke_test:
        cfg_path = cfg_dir / "smoke_test.yaml"
        tag = "iteration1_smoke"
    else:
        cfg_path = cfg_dir / "full_experiment.yaml"
        tag = "iteration1_full"

    cfg = load_yaml(cfg_path)

    try:
        run_experiment(cfg, tag=tag, smoke=args.smoke_test)
    except Exception as exc:
        print(f"[Error] {exc}")
        raise


if __name__ == "__main__":
    main()
