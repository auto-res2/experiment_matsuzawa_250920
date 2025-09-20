import argparse
import os
from typing import Dict, Any

import yaml

from .preprocess import load_config
from .train import train_sketch
from .evaluate import evaluate_sketch

# ---------------------------------------------------------------------------
#   Helper for locating the YAML configuration files shipped with the repo
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config")
SMOKE_CFG = os.path.join(CONFIG_DIR, "smoke_test.yaml")
FULL_CFG  = os.path.join(CONFIG_DIR, "full_experiment.yaml")


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ZeroSketch smoke/full experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true", help="Run the lightweight CI / smoke test")
    group.add_argument("--full-experiment", action="store_true", help="Run the full-scale sketch benchmark")
    return parser.parse_args()


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            return yaml.safe_load(fp)
    except FileNotFoundError as e:
        raise RuntimeError(f"Config file not found: {path}") from e


def main() -> None:
    args = _parse_cli()
    cfg_path = SMOKE_CFG if args.smoke_test else FULL_CFG
    run_cfg = _load_yaml(cfg_path)

    # ------------------------------------------------------------------
    # The preprocess step is trivial here but keeps the directory layout
    # consistent with larger, multi-stage experiments.
    # ------------------------------------------------------------------
    cfg = load_config(run_cfg)

    metrics = train_sketch(cfg)
    evaluate_sketch(metrics)


if __name__ == "__main__":
    main()