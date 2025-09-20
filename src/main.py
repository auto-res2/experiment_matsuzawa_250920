import argparse
import sys
from pathlib import Path
import yaml

# relative intra-package imports
from . import preprocess as _pre
from . import train as _train
from . import evaluate as _eval

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_cfg(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        print(f"Config file not found: {path}")
        raise e


def _run(cfg: dict):
    _pre.load_data(cfg)  # no-op stub
    metrics = _train.train(cfg)
    _eval.evaluate(metrics, cfg)


def main(argv=None):
    parser = argparse.ArgumentParser(description="HAWQ-Skim experiment orchestrator")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="Run quick smoke-test config")
    g.add_argument("--full-experiment", action="store_true", help="Run full experimental config")
    args = parser.parse_args(argv)

    if args.smoke_test:
        cfg_path = _CONFIG_DIR / "smoke_test.yaml"
    else:
        cfg_path = _CONFIG_DIR / "full_experiment.yaml"

    cfg = _load_cfg(cfg_path)
    try:
        _run(cfg)
    except KeyboardInterrupt:
        print("Aborted by user", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
