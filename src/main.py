import argparse
import yaml
from pathlib import Path

from .preprocess import load_dataset
from .train import build_model, train
from .evaluate import evaluate

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
RESEARCH_DIR = Path(__file__).resolve().parent.parent / ".research" / "iteration1"


def run(cfg_path: Path):
    with open(cfg_path) as fp:
        cfg = yaml.safe_load(fp)

    # ---------------- data -----------------
    data = load_dataset(cfg["dataset"], Path(cfg.get("data_root", "./data")))

    # -------------- model ------------------
    model = build_model(
        in_channels=data.num_features,
        hidden=cfg["model"]["hidden"],
        num_classes=cfg["model"]["num_classes"],
        num_layers=cfg["model"]["num_layers"],
        heads=cfg["model"].get("heads", 4),
        solver_cfg=cfg["solver"],
    )

    # -------------- train ------------------
    train_results = train(model, data, cfg["optim"],
                          RESEARCH_DIR / f"result_{cfg['run_name']}_train.json")

    # ------------- evaluate ---------------
    evaluate(model, data, RESEARCH_DIR / f"result_{cfg['run_name']}_eval.json")


def main():
    parser = argparse.ArgumentParser(description="ECO-GAT experiment runner")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="run quick smoke test")
    g.add_argument("--full-experiment", action="store_true", help="run full experiment")
    args = parser.parse_args()

    if args.smoke_test:
        cfg_file = CONFIG_DIR / "smoke_test.yaml"
    else:
        cfg_file = CONFIG_DIR / "full_experiment.yaml"

    run(cfg_file)


if __name__ == "__main__":
    main()
