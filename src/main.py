import argparse
import os
from pathlib import Path

import torch
import yaml

from .preprocess import load_dataset
from .train import ECOGAT, train
from .evaluate import evaluate_and_log

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
SMOKE_CFG = CONFIG_DIR / "smoke_test.yaml"
FULL_CFG = CONFIG_DIR / "full_experiment.yaml"


def load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def run_experiment(cfg: dict, run_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data -----------------------------------------------------------------
    data, num_features = load_dataset(cfg["dataset"])
    data = data.to(device)

    # 2. Model ----------------------------------------------------------------
    model = ECOGAT(
        in_channels=num_features,
        hidden_channels=cfg.get("hidden_channels", 64),
        out_channels=int(data.y.max().item() + 1),
        heads=cfg.get("heads", 4),
        flop_budget=cfg.get("flop_budget", 0.2),
        q_err_budget=cfg.get("quant_err_budget", 0.02),
    ).to(device)

    # 3. Optimiser ------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 0.005), weight_decay=5e-4)

    # 4. Training loop --------------------------------------------------------
    for epoch in range(1, cfg.get("epochs", 1) + 1):
        loss = train(model, data, optimizer, device)
        if epoch % max(1, cfg.get("log_every", 1)) == 0:
            print(f"Epoch {epoch:03d} | loss = {loss:.4f}")

    # 5. Evaluation & Logging -------------------------------------------------
    evaluate_and_log(model, data, device, run_name)


# -----------------------------------------------------------------------------
# CLI entry-point
# -----------------------------------------------------------------------------

def cli():
    parser = argparse.ArgumentParser(description="ECO-GAT experiment runner")
    mgroup = parser.add_mutually_exclusive_group(required=True)
    mgroup.add_argument("--smoke-test", action="store_true", help="run the lightweight smoke test")
    mgroup.add_argument("--full-experiment", action="store_true", help="run the full experiment")
    args = parser.parse_args()

    # Resolve configuration ---------------------------------------------------
    if args.smoke_test:
        cfg_path = SMOKE_CFG
        phase = "smoke"
    else:
        cfg_path = FULL_CFG
        phase = "full"

    cfg = load_cfg(cfg_path)

    # Phase 1: always run smoke-test first when full experiment is requested
    if phase == "full":
        print("[INFO] Running preliminary smoke test …")
        run_experiment(load_cfg(SMOKE_CFG), run_name="smoke_test")
        print("[INFO] Smoke test completed ✅ – proceeding to full experiment …")

    run_experiment(cfg, run_name=phase)


if __name__ == "__main__":
    cli()
