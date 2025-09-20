"""Entry-point orchestrating smoke-test and full experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import yaml

from .evaluate import IMAGES_DIR, save_results
from .preprocess import get_loader
from .train import build_model, run_one_stream


###############################################################################
#                               UTILITIES                                     #
###############################################################################

def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


###############################################################################
#                          EXPERIMENT  EXECUTION                              #
###############################################################################

def run_experiments(cfg: dict, smoke: bool):  # noqa: D401
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_cap = 128 if smoke else None

    for model_cfg in cfg["models"]:
        model_name: str = model_cfg["name"]
        quant: bool = model_cfg["quantized"]
        for seed in cfg["seed_list"]:
            _set_seed(seed)
            tag = f"{model_name}_{'int8' if quant else 'fp16'}_seed{seed}"
            print(f"\n=== Building model {tag} ===", flush=True)
            model = build_model(model_name, quant, device)

            global_stats = defaultdict(dict)
            for ds_name in cfg["datasets"]:
                loader = get_loader(
                    ds_name,
                    batch_size=cfg["batch_size"],
                    sample_cap=(cfg.get("num_batches") or 0) * cfg["batch_size"]
                    if cfg.get("num_batches") else sample_cap,
                )
                stats = run_one_stream(model, loader, device)
                global_stats[ds_name] = stats

            # ---------------- save & visualise ----------------
            results_file = save_results(tag, global_stats)
            print("-" * 64)
            print(f"Experiment results for '{tag}':")
            print(json.dumps(global_stats, indent=2))
            print(f"JSON saved → {results_file}")

            # quick bar-plot per dataset
            import matplotlib.pyplot as plt
            import seaborn as sns

            fig, ax = plt.subplots(figsize=(7, 3))
            sns.barplot(x=list(global_stats.keys()), y=[v["top1_error"] for v in global_stats.values()], ax=ax)
            for i, (k, v) in enumerate(global_stats.items()):
                ax.text(i, v["top1_error"] + 0.5, f"{v['top1_error']:.1f}", ha="center")
            ax.set_ylabel("Top-1 Error (%)")
            ax.set_title(f"Error across datasets – {tag}")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            pdf_path = IMAGES_DIR / f"barplot_{tag}.pdf"
            plt.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Figure saved → {pdf_path}")


###############################################################################
#                                    CLI                                      #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="run quick validation")
    parser.add_argument("--full-experiment", action="store_true", help="run full benchmark")
    args = parser.parse_args()

    if not (args.smoke_test ^ args.full_experiment):
        print("Must pass exactly one of --smoke-test or --full-experiment", file=sys.stderr)
        sys.exit(1)

    root = Path(__file__).resolve().parent.parent
    cfg_file = root / "config" / ("smoke_test.yaml" if args.smoke_test else "full_experiment.yaml")
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("=" * 72)
    print(f"Loaded configuration from {cfg_file}")
    print(json.dumps(cfg, indent=2))
    print("=" * 72)

    run_experiments(cfg, smoke=args.smoke_test)
