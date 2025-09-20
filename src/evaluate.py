import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------
# directory layout required by the spec
# ---------------------------------------------------------------------
BASE_RESEARCH_DIR = Path(".research") / "iteration1"
IMAGES_DIR = BASE_RESEARCH_DIR / "images"
RESULTS_DIR = BASE_RESEARCH_DIR
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
def _annotate_line(ax):
    for line in ax.lines:
        xs, ys = line.get_xdata(), line.get_ydata()
        for x, y in zip(xs, ys):
            ax.text(x, y, f"{y:.1f}", fontsize=6)


def plot_accuracy_curve(acc_hist: Dict[int, float], tag: str) -> Path:
    """Simple line-plot helper – returns path of saved pdf."""

    fig, ax = plt.subplots(figsize=(6, 3))
    xs, ys = list(acc_hist.keys()), list(acc_hist.values())
    sns.lineplot(x=xs, y=ys, ax=ax, marker="o", label="Top-1 Error (%)")
    _annotate_line(ax)
    ax.set_xlabel("Batch #")
    ax.set_ylabel("Error %")
    ax.set_title("Online Stream Accuracy")
    ax.legend()
    pdf_path = IMAGES_DIR / f"accuracy_{tag}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def save_results(exp_name: str, stats: Dict) -> Path:
    json_path = RESULTS_DIR / f"{exp_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return json_path
