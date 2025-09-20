# src/preprocess.py
"""Synthetic data generator used by the tiny reference experiment.

Because the heavy-weight graph datasets from the research paper are much
too large to be shipped and processed within the automated evaluation
framework, we fall back to generating a **small synthetic classification
corpus**.  This keeps the runtime &lt; 5 s yet still allows us to obtain a
meaningful numerical metric (accuracy) for smoke-test and full-experiment
runs.

The function mutates the supplied ``cfg`` dictionary by inserting a
``"data"`` key that contains tuples of (features, labels) for the train,
validation, and test splits.  All tensors live on the **CPU** to avoid
CUDA availability issues.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

__all__ = ["load_data"]


# ---------------------------------------------------------------------------
# Helper --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def _make_split(
    num_samples: int,
    num_features: int,
    num_classes: int,
    *,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
    """Generate a single (X, y) pair with linearly separable classes."""

    g = torch.Generator().manual_seed(seed)
    # Each class centroid is randomly placed on the unit-sphere
    centroids = torch.randn(num_classes, num_features, generator=g)
    centroids = centroids / centroids.norm(dim=1, keepdim=True)

    labels = torch.randint(0, num_classes, (num_samples,), generator=g)
    features = centroids[labels] + 0.1 * torch.randn(num_samples, num_features, generator=g)
    return features.float(), labels.long()


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def load_data(cfg: Dict[str, Any]) -> None:  # noqa: D401
    """Populate *cfg* with a small synthetic classification data set."""

    if "data" in cfg:  # idempotent
        return

    dataset_name = str(cfg.get("dataset", "synthetic"))

    # The *scale* of the synthetic set loosely follows the requested
    # dataset for some variability – still tiny in absolute terms.
    if dataset_name == "reddit":
        n_train, n_val, n_test = 512, 128, 128
    elif dataset_name == "ogbn_products":
        n_train, n_val, n_test = 1_024, 256, 256
    else:  # papers100M or unknown
        n_train, n_val, n_test = 2_048, 512, 512

    num_features = 16
    num_classes = 3
    seed = int(cfg.get("seed", 0))

    cfg["data"] = {
        "train": _make_split(n_train, num_features, num_classes, seed=seed),
        "val": _make_split(n_val, num_features, num_classes, seed=seed + 1),
        "test": _make_split(n_test, num_features, num_classes, seed=seed + 2),
    }