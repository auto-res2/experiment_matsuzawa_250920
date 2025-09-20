"""Dataset loading and pre-processing utilities."""
import os
from typing import Tuple

import torch
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures


SUPPORTED_DATASETS = {
    "cora": lambda root: Planetoid(root, "Cora", transform=NormalizeFeatures()),
    "reddit": lambda root: Reddit(root, transform=NormalizeFeatures()),
}


def load_dataset(name: str) -> Tuple[torch_geometric.data.Data, int]:  # type: ignore[name-defined]
    name = name.lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {list(SUPPORTED_DATASETS)}")

    root = os.path.join(os.getenv("DATA_DIR", ".cache"), name)
    try:
        dataset = SUPPORTED_DATASETS[name](root)
    except Exception as exc:
        # Retry once in case of transient download issues (CI environments)
        print(f"[WARN] dataset download failed: {exc}. Retrying once …")
        dataset = SUPPORTED_DATASETS[name](root)
    data = dataset[0]
    num_features = dataset.num_node_features
    return data, num_features
