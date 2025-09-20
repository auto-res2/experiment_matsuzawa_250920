from typing import Tuple
from torch_geometric.datasets import Planetoid, Reddit
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.data import Data
from pathlib import Path

__all__ = ["load_dataset"]


def load_dataset(name: str, root: Path) -> Data:
    if name.lower() in {"cora", "citeseer", "pubmed"}:
        ds = Planetoid(str(root), name, transform=NormalizeFeatures())
        return ds[0]
    elif name.lower() == "reddit":
        ds = Reddit(str(root), transform=NormalizeFeatures())
        data = ds[0]
        # convert to undirected for consistency
        from torch_geometric.utils import to_undirected
        data.edge_index = to_undirected(data.edge_index)
        return data
    else:
        raise ValueError(f"Unsupported dataset {name}")
