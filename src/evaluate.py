import json
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

__all__ = ["evaluate"]


def evaluate(model: torch.nn.Module, data: Data, results_path: Path) -> Dict[str, Any]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
        for i, layer in enumerate(model):
            x = layer(x, edge_index)
            if i != len(model) - 1:
                x = F.elu(x)
        pred = x.argmax(dim=-1)
        acc = (pred[data.test_mask] == y[data.test_mask]).float().mean().item()

    metrics = {"test_accuracy": acc}
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fp:
        json.dump(metrics, fp, indent=2)
    print("=== Evaluation results ===")
    print(json.dumps(metrics, indent=2))
    return metrics
