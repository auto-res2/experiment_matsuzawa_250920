# src/evaluate.py
"""Simple evaluation helper.

The original refactor contained only a stub.  We now provide a minimal
implementation that works with the synthetic data set generated in
``src.preprocess`` and the small MLP trained in ``src.train``.  The
function returns a metrics dict so that ``src.main`` can serialise it
into the research artefact JSON file.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def evaluate(cfg: Dict[str, Any]) -> Dict[str, float]:  # noqa: D401
    """Run inference on the held-out *test* split and return metrics."""

    if "data" not in cfg or "model_obj" not in cfg:
        raise RuntimeError(
            "Evaluation requires both the data split and a trained model.  "
            "Make sure to call preprocess.load_data and train.train first."
        )

    (x_test, y_test) = cfg["data"]["test"]
    model = cfg["model_obj"].to(torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        preds = model(x_test).argmax(dim=1)
    correct = (preds == y_test).sum().item()
    acc = correct / y_test.numel()

    # Attach to cfg for downstream introspection
    cfg.setdefault("metrics", {})["test_accuracy"] = acc

    return {"test_accuracy": acc}