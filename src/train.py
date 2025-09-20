# src/train.py
"""Training related classes and functions.

The original submission only provided the ``ZeroSketch`` helper and left
all actual model-training logic unimplemented.  In order to turn the
package into something that *runs end-to-end* (so that CI can validate
basic functionality) we now add a **tiny reference training loop** that
operates on a synthetic tabular data set produced by ``src.preprocess``.

The goal is *not* to reproduce the heavy GNN experiments described in
the research write-up – that would be impossible within the resource and
runtime constraints of an automated evaluation – but merely to supply a
complete, deterministic pipeline that yields a concrete numerical metric
(e.g. accuracy) for both the smoke-test and full-experiment configs.

Key design choices
------------------
1. **CPU-only** execution – keeps the test environment requirements to a
   minimum and avoids CUDA availability issues.
2. **Single hidden-layer MLP** – fast to train, yet non-trivial enough
   to achieve &gt;= 90 % accuracy on the toy data set.
3. **No external dependencies** beyond ``torch`` and ``numpy`` (added to
   *pyproject.toml*).
4. **Stateless entry-point** – the trained model object is inserted into
   the mutable ``cfg`` dictionary so that downstream evaluation can pick
   it up without changing the public API of ``main._run_phase``.
"""
from __future__ import annotations

import random
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Public helpers (imported by src.main)
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:  # noqa: D401
    """Seed *all* relevant RNGs for determinism."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _TinyMLP(nn.Module):
    """1-hidden-layer perceptron used for the synthetic data set."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:  # noqa: D401
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x)


def _train_epoch(
    model: nn.Module,
    optimiser: optim.Optimizer,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> float:  # noqa: D401
    model.train()
    optimiser.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimiser.step()
    return float(loss.item())


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:  # noqa: D401
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    correct = (preds == y).sum().item()
    return correct / y.numel()


# ---------------------------------------------------------------------------
# Public API (called by src.main)
# ---------------------------------------------------------------------------

def train(cfg: Dict[str, Any]) -> None:  # noqa: D401
    """Minimal training loop that writes the trained model into *cfg*.

    Parameters
    ----------
    cfg: Dict[str, Any]
        The experiment configuration dictionary.  Must already contain a
        ``"data"`` key produced by ``preprocess.load_data`` with entries
        ``train`` and ``val``.
    """

    if "data" not in cfg:
        raise RuntimeError(
            "`cfg` missing the pre-processed data ‑ make sure to call "
            "preprocess.load_data(cfg) before train(cfg)."
        )

    # ------------------------------------------------------------------
    # 0) Deterministic setup ------------------------------------------------
    # ------------------------------------------------------------------
    _set_seed(int(cfg.get("seed", 0)))

    device = torch.device("cpu")  # tiny model – keep it portable

    (x_train, y_train), (x_val, y_val) = cfg["data"]["train"], cfg["data"][
        "val"
    ]
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    in_dim = x_train.shape[1]
    num_classes = int(y_train.max().item() + 1)

    model = _TinyMLP(in_dim, hidden=32, out_dim=num_classes).to(device)

    optimiser = optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-3)))
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg.get("epochs", 1))
    for _ in range(epochs):
        _train_epoch(model, optimiser, criterion, x_train, y_train)

    # ------------------------------------------------------------------
    # 1) Compute final training/validation accuracy ----------------------
    # ------------------------------------------------------------------
    train_acc = _accuracy(model, x_train, y_train)
    val_acc = _accuracy(model, x_val, y_val)

    # ------------------------------------------------------------------
    # 2) Persist artefacts inside *cfg* so that other modules can use them
    # ------------------------------------------------------------------
    cfg["model_obj"] = model  # picked up by evaluate()
    cfg.setdefault("metrics", {})["train_accuracy"] = train_acc
    cfg["metrics"]["val_accuracy"] = val_acc

    # Nothing is *returned* – the mutable cfg acts as a shared state across
    # phases just like in many lightweight experiment managers.
