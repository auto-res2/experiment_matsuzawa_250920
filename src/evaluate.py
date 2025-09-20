# src/evaluate.py
"""Evaluation & analysis utilities extracted from the original script.

No dedicated evaluation code was provided inside the supplied
monolithic file; only the `ZeroSketch` helper existed.  In order to keep
import paths functional while *not* adding new business logic, we expose
an evaluator stub that purposefully raises `NotImplementedError` when
used.  This is consistent with the refactor-only mandate – we refrain
from creating new evaluation metrics that were **not** in the original
script.
"""
from __future__ import annotations

from typing import Any


def evaluate(*_args: Any, **_kwargs: Any) -> None:  # noqa: D401
    raise NotImplementedError(
        "Evaluation logic was not present in the original Experiment Code."
    )