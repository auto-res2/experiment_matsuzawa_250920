# src/preprocess.py
"""Data loading / pre-processing extracted from the original script.

The monolithic Experiment Code did not include any explicit
pre-processing routines.  To respect the instruction of *not* generating
novel logic*, this file only contains a stub that raises an explanatory
exception if called.
"""
from __future__ import annotations

from typing import Any


def load_data(*_args: Any, **_kwargs: Any) -> None:  # noqa: D401
    raise NotImplementedError(
        "Data loading was not included in the provided Experiment Code."
    )