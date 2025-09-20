# src/train.py
"""Training related classes and functions extracted from the original
script.  The *only* piece of runnable logic that was provided inside the
"Experiment Code" section is the `ZeroSketch` implementation; nothing
else (optimizer, model, training‐loop, etc.) was supplied.  Consequently,
this file contains that very class **verbatim** plus thin helper stubs
so that it can be imported from other modules without raising
`ImportError`.

If further training utilities existed in the original monolithic script,
they would have been moved here as well.  Because the source did **not**
provide any additional code, we intentionally keep the surface minimal
and do **not** invent new logic – this honours the
"refactor-but-do-not-extend" requirement.
"""
from __future__ import annotations

import os
from typing import Any

import torch


class ZeroSketch:
    """Count-min style sketch with *zero* per-edge updates (on-the-fly
    hashing).  Copied verbatim from the monolithic script.

    Parameters
    ----------
    rows: int, default 3
        Number of hash rows.
    w: int, default 32
        Hash table width per row.
    """

    def __init__(self, rows: int = 3, w: int = 32) -> None:  # noqa: D401
        self.w = w
        self.rows = rows

        # Large random seeds stored as 64-bit ints so we can multiply with
        # 32-bit node IDs without overflow.
        self.hash_a = torch.randint(
            1, 2 ** 31, (rows,), device="cuda", dtype=torch.int64
        )

        # Global accumulator shared across *all* nodes in a layer.
        # We keep it in fp16 – this matches the original code and reduces
        # memory traffic.
        self.C = torch.zeros(rows, w, device="cuda", dtype=torch.float16)

    def query(self, keys: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Return the *minimum* counter across all rows for each key.

        keys: 1-D int64 tensor containing node IDs or hashes.
        returns: fp16 tensor with per-key counts.
        """
        if keys.dtype != torch.int64:
            keys = keys.to(torch.int64)
        idx = ((keys[:, None] * self.hash_a) % self.w).long()
        return torch.min(self.C[:, idx].transpose(0, 1), dim=1).values

    @torch.cuda.amp.autocast(dtype=torch.float16)
    def update(self, keys: torch.Tensor, vals: torch.Tensor) -> None:  # noqa: D401
        """Conservative update (atomic min) – *zero* per-edge writes.

        Only elements that would increase the sketch counter are written
        back.  This is precisely the logic from the original snippet.
        """
        if keys.dtype != torch.int64:
            keys = keys.to(torch.int64)
        idx = ((keys[:, None] * self.hash_a) % self.w).long()
        cur = self.C[:, idx].transpose(0, 1)
        need = (vals.unsqueeze(1) > cur).float()
        # scatter_add over the 2-nd dim (width) as in the source code.
        self.C.scatter_add_(1, (need * idx).long(), need * vals.unsqueeze(1))


# ---------------------------------------------------------------------------
# Optional thin façade so that the rest of the project can import a generic
# `train` symbol even though the real training loop was *not* supplied.
# ---------------------------------------------------------------------------

def train(*_args: Any, **_kwargs: Any) -> None:  # noqa: D401
    """Placeholder that raises to indicate missing logic from the source.

    The monolithic script we received did *not* contain any training
    function – only the `ZeroSketch` helper class.  To keep the package
    importable without inventing new functionality, we raise an
    informative error here.
    """

    raise NotImplementedError(
        "The original Experiment Code did not include a training loop.  "
        "If you possess the missing parts, please place them into "
        "src/train.py so they can be called from src/main.py."
    )