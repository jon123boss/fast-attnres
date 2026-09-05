"""Small trainable building blocks for CUDA BF16 Attention Residuals."""

from __future__ import annotations

import math
import operator

import torch
from torch import nn


def _require_index(value, name: str) -> int:
    """Accept integer-like dimensions without truncating other numbers."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


class LearnedQuery(nn.Module):
    """A static trainable query vector."""

    def __init__(self, rank: int, *, init_std: float = 0.02) -> None:
        super().__init__()
        rank = _require_index(rank, "rank")
        init_std = float(init_std)
        if rank < 1:
            raise ValueError("rank must be positive")
        if not math.isfinite(init_std) or init_std <= 0:
            raise ValueError("init_std must be finite and positive")
        self.query = nn.Parameter(torch.empty(rank, dtype=torch.bfloat16))
        nn.init.normal_(self.query, mean=0.0, std=init_std)

    def forward(self) -> torch.Tensor:
        """Return a BF16 query while preserving gradients to the parameter."""
        return self.query.to(dtype=torch.bfloat16)


__all__ = ["LearnedQuery"]
