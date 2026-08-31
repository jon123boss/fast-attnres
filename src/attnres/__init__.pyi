import torch
from torch import Tensor

from .modules import LearnedQuery as LearnedQuery

__version__: str


def attnres(
    values: Tensor | list[Tensor] | tuple[Tensor, ...],
    query: Tensor,
    *,
    eps: float = ...,
    scale: float = ...,
) -> Tensor: ...


def reference_attnres(
    values: Tensor,
    query: Tensor,
    *,
    eps: float = ...,
    scale: float = ...,
    compute_dtype: torch.dtype = ...,
) -> Tensor: ...


__all__ = ["attnres", "reference_attnres", "LearnedQuery", "__version__"]
