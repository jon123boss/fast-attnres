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


__all__ = ["LearnedQuery", "__version__", "attnres"]
