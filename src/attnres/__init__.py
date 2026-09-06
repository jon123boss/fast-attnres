"""CUDA BF16 Attention Residuals with full-width values and sliced tail keys."""

# Keep the version available from a source checkout as well as an installed
# distribution.  The project version in ``pyproject.toml`` is intentionally
# kept in sync with this value.
__version__ = "2.0.1"

from .api import attnres
from .modules import LearnedQuery

__all__ = [
    "LearnedQuery",
    "__version__",
    "attnres",
]
