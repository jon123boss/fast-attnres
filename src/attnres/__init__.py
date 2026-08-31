"""Attention Residuals with full-width values and sliced tail keys."""

# Keep the version available from a source checkout as well as an installed
# distribution.  The project version in ``pyproject.toml`` is intentionally
# kept in sync with this value.
__version__ = "1.0.0"

from .api import attnres
from .modules import LearnedQuery
from .reference import reference_attnres

__all__ = [
    "attnres",
    "reference_attnres",
    "LearnedQuery",
    "__version__",
]
