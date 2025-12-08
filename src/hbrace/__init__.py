"""Core package for hierarchical breast cancer response modeling."""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("hbrace")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
