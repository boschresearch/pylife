import sys

from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "pylife"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError


from .core import PylifeSignal, Broadcaster, DataValidator

__all__ = [
    'PylifeSignal',
    'Broadcaster',
    'DataValidator'
]
