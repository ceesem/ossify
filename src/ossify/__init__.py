import importlib as _importlib

from . import algorithms, plot, structured_prediction
from .base import *
from .file_io import *
from .translate import *

__version__ = "0.1.2"


def __getattr__(name):
    """Lazy-import the optional 3D plotting submodule with a clear error.

    ``ossify.plot3d`` depends on pyvista, which is an optional extra. We
    don't import it at package load time so that ``import ossify`` keeps
    working in environments without pyvista; instead, we defer the import
    until first access and re-raise a clear message pointing at the extra.

    ``importlib.import_module`` is used here rather than ``from . import
    plot3d`` because the latter would re-enter this ``__getattr__`` while
    resolving the ``plot3d`` attribute on the parent package, causing
    infinite recursion.
    """
    if name == "plot3d":
        try:
            return _importlib.import_module(".plot3d", __name__)
        except ImportError as e:
            raise ImportError(
                "ossify.plot3d requires pyvista. Install with "
                "`pip install ossify[viz]` (or `uv add 'ossify[viz]'`)."
            ) from e
    if name == "compartments":
        # Module imports cleanly without xgboost; xgboost is only needed at
        # predict/load time (with a clear error). Lazy to mirror the optional extra.
        return _importlib.import_module(".compartments", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
