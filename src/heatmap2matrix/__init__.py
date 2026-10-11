"""Deprecated v0.1 import alias. Implementation lives in heatmap_decoder."""

import importlib
import sys
import warnings

from heatmap_decoder import *  # noqa: F403
from heatmap_decoder import __all__, __version__  # noqa: F401

warnings.warn(
    "heatmap2matrix is deprecated; import heatmap_decoder instead",
    DeprecationWarning,
    stacklevel=2,
)
for _module in (
    "calibration",
    "diagnostics",
    "extract",
    "geometry",
    "io",
    "models",
    "sampling",
    "version",
):
    sys.modules[f"{__name__}.{_module}"] = importlib.import_module(f"heatmap_decoder.{_module}")
