"""Deprecated command alias; no separate CLI implementation."""

import warnings

from heatmap_decoder.cli import app


def legacy_main() -> None:
    warnings.warn("heatmap2matrix is deprecated; use heatmap-decoder", FutureWarning, stacklevel=2)
    app(prog_name="heatmap-decoder")
