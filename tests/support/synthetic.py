"""Canonical synthetic constructors shared across test tiers.

Each constructor creates fresh image/configuration data. Keep the original
example and benchmark generators unchanged so frozen evidence remains usable.
"""

from benchmarks.m14.fixtures import PALETTES as PALETTES
from benchmarks.m14.fixtures import fixture as palette_fixture
from examples.m10.generate import layouts as layout_cases
from examples.m10.generate import make_layout as layout_fixture
from examples.m11.generate import make_fixture as legend_fixture
from examples.m12.generate import cases as scale_cases
from examples.m12.generate import make_fixture as scale_fixture

__all__ = [
    "PALETTES",
    "layout_cases",
    "layout_fixture",
    "legend_fixture",
    "palette_fixture",
    "scale_cases",
    "scale_fixture",
]
