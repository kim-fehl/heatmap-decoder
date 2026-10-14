"""Evaluation-only coordinate tolerance; counts and group membership remain exact."""

import math
from typing import Any

from heatmap_decoder.geometry import resolve_grid
from heatmap_decoder.models import GridConfig


def compare_grid_geometry(
    actual: GridConfig, expected: GridConfig, tolerance_pixels: float = 1.5
) -> dict[str, Any]:
    if not math.isfinite(tolerance_pixels) or tolerance_pixels < 0:
        raise ValueError("coordinate tolerance must be finite and nonnegative")
    a, e = resolve_grid(actual), resolve_grid(expected)
    axes = {}
    for name in ("rows", "columns"):
        aa, ee = getattr(a, name), getattr(e, name)
        counts_match = len(aa.slots) == len(ee.slots)
        groups_match = aa.ownership == ee.ownership and len(aa.gaps) == len(ee.gaps)
        errors = (
            [
                abs(x - y)
                for sa, se in zip(aa.slots, ee.slots)
                for x, y in ((sa.start, se.start), (sa.end, se.end))
            ]
            if counts_match
            else []
        )
        maximum = max(errors, default=0.0) if counts_match else None
        axes[name] = dict(
            counts_match=counts_match,
            groups_match=groups_match,
            maximum_boundary_error_pixels=maximum,
            within_tolerance=counts_match
            and groups_match
            and maximum is not None
            and maximum <= tolerance_pixels,
        )
    return dict(
        tolerance_pixels=tolerance_pixels,
        axes=axes,
        within_tolerance=all(axis["within_tolerance"] for axis in axes.values()),
        exact=all(
            axis["within_tolerance"] and axis["maximum_boundary_error_pixels"] == 0
            for axis in axes.values()
        ),
    )
