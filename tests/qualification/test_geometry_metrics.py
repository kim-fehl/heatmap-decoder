"""Geometry metrics contracts and fixtures."""

import pytest

from heatmap_decoder.models import GridConfig
from tests.support import ROOT as REPO_ROOT
from tests.support import load_script

compare_grid_geometry = load_script(str(REPO_ROOT / "benchmarks/geometry_metrics.py"))[
    "compare_grid_geometry"
]


def grid(end=40, start=60, counts=(4, 4)):
    return GridConfig(
        heatmap_bbox=(0, 0, 50, 100),
        columns=5,
        rows_segments=(
            {"start": 0, "end": end, "count": counts[0]},
            {"start": start, "end": 100, "count": counts[1]},
        ),
    )


def test_one_and_half_pixel_jitter_is_tolerated_but_still_reported():
    actual, expected = grid(41.5, 61.5), grid()
    result = compare_grid_geometry(actual, expected)
    assert result["within_tolerance"] and not result["exact"]
    assert result["axes"]["rows"]["maximum_boundary_error_pixels"] == 1.5
    assert not compare_grid_geometry(actual, expected, 1.0)["within_tolerance"]
    assert compare_grid_geometry(expected, expected, 0)["exact"]


def test_counts_and_group_membership_must_remain_exact():
    assert not compare_grid_geometry(grid(counts=(5, 4)), grid())["within_tolerance"]
    result = compare_grid_geometry(grid(counts=(3, 5)), grid())
    assert result["axes"]["rows"]["counts_match"]
    assert not result["axes"]["rows"]["groups_match"]
    assert not result["within_tolerance"]


@pytest.mark.parametrize("tolerance", [-1, float("inf"), float("nan")])
def test_invalid_tolerances_are_rejected(tolerance):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_grid_geometry(grid(), grid(), tolerance)
