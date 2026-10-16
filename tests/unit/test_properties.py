"""Properties contracts and fixtures."""

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from heatmap_decoder import ExtractionConfig
from heatmap_decoder.calibration import calibrate_values
from heatmap_decoder.geometry import cell_slices, validate_geometry


@given(
    rows=st.integers(1, 15),
    columns=st.integers(1, 15),
    cell_height=st.integers(6, 30),
    cell_width=st.integers(6, 30),
    inset=st.floats(min_value=0.1, max_value=0.25),
)
def test_interior_pixels_stay_in_their_cells(rows, columns, cell_height, cell_width, inset):
    height, width = rows * cell_height, columns * cell_width
    cfg = ExtractionConfig(
        heatmap_bbox=(3, 5, 3 + width, 5 + height),
        colourbar_bbox=(3 + width, 5, 7 + width, 15),
        rows=rows,
        columns=columns,
        sampling_inset_fraction=inset,
        colourbar_orientation="vertical",
        colourbar_start_value=-2,
        colourbar_end_value=2,
    )
    validate_geometry(cfg, (max(5 + height, 15), 7 + width, 3))
    slices = cell_slices(cfg)
    assert len(slices) == rows * columns
    for (ys, xs), (row, column) in zip(slices, np.ndindex(rows, columns), strict=True):
        assert 5 + row * cell_height < ys.start < ys.stop < 5 + (row + 1) * cell_height
        assert 3 + column * cell_width < xs.start < xs.stop < 3 + (column + 1) * cell_width


@given(
    start=st.floats(min_value=-1000, max_value=1000),
    span=st.floats(min_value=0.01, max_value=1000),
    ascending=st.booleans(),
    count=st.integers(2, 1000),
)
def test_calibration_preserves_direction_and_endpoints(start, span, ascending, count):
    end = start + span if ascending else start - span
    values = calibrate_values(count, start, end)
    assert values[0] == start and values[-1] == end
    assert np.isfinite(values).all()
    assert np.all(np.diff(values) > 0) if ascending else np.all(np.diff(values) < 0)
