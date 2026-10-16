"""Reconstruction contracts and fixtures."""

import numpy as np
import pytest
from PIL import Image

from heatmap_decoder import ExtractionConfig, extract, load_raster, write_csv


def palette(t, kind):
    t = np.asarray(t)[..., None]
    if kind == "sequential":
        rgb = np.concatenate((30 + 200 * t, 15 + 160 * t, 80 + 120 * t), axis=-1)
    else:
        blue, white, red = (
            np.array([25, 60, 210]),
            np.array([245, 245, 245]),
            np.array([210, 35, 25]),
        )
        rgb = np.where(t <= 0.5, blue + 2 * t * (white - blue), white + (2 * t - 1) * (red - white))
    return np.rint(rgb).astype(np.uint8)


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("kind", ["sequential", "diverging"])
@pytest.mark.parametrize("cell_size", [4, 9, 16])
def test_synthetic_reconstruction(tmp_path, orientation, kind, cell_size):
    positions = np.random.default_rng(2026).uniform(0, 1, (20, 20))
    positions[0, :3] = [0, 0.5, 1]
    matrix_rgb = palette(positions, kind)
    side, bar_length = 20 * cell_size, 512
    image = np.full((max(side, bar_length) + 30,) * 2 + (3,), 255, dtype=np.uint8)
    image[10 : 10 + side, 10 : 10 + side] = np.repeat(
        np.repeat(matrix_rgb, cell_size, axis=0), cell_size, axis=1
    )
    bar = palette(np.linspace(0, 1, bar_length), kind)
    if orientation == "vertical":
        bar_box = (side + 15, 10, side + 25, 10 + bar_length)
        image[10 : 10 + bar_length, side + 15 : side + 25] = bar[:, None, :]
    else:
        bar_box = (10, side + 15, 10 + bar_length, side + 25)
        image[side + 15 : side + 25, 10 : 10 + bar_length] = bar[None, :, :]
    # Exercise both ascending and descending numerical scales.
    start, end = (2.0, -2.0) if orientation == "vertical" else (-2.0, 2.0)
    expected = start + positions * (end - start)
    config = ExtractionConfig(
        heatmap_bbox=(10, 10, 10 + side, 10 + side),
        colourbar_bbox=bar_box,
        rows=20,
        columns=20,
        colourbar_orientation=orientation,
        colourbar_start_value=start,
        colourbar_end_value=end,
    )
    path = tmp_path / "reference.png"
    Image.fromarray(image).save(path)
    loaded = load_raster(path)
    result = extract(loaded, config)
    errors = np.abs(result.values - expected)
    mae, maximum = float(errors.mean()), float(errors.max())
    print(
        f"{kind}/{orientation}/{cell_size}px: MAE={mae:.6f}, max={maximum:.6f}, "
        f"MAE/range={mae / abs(end - start):.3%}; flagged={np.mean(result.flags != 0):.3%}"
    )
    assert not np.any(result.flags)
    assert mae < 0.02 * abs(end - start)
    assert maximum < 0.02 * abs(end - start)
    np.testing.assert_array_equal(result.values, extract(loaded, config).values)
    np.testing.assert_array_equal(loaded, image)
    csv_path = tmp_path / "matrix.csv"
    write_csv(result, csv_path)
    np.testing.assert_array_equal(np.loadtxt(csv_path, delimiter=","), result.values)
