"""Core contracts and fixtures."""

import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from heatmap_decoder import ExtractionConfig, extract, load_raster
from heatmap_decoder.calibration import calibrate_values, match_colours
from heatmap_decoder.geometry import cell_slices, validate_geometry
from heatmap_decoder.sampling import sample_cells, sample_colourbar


def config(**changes):
    values = dict(
        heatmap_bbox=(0, 0, 8, 8),
        colourbar_bbox=(8, 0, 12, 8),
        rows=2,
        columns=2,
        colourbar_orientation="vertical",
        colourbar_start_value=2.0,
        colourbar_end_value=-2.0,
    )
    values.update(changes)
    return ExtractionConfig(**values)


@pytest.mark.parametrize(
    "changes",
    [
        {"rows": 0},
        {"columns": -1},
        {"rows": True},
        {"rows": 1.5},
        {"heatmap_bbox": (-1, 0, 8, 8)},
        {"heatmap_bbox": (0, 0, 0, 8)},
        {"heatmap_bbox": (0, 0, 1.5, 8)},
        {"sampling_inset_fraction": 0.5},
        {"sampling_inset_fraction": -0.1},
        {"colourbar_orientation": "diagonal"},
        {"colourbar_end_value": float("nan")},
        {"colourbar_end_value": float("inf")},
        {"colourbar_end_value": 2.0},
        {"unknown": 1},
    ],
)
def test_invalid_configuration(changes):
    with pytest.raises(ValidationError):
        config(**changes)


def test_json_configuration():
    original = config()
    assert ExtractionConfig.model_validate_json(original.model_dump_json()) == original


def test_geometry():
    cfg = config()
    validate_geometry(cfg, (8, 12, 3))  # right/bottom at image limits are valid
    assert cell_slices(cfg) == [
        (slice(1, 3), slice(1, 3)),
        (slice(1, 3), slice(5, 7)),
        (slice(5, 7), slice(1, 3)),
        (slice(5, 7), slice(5, 7)),
    ]
    with pytest.raises(ValueError, match="inside"):
        validate_geometry(cfg, (8, 11, 3))
    with pytest.raises(ValueError, match="one pixel"):
        validate_geometry(config(columns=9), (8, 12, 3))
    with pytest.raises(ValueError, match="two positions"):
        validate_geometry(config(colourbar_bbox=(8, 0, 12, 1)), (8, 12, 3))


def test_fractional_geometry_and_empty_sampling():
    assert cell_slices(config(heatmap_bbox=(0, 0, 7, 7)))[-1] == (slice(4, 6), slice(4, 6))
    with pytest.raises(ValueError, match="no pixel centres"):
        cell_slices(
            config(heatmap_bbox=(0, 0, 2, 2), rows=1, columns=1, sampling_inset_fraction=0.49)
        )


def test_cell_sampling_excludes_borders_and_uses_median():
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    for ys, xs in cell_slices(config()):
        image[ys, xs] = [30, 80, 120]
        image[ys.start, xs.start] = 255  # one corrupted interior pixel
    expected = np.tile([30, 80, 120], (4, 1))
    np.testing.assert_array_equal(sample_cells(image, config()), expected)


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_bar_sampling_direction_and_central_strip(orientation):
    gradient = np.repeat(np.arange(8, dtype=np.uint8)[:, None] * 30, 3, axis=1)
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    if orientation == "vertical":
        box = (8, 0, 12, 8)
        image[:8, 9:11] = gradient[:, None]
    else:
        box = (0, 8, 8, 12)
        image[9:11, :8] = gradient[None]
    cfg = config(colourbar_bbox=box, colourbar_orientation=orientation)
    np.testing.assert_array_equal(sample_colourbar(image, cfg), gradient)


def test_calibration_and_perceptual_match():
    np.testing.assert_array_equal(calibrate_values(3, 2, -2), [2, 0, -2])
    np.testing.assert_array_equal(calibrate_values(3, -2, 2), [-2, 0, 2])
    bar = np.array([[0, 0, 255], [240, 240, 240], [255, 0, 0]], dtype=float)
    cells = np.array([[250, 3, 0], [0, 0, 255], [238, 241, 239]], dtype=float)
    np.testing.assert_array_equal(match_colours(cells, bar), [2, 0, 1])
    with pytest.raises(ValueError):
        calibrate_values(1, 0, 1)
    with pytest.raises(ValueError):
        match_colours(cells * 2, bar)


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((8, 12)),
        np.zeros((8, 12, 5), dtype=np.uint8),
        np.zeros((0, 12, 3), dtype=np.uint8),
    ],
)
def test_invalid_in_memory_raster(image):
    with pytest.raises(ValueError):
        extract(image, config())


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_raster_loading(tmp_path, image_format):
    path = tmp_path / "image"
    Image.new("RGB", (12, 8), (20, 30, 40)).save(path, format=image_format)
    loaded = load_raster(path)
    assert loaded.shape == (8, 12, 3)
    assert loaded.dtype == np.uint8


def test_unsupported_and_transparent_raster(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGBA", (2, 2), (255, 0, 0, 0)).save(path)
    assert load_raster(path).shape == (2, 2, 4)
    Image.new("RGB", (2, 2)).save(path, format="BMP")
    with pytest.raises(ValueError, match="PNG and JPEG"):
        load_raster(path)


def test_no_ocr_dependency_is_imported_by_core_or_geometry():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import heatmap_decoder.labels; assert 'pytesseract' not in sys.modules",
        ],
        check=True,
    )
