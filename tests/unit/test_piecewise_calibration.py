"""Piecewise calibration contracts and fixtures."""

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from heatmap_decoder.calibration import calibrate_values, calibration_evidence, piecewise_values
from heatmap_decoder.diagnostic_style import DiagnosticCanvas
from heatmap_decoder.gradient import inspect_gradient
from heatmap_decoder.models import ExtractionConfig, OCRConfig, ScaleConfig
from tests.support.piecewise_calibration import anchors


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
@pytest.mark.parametrize("reverse", [False, True])
def test_piecewise_original_coordinates(orientation, reverse):
    box = (10, 0, 111, 8) if orientation == "horizontal" else (0, 10, 8, 111)
    knots = anchors(values=(5, 1, 0) if reverse else (0, 1, 5))
    values = piecewise_values(knots, box, orientation, 101)
    assert values[0] == knots[0].value
    assert values[30] == knots[1].value
    assert values[-1] == knots[-1].value
    assert values[65] == pytest.approx((knots[1].value + knots[2].value) / 2)
    e = calibration_evidence(
        ScaleConfig(model="piecewise_linear", source="manual_ticks", anchors=knots),
        box,
        orientation,
    )
    assert e["full_span"]
    assert not e["linear_comparison"]["within_residual_limits"]


@pytest.mark.parametrize(
    "positions,values",
    [
        ((10, 10, 110), (0, 1, 5)),
        ((40, 10, 110), (0, 1, 5)),
        ((9, 40, 110), (0, 1, 5)),
        ((10, 40, 111), (0, 1, 5)),
        ((10, 40, 110), (0, 0, 5)),
        ((10, 40, 110), (0, 5, 1)),
    ],
)
def test_invalid_knots(positions, values):
    with pytest.raises(ValueError):
        piecewise_values(anchors(positions, values), (10, 0, 111, 8), "horizontal", 101)


def test_dense_gradient_does_not_smooth_narrow_transition():
    image = np.full((20, 120, 3), 120, dtype=np.uint8)
    image[:, 40] = (255, 0, 0)
    c = ExtractionConfig(
        heatmap_bbox=(0, 10, 20, 20),
        rows=1,
        columns=1,
        colourbar_bbox=(10, 0, 111, 8),
        colourbar_orientation="horizontal",
        colourbar_start_value=0,
        colourbar_end_value=5,
    )
    e = inspect_gradient(image, c, (10, 40, 110))
    assert not e["axial_aggregation"]
    point = next(p for p in e["checkpoints"] if p["position"] == 40)
    assert point["rgb"] == [255, 0, 0]
    assert point["footprint"] == [40, 2, 41, 6]


def test_model_and_confidence_validation():
    with pytest.raises(ValidationError):
        ScaleConfig(model="piecewise_linear", source="manual_endpoints")
    with pytest.raises(ValidationError):
        ScaleConfig(model="piecewise_linear", source="manual_ticks", anchors=anchors()[:2])
    with pytest.raises(ValidationError):
        OCRConfig(min_confidence=95, high_confidence=90)
    with pytest.raises(ValidationError):
        OCRConfig(retry_variants=("contrast", "contrast"))
    np.testing.assert_array_equal(calibrate_values(101, -2, 2), np.linspace(-2, 2, 101))


def test_confidence_status_are_independent():
    canvas = DiagnosticCanvas(Image.new("RGB", (100, 100), "black"))
    canvas.roi((5, 5, 30, 30), "heatmap", "manual")
    canvas.ocr((40, 40, 60, 60), 98, "T0", "wrong sign", "ambiguous")
    assert canvas.image.getpixel((40, 40)) == (0, 126, 128)
    canvas.ocr((70, 70, 90, 90), None, "S0", "manual name", "manual")
    assert canvas.image.getpixel((70, 70)) == (102, 102, 102)


def test_subpixel_confidence_frame():
    canvas = DiagnosticCanvas(Image.new("RGB", (100, 100), "white"))
    canvas.ocr((5, 5, 5.33, 5.66), 30, "T0", ".", "raw")
    assert canvas.image.getpixel((5, 5)) == (197, 38, 71)
