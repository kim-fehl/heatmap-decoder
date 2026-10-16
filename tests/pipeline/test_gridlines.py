"""Gridlines contracts and fixtures."""

import numpy as np
import pytest
from PIL import Image, ImageFilter

from heatmap_decoder import AutomationRequest, DetectionHints, ExtractionConfig, automate, extract
from heatmap_decoder.detection import refine_box
from heatmap_decoder.lattice import observe_axis
from tests.support.synthetic import palette_fixture


def fixture(variant="clean", grid=True):
    return palette_fixture(
        dict(
            palette="blue_white_red",
            layout=1 if grid else 0,
            seed=811030,
            variant=variant,
            negative=None,
        )
    )


def request(truth):
    config = truth["config"]
    return AutomationRequest(
        hints=DetectionHints(
            rows=24,
            columns=12,
            requested=("heatmap", "colourbar"),
            colourbar_start_value=config["colourbar_start_value"],
            colourbar_end_value=config["colourbar_end_value"],
        ),
        numeric_only=True,
    )


@pytest.mark.parametrize("variant", ["clean", "jpeg", "blur"])
def test_gridline_geometry_and_values_survive_raster_degradation(variant):
    image, truth = fixture(variant)
    result = automate(image, request(truth), experimental_export=True)
    decisions = {d.requirement: d for d in result.decisions}
    if variant != "blur":
        assert decisions["heatmap"].status == "accepted"
    else:
        assert not any(
            "phase" in reason or "peaks" in reason for reason in decisions["heatmap"].reasons
        )
    assert decisions["colourbar"].status == "accepted"
    config = result.diagnostic_config
    if config is None:
        config = ExtractionConfig.model_validate(truth["config"])
    assert config.heatmap_bbox == truth["config"]["heatmap_bbox"]
    values = extract(image, config)
    finite = np.isfinite(values.values)
    assert np.max(np.abs(values.values[finite] - np.array(truth["values"])[finite])) / 8 < 0.05
    if variant == "clean":
        assert result.automatic_export_allowed


def test_final_gridline_kept_but_external_frame_removed():
    image, truth = fixture()
    box = truth["config"]["heatmap_bbox"]
    assert refine_box(image, box, 0, 12) == box
    x, y, r, s = box
    framed = image.copy()
    framed[y - 1 : s + 1, x - 1] = framed[y - 1 : s + 1, r] = 0
    framed[y - 1, x - 1 : r + 1] = framed[s, x - 1 : r + 1] = 0
    assert refine_box(framed, (x - 1, y - 1, r + 1, s + 1), 0, 12) == box


def test_blurred_stroke_width_and_centre_uncertainty_are_distinct():
    image, truth = fixture("blur")
    axis = observe_axis(image, truth["config"]["heatmap_bbox"], "rows")
    assert len(axis.transitions) == 23
    assert np.max(np.abs(np.array(axis.transitions) - np.arange(96, 464, 16))) <= 1
    assert max(axis.transition_widths) > 3
    assert max(axis.transition_uncertainties) < 1


@pytest.mark.parametrize("damage", ["irregular", "wide_strokes", "strong_blur", "wrong_counts"])
def test_uncertain_geometry_remains_review_required(damage):
    image, truth = fixture()
    x, y, r, s = truth["config"]["heatmap_bbox"]
    options = request(truth)
    if damage == "irregular":
        # Move alternating transitions by four pixels, preserving sharp strokes.
        for k in range(1, 24, 2):
            at = y + 16 * k
            image[at - 1 : at + 4, x:r] = image[at - 2, x:r]
            image[at + 3, x:r] = 0
    elif damage == "wide_strokes":
        for at in range(y + 16, s, 16):
            image[at - 4 : at + 4, x:r] = 0
    elif damage == "strong_blur":
        image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(3)))
    else:
        options = options.model_copy(
            update={"hints": options.hints.model_copy(update={"rows": 23})}
        )
    result = automate(image, options, experimental_export=True)
    assert result.complete.status != "accepted"
    assert not result.automatic_export_allowed


def test_smooth_within_cell_structure_is_not_explained_as_pixel_noise():
    image, truth = fixture(grid=False)
    x, y, r, s = truth["config"]["heatmap_bbox"]
    rgb = image[y:s, x:r].astype(float)
    ramp = (np.arange(s - y) % 16 - 7.5) * 6
    image[y:s, x:r] = np.clip(rgb + ramp[:, None, None], 0, 255).astype(np.uint8)
    result = automate(image, request(truth), experimental_export=True)
    assert result.complete.status != "accepted"
