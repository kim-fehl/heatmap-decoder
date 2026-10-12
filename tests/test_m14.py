"""Palette-independent ambiguity evidence, structural masking and acceptance negatives."""

import csv
import importlib.util
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from heatmap_decoder import AutomationRequest, DetectionHints, ExtractionConfig, automate, extract
from heatmap_decoder.calibration import match_with_quality
from heatmap_decoder.detection import colourbar_candidates, refine_box
from heatmap_decoder.io import write_outputs
from heatmap_decoder.lattice import observe_axis

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = runpy.run_path(str(ROOT / "benchmarks/m14/fixtures.py"))


def case(palette="teal_black_red", layout=0, negative=None):
    return dict(palette=palette, layout=layout, negative=negative, seed=773100, variant="clean")


def request(config):
    return AutomationRequest(
        hints=DetectionHints(
            rows=config.rows,
            columns=config.columns,
            requested=("heatmap", "colourbar"),
            colourbar_start_value=config.colourbar_start_value,
            colourbar_end_value=config.colourbar_end_value,
        ),
        numeric_only=True,
    )


def test_components_distinguish_contiguous_and_disconnected_without_relaxing_flags():
    bar = np.array([[0, 0, 0], [0, 0, 0], [255, 255, 255], [0, 0, 0]], float)
    matched = match_with_quality(np.zeros((1, 3)), bar, ambiguity_colour_tolerance=0)
    assert matched.candidate_intervals == (((0, 1), (3, 3)),)
    assert matched.ambiguous[0]
    plateau = match_with_quality(np.zeros((1, 3)), np.zeros((20, 3)), ambiguity_colour_tolerance=0)
    assert plateau.candidate_intervals == (((0, 19),),)
    assert plateau.ambiguous[0]


@pytest.mark.parametrize("direction", [1, -1])
def test_actual_calibrated_scale_components_and_json_csv(direction, tmp_path):
    image = np.full((50, 90, 3), 255, np.uint8)
    image[5:15, 5:15] = 0
    image[25:35, 30:70] = 0
    image[25:35, 45:55] = 255
    cfg = ExtractionConfig(
        heatmap_bbox=(5, 5, 15, 15),
        rows=1,
        columns=1,
        colourbar_bbox=(30, 25, 70, 35),
        colourbar_orientation="horizontal",
        scale={
            "source": "manual_ticks",
            "anchors": [
                {"position": 30, "value": direction * 100},
                {"position": 69, "value": direction * 490},
            ],
        },
        ambiguity_colour_tolerance=0,
    )
    result = extract(image, cfg)
    assert np.isnan(result.values[0, 0])
    evidence = result.colour_evidence[0]
    assert evidence.kind == "disconnected_near_matches"
    assert [(c.first_index, c.last_index) for c in evidence.components] == [(0, 14), (25, 39)]
    for component in evidence.components:
        first = direction * (100 + 10 * component.first_index)
        last = direction * (100 + 10 * component.last_index)
        assert component.minimum_value == pytest.approx(min(first, last))
        assert component.maximum_value == pytest.approx(max(first, last))
        assert component.first_position == pytest.approx(component.first_index / 39)
    assert evidence.value_span == pytest.approx(390)
    write_outputs(result, tmp_path, input_filename="synthetic.png", input_sha256="x")
    metadata = json.loads((tmp_path / "provenance.json").read_text())
    assert metadata["quality"]["components_are_statistical_intervals"] is False
    with (tmp_path / "quality.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert json.loads(row["colour_candidate_evidence"])["kind"] == evidence.kind


@pytest.mark.parametrize("palette", FIXTURES["PALETTES"])
@pytest.mark.parametrize("layout", [0, 3])
def test_clean_manual_palette_accuracy_and_black_centre(palette, layout):
    if (
        palette in ("viridis", "magma", "cividis")
        and importlib.util.find_spec("matplotlib") is None
    ):
        pytest.skip("named LUT fixtures use benchmark-only Matplotlib")
    image, truth = FIXTURES["fixture"](case(palette, layout))
    config = ExtractionConfig.model_validate(truth["config"])
    result = extract(image, config)
    finite = np.isfinite(result.values)
    error = np.abs(result.values[finite] - np.array(truth["values"])[finite]) / 8
    assert error.mean() < 0.02
    assert error.max() < 0.05
    if palette != "asymmetric":
        assert finite.all() and not result.flags.any()
    if palette == "teal_black_red":
        assert result.values[0, 1] == pytest.approx(1, abs=0.05)
        assert result.valid_fraction[0, 1] == 1


def test_broad_dark_band_preserved_while_thin_grid_stroke_excluded():
    # All true row transition evidence lives in a broad dark-valued column.
    image = np.full((100, 100, 3), 200, np.uint8)
    for i, v in enumerate((5, 18, 32, 45)):
        image[10 + i * 20 : 30 + i * 20, 10:60] = v
    observed = observe_axis(image, (10, 10, 90, 90), "rows")
    assert np.allclose(observed.transitions, (30, 50, 70))
    assert observed.strokes == ()
    image[10:90, 75:76] = 0  # a thin perpendicular stroke is removable
    lined = observe_axis(image, (10, 10, 90, 90), "rows")
    assert np.allclose(lined.transitions, observed.transitions)


def test_full_dark_sample_columns_are_not_trimmed_as_outer_borders():
    image, truth = FIXTURES["fixture"](case(negative="dark_band"))
    config = ExtractionConfig.model_validate(truth["config"])
    hints = request(config).hints
    assert (
        refine_box(image, config.heatmap_bbox, 0, hints.detector.background_distance)
        == config.heatmap_bbox
    )
    result = automate(image, request(config), experimental_export=True)
    assert result.complete.status == "accepted"
    assert result.automatic_export_allowed
    values = extract(image, result.diagnostic_config)
    assert np.max(np.abs(values.values - np.array(truth["values"]))) / 8 < 0.05


@pytest.mark.parametrize("palette,layout", [("blue_white_red", 0), ("teal_black_red", 3)])
def test_neutral_centre_fragments_do_not_create_false_competing_bars(palette, layout):
    image, truth = FIXTURES["fixture"](case(palette, layout))
    config = ExtractionConfig.model_validate(truth["config"])
    audit = []
    bars = colourbar_candidates(image, request(config).hints, audit)
    assert len(bars) == 1
    assert bars[0].bbox == config.colourbar_bbox
    assert any(item.get("rejection") == "continued_core_fragment" for item in audit)
    result = automate(image, request(config), experimental_export=True)
    assert result.complete.status == "accepted"


@pytest.mark.parametrize(
    "negative",
    ["repeated", "plateau", "categorical", "competing_bars", "off_bar", "quantised", "clipped"],
)
def test_negative_requests_never_export(negative):
    if importlib.util.find_spec("matplotlib") is None and negative not in ("repeated", "plateau"):
        pytest.skip("benchmark named LUT dependency")
    palette = negative if negative in ("repeated", "plateau") else "viridis"
    image, truth = FIXTURES["fixture"](case(palette, negative=negative))
    config = ExtractionConfig.model_validate(truth["config"])
    result = automate(image, request(config), experimental_export=True)
    assert result.complete.status != "accepted"
    assert not result.automatic_export_allowed
