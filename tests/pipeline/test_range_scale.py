"""Expression-range extraction, uncertainty, and CLI output contracts."""

import copy
import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder.cli import app
from heatmap_decoder.models import CellFlag
from heatmap_decoder.range_scale import RangeExtractionRequest, extract_ranges


def range_fixture():
    image = np.full((40, 40, 3), 255, np.uint8)
    colours = [[200, 200, 200], [60, 140, 220], [0, 0, 0]]
    for i, colour in enumerate(colours):
        image[0:6, i * 10 : (i + 1) * 10] = colour
        image[20:40, i * 10 : (i + 1) * 10] = colour
    request = {
        "quantity": "TPM",
        "position_config": {
            "heatmap_bbox": [0, 20, 30, 40],
            "rows": 1,
            "columns": 3,
            "colourbar_bbox": [0, 0, 30, 6],
            "colourbar_orientation": "horizontal",
            "colourbar_start_value": 0,
            "colourbar_end_value": 1,
        },
        "bands": [
            {"label": "below cutoff", "start": 0, "end": 10 / 29, "lower": 0, "upper": 0.5},
            {"label": "low", "start": 10 / 29, "end": 20 / 29, "lower": 0.5, "upper": 11},
            {"label": "high", "start": 20 / 29, "end": 1, "lower": 11, "upper": None},
        ],
    }
    return image, request


def test_plateau_within_one_range_is_resolved_without_a_point_value():
    image, payload = range_fixture()
    result = extract_ranges(image, RangeExtractionRequest.model_validate(payload))
    assert result["complete"]
    assert not result["point_values_recovered"] and not result["statistical_intervals"]
    assert [r["range"]["label"] for r in result["records"]] == ["below cutoff", "low", "high"]
    assert all(r["positional_flags"] & CellFlag.AMBIGUOUS for r in result["records"])
    assert result["records"][-1]["range"]["upper"] is None
    assert "value" not in result["records"][-1]
    assert result["records"][0]["range"]["upper_inclusive"] is False


@pytest.mark.parametrize("kind", ["cross_range", "gap", "poor", "alpha"])
def test_uncertainty_is_preserved_instead_of_assigning_a_range(kind):
    image, payload = range_fixture()
    expected = {
        "cross_range": "range_ambiguous",
        "gap": "uncovered_range",
        "poor": "poor_colour_match",
        "alpha": "missing",
    }[kind]
    if kind == "cross_range":
        image[0:6, 20:30] = image[0:6, 0:10]
    elif kind == "gap":
        payload["bands"][0]["start"] = 2 / 29
    elif kind == "poor":
        image[20:40, 0:10] = [0, 255, 0]
    else:
        image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
        image[20:40, 0:10, 3] = 0
    result = extract_ranges(image, RangeExtractionRequest.model_validate(payload))
    assert not result["complete"]
    assert result["records"][0]["status"] == expected
    assert result["records"][0]["range"] is None
    if kind == "cross_range":
        assert result["records"][0]["candidate_ranges"] == ["below cutoff", "high"]


@pytest.mark.parametrize("kind", ["overlap", "bounds", "scale", "blank", "duplicate"])
def test_invalid_range_requests_fail(kind):
    _, payload = range_fixture()
    if kind == "overlap":
        payload["bands"][1]["start"] = 0.1
    elif kind == "bounds":
        payload["bands"][0]["upper"] = -1
    elif kind == "scale":
        payload["position_config"]["colourbar_end_value"] = 4000
    elif kind == "blank":
        payload["bands"][0]["label"] = " "
    else:
        payload["bands"][1]["label"] = payload["bands"][0]["label"]
    with pytest.raises(ValueError):
        RangeExtractionRequest.model_validate(payload)


def test_range_cli_outputs_overwrite_and_incomplete_status(tmp_path):
    image, payload = range_fixture()
    figure, request, out = tmp_path / "figure.png", tmp_path / "request.json", tmp_path / "out"
    Image.fromarray(image).save(figure)
    request.write_text(json.dumps(payload))
    args = ["extract-ranges", str(figure), "--request", str(request), "--out", str(out)]
    runner = CliRunner()
    assert runner.invoke(app, args).exit_code == 0
    assert json.loads((out / "ranges.json").read_text())["input_sha256"]
    assert "lower\tupper" in (out / "ranges.tsv").read_text()
    assert runner.invoke(app, args).exit_code == 2
    changed = copy.deepcopy(payload)
    changed["bands"][0]["start"] = 0.1
    request.write_text(json.dumps(changed))
    assert runner.invoke(app, args + ["--overwrite"]).exit_code == 4
    assert not json.loads((out / "ranges.json").read_text())["complete"]
    # An output path aliasing either input is protected even with overwrite.
    request.write_text(json.dumps(payload))
    protected = tmp_path / "protected"
    protected.mkdir()
    (protected / "ranges.json").symlink_to(request)
    assert runner.invoke(app, args[:-1] + [str(protected), "--overwrite"]).exit_code == 2
    assert json.loads(request.read_text()) == payload


def test_vertical_bar_and_segmented_cells_reuse_shared_geometry():
    image, payload = range_fixture()
    image = image.transpose(1, 0, 2).copy()
    config = payload["position_config"]
    config.update(
        heatmap_bbox=[20, 0, 40, 30],
        rows=3,
        columns=1,
        rows_segments=[{"start": 0, "end": 10, "count": 1}, {"start": 20, "end": 30, "count": 2}],
        colourbar_bbox=[0, 0, 6, 30],
        colourbar_orientation="vertical",
    )
    result = extract_ranges(image, RangeExtractionRequest.model_validate(payload))
    assert result["complete"]
    assert [r["range"]["label"] for r in result["records"]] == ["below cutoff", "high", "high"]
