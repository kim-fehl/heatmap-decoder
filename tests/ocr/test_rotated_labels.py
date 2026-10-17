"""Rotated labels contracts and fixtures."""

import json

import numpy as np
import pytest

from examples.m13.generate import CLEAN_STRINGS, fixture
from heatmap_decoder.extract import extract, extract_labels
from heatmap_decoder.io import extract_file, load_raster
from heatmap_decoder.models import ExtractionConfig
from tests.support import ROOT as REPO_ROOT


@pytest.mark.parametrize("angle", [90, -90])
@pytest.mark.parametrize("placement", ["above", "below"])
@pytest.mark.parametrize("mode", ["slots", "sparse"])
@pytest.mark.parametrize("engine", ["tesseract", "rapidocr"])
def test_rotated_geometry_and_preserved_engine_readings(angle, placement, mode, engine):
    image, payload, _ = fixture(angle, placement, strings=CLEAN_STRINGS)
    payload["labels"]["columns"]["recognition"] = mode
    payload["labels"]["ocr"] = {"engine": engine}
    result = extract_labels(np.asarray(image), ExtractionConfig.model_validate(payload))
    columns = result.axes[1]
    if engine == "tesseract":
        assert [s.raw_text for s in columns.slots] == CLEAN_STRINGS
        assert [s.final_text for s in columns.slots] == CLEAN_STRINGS
    else:
        # PP-OCR may drop internal spaces; preserve that observation rather
        # than repairing it from expected strings. Exact spelling is scored
        # separately from the geometry contract and retains human review.
        assert len(columns.slots) == len(CLEAN_STRINGS)
        assert all(s.raw_text and s.final_text == s.raw_text for s in columns.slots)
    assert result.unresolved_count == 0
    assert not columns.unmatched_candidate_indices
    assert all(
        40 + s.index * 64 <= s.bbox[0] < s.bbox[2] <= 40 + (s.index + 1) * 64 for s in columns.slots
    )


def test_reviewed_rotated_example_unchanged_values(tmp_path):
    root = REPO_ROOT / "examples/m13"
    config = ExtractionConfig.model_validate_json((root / "config.json").read_text())
    baseline = extract(load_raster(root / "figure.png"), config.model_copy(update={"labels": None}))
    result = extract_file(
        root / "figure.png",
        config,
        tmp_path / "reviewed",
        labels_review=root / "reviewed_labels.tsv",
    )
    truth = json.loads((root / "truth.json").read_text())
    assert [s.final_text for s in result.labels.axes[1].slots] == truth["columns"]
    np.testing.assert_array_equal(result.values, baseline.values)
    assert np.isfinite(result.values).all()
    assert np.max(np.abs(result.values - truth["values"])) < 0.01
