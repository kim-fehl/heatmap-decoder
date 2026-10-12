"""Opt-in M13 real-engine tests; spelling and geometric assignment are separate."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.m13.generate import CLEAN_STRINGS, fixture  # noqa: E402
from heatmap_decoder.extract import extract, extract_labels
from heatmap_decoder.io import extract_file, load_raster
from heatmap_decoder.models import ExtractionConfig

pytestmark = [
    pytest.mark.ocr_integration,
    pytest.mark.skipif(
        os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1",
        reason="opt in with HEATMAP_DECODER_OCR_TESTS=1",
    ),
]


@pytest.mark.parametrize("angle", [90, -90])
@pytest.mark.parametrize("placement", ["above", "below"])
@pytest.mark.parametrize("mode", ["slots", "sparse"])
def test_clean_supported_strings_in_displayed_order(angle, placement, mode):
    image, payload, _ = fixture(angle, placement, strings=CLEAN_STRINGS)
    payload["labels"]["columns"]["recognition"] = mode
    result = extract_labels(np.asarray(image), ExtractionConfig.model_validate(payload))
    columns = result.axes[1]
    assert [s.raw_text for s in columns.slots] == CLEAN_STRINGS
    assert [s.final_text for s in columns.slots] == CLEAN_STRINGS
    assert result.unresolved_count == 0
    assert not columns.unmatched_candidate_indices
    assert all(
        40 + s.index * 64 <= s.bbox[0] < s.bbox[2] <= 40 + (s.index + 1) * 64 for s in columns.slots
    )


def test_reviewed_rotated_example_unchanged_values(tmp_path):
    root = Path(__file__).resolve().parents[1] / "examples/m13"
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
