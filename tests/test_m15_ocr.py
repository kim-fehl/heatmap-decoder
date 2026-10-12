"""Opt-in real OCR and review regression, using the shipped publication-style example."""

import os
from pathlib import Path

import numpy as np
import pytest

from heatmap_decoder import DetectionHints, ExtractionConfig, detect, extract, load_raster
from heatmap_decoder.io import extract_file

pytestmark = [
    pytest.mark.ocr_integration,
    pytest.mark.skipif(os.getenv("HEATMAP_DECODER_OCR_TESTS") != "1", reason="opt-in local OCR"),
]
ROOT = Path(__file__).resolve().parents[1]


def test_real_ocr_preserves_both_plausible_signs_for_review():
    image = load_raster(ROOT / "examples/m15/figure.png")
    hints = DetectionHints.model_validate_json((ROOT / "examples/m15/hints.json").read_text())
    result = detect(image, hints)
    correct = next(
        r for r in result.label_evidence if r.text_rotation_degrees == 90 and r.side == "above"
    )
    assert correct.assignment is not None
    assert [s.raw_text for s in correct.assignment.slots] == [
        "Alpha Case-A",
        "Bravo Case-B",
        "Charlie Case-C",
        "Delta Case-D",
        "Echo Case-E",
        "Foxtrot Case-F",
        "Golf Case-G",
        "Hotel Case-H",
    ]
    assert correct.orientation_status == "ambiguous"
    assert not correct.count_geometry_eligible
    assert result.acceptance.overall.status != "accepted"


def test_reviewed_label_names_preserve_numeric_matrix(tmp_path):
    base = ROOT / "examples/m15"
    payload = ExtractionConfig.model_validate_json((base / "config.json").read_text())
    labelled = payload.model_copy(update={"labels": None})
    expected = extract(load_raster(base / "figure.png"), labelled)
    # Existing M13 explicit-angle path remains the reference for reviewed names.
    from heatmap_decoder.models import LabelsConfig

    labelled = payload.model_copy(
        update={
            "labels": LabelsConfig(
                columns={
                    "source": "ocr",
                    "bbox": (107, 165, 1247, 291),
                    "text_rotation_degrees": 90,
                    "recognition": "slots",
                },
                ocr={"upscale_factor": 2},
            )
        }
    )
    config = tmp_path / "config.json"
    config.write_text(labelled.model_dump_json())
    actual = extract_file(
        base / "figure.png",
        labelled,
        tmp_path / "out",
        config_path=config,
        labels_review=base / "reviewed_labels.tsv",
    )
    assert np.array_equal(expected.values, actual.values)
