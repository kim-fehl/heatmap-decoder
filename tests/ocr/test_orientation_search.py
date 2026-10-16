"""Orientation search contracts and fixtures."""

import numpy as np
import pytest

from heatmap_decoder import DetectionHints, ExtractionConfig, detect, extract, load_raster
from heatmap_decoder.io import extract_file
from tests.support import ROOT as REPO_ROOT

ROOT = REPO_ROOT


@pytest.mark.parametrize(
    "engine,language",
    [("tesseract", "eng"), ("rapidocr", "eng"), ("rapidocr", "chinese")],
)
def test_real_ocr_preserves_both_plausible_signs_for_review(engine, language):
    image = load_raster(ROOT / "examples/m15/figure.png")
    hints = DetectionHints.model_validate_json((ROOT / "examples/m15/hints.json").read_text())
    hints = hints.model_copy(
        update={
            "labels": hints.labels.model_copy(
                update={
                    "ocr": hints.labels.ocr.model_copy(
                        update={"engine": engine, "language": language}
                    )
                }
            )
        }
    )
    result = detect(image, hints)
    correct = next(
        r for r in result.label_evidence if r.text_rotation_degrees == 90 and r.side == "above"
    )
    assert correct.assignment is not None
    expected = [
        "Alpha Case-A",
        "Bravo Case-B",
        "Charlie Case-C",
        "Delta Case-D",
        "Echo Case-E",
        "Foxtrot Case-F",
        "Golf Case-G",
        "Hotel Case-H",
    ]
    if language == "eng":
        assert [s.raw_text for s in correct.assignment.slots] == expected
    else:
        assert len(correct.assignment.slots) == len(expected)
        assert all(s.raw_text for s in correct.assignment.slots)
    if language == "eng":
        assert correct.orientation_status == "ambiguous"
        assert not correct.count_geometry_eligible
        competing = next(
            r for r in result.label_evidence if r.text_rotation_degrees == -90 and r.side == "above"
        )
        assert competing.orientation_status == "ambiguous"
        assert not competing.count_geometry_eligible
        assert "contradictory_orientation_or_side_hypotheses" in correct.orientation_reasons
    else:
        # The legacy Chinese model supplies only one adequate geometric angle;
        # the English model reads all upright names but retains the competing flip.
        assert correct.orientation_status == "unique"
        assert correct.count_geometry_eligible
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
