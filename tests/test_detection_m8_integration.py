"""Opt-in M8 region review using the fixed M6 upright OCR/numeric reference."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from heatmap_decoder import DetectionHints, extract_file, load_raster
from heatmap_decoder.io import detect_file, review_detection
from heatmap_decoder.models import ROIReview

pytestmark = [
    pytest.mark.ocr_integration,
    pytest.mark.skipif(
        os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1",
        reason="enable the OCR extra and HEATMAP_DECODER_OCR_TESTS=1",
    ),
]
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "examples/labels"


def test_reviewed_localisation_keeps_numbers_and_exact_label_order(tmp_path):
    figure = FIXTURE / "reconstruction.png"
    hints = DetectionHints(
        rows=4,
        columns=3,
        heatmap_bbox=(220, 140, 1000, 460),
        requested=("colourbar", "labels_rows", "labels_columns"),
        colourbar_start_value=2,
        colourbar_end_value=-2,
    )
    saved = detect_file(figure, hints, tmp_path / "roi")
    assert saved.result.label_evidence[0].preprocessing.upscale_factor == 3
    assert saved.result.label_evidence[0].preprocessing.min_confidence == 80
    assert saved.result.unresolved_requirements
    assert not saved.proposed_config_complete
    bar = next(c for c in saved.result.candidates if c.bbox == (1040, 140, 1060, 460))
    assert any(a.colourbar_id == bar.id for a in saved.result.associations)
    annotations = json.loads((ROOT / "examples/roi_m8/labels_annotation.json").read_text())
    for axis, side in (("rows", "left"), ("columns", "above")):
        region = next(c for c in saved.result.candidates if c.id == f"labels_{axis}-{side}")
        bounds = annotations["axes"][axis]["text_bounds"]
        assert (
            np.max(np.abs(np.asarray(region.bbox) - bounds)) <= annotations["edge_tolerance_pixels"]
        )
        assert region.bbox[0] <= bounds[0] and region.bbox[1] <= bounds[1]
        assert region.bbox[2] >= bounds[2] and region.bbox[3] >= bounds[3]
    selections = ROIReview(
        regions={
            "colourbar": bar.id,
            "labels_rows": "labels_rows-left",
            "labels_columns": "labels_columns-above",
        }
    )
    data = (tmp_path / "roi/roi_candidates.json").read_bytes()
    payload = json.loads((tmp_path / "roi/config_proposed.json").read_text())
    config, _ = review_detection(
        data, load_raster(figure), saved.image_sha256, payload, None, selections
    )
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(selections.model_dump_json())
    result = extract_file(
        figure,
        config,
        tmp_path / "result",
        roi_review=tmp_path / "roi/roi_candidates.json",
        roi_selections=selection_path,
    )
    baseline = extract_file(figure, config.model_copy(update={"labels": None}), tmp_path / "manual")
    np.testing.assert_array_equal(result.values, baseline.values)
    truth = json.loads((FIXTURE / "truth.json").read_text())
    assert [slot.final_text for slot in result.labels.axes[0].slots] == truth["rows"]
    assert [slot.final_text for slot in result.labels.axes[1].slots] == truth["columns"]
    assert result.labels.unresolved_count == 0
    assert [slot.index for slot in result.labels.axes[0].slots] == [0, 1, 2, 3]
    metadata = json.loads((tmp_path / "result/provenance.json").read_text())
    assert (
        metadata["roi_review"]["selected_regions"]["labels_rows"]["origin"] == "reviewed_detection"
    )
    print("M8 reviewed localisation: exact OCR 7/7, unresolved 0/7, numeric change 0")
