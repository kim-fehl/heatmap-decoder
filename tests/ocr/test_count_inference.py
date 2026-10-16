"""Count inference contracts and fixtures."""

import pytest

from heatmap_decoder import AutomationRequest, DetectionHints, automate
from tests.support.synthetic import layout_fixture


@pytest.mark.parametrize("engine", ["tesseract", "rapidocr"])
def test_label_geometry_count_inference_accepts_or_safely_abstains(engine):
    image, _ = layout_fixture(rows=20, columns=6, row_side="left", column_side="bottom")
    result = automate(
        image,
        AutomationRequest(
            hints=DetectionHints(
                infer_rows=True,
                infer_columns=True,
                requested=("heatmap", "colourbar"),
                colourbar_start_value=2,
                colourbar_end_value=-2,
                detector={"count_label_geometry": True},
                labels={"ocr": {"engine": engine}},
            ),
            numeric_only=True,
        ),
    )
    # M17 measures the regular lattice independently of label spelling/OCR.
    # Both engines may now complete numeric geometry without identifier acceptance.
    assert result.complete.status == "accepted"
    assert (result.diagnostic_config.rows, result.diagnostic_config.columns) == (20, 6)
    assert result.detection.selected_counts["rows"].origin == "measured_inference"
    assert result.detection.selected_counts["columns"].origin == "measured_inference"
    assert not result.automatic_export_allowed
    assert result.detection.acceptance.identifier_verification == "not_requested"
