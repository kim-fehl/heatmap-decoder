"""Layout detection contracts and fixtures."""

import pytest

from heatmap_decoder.models import (
    GridConfig,
    LabelsConfig,
)
from tests.support.synthetic import layout_fixture


@pytest.mark.parametrize("row_side,column_side", [("left", "top"), ("right", "bottom")])
def test_real_ocr_30_genes_10_samples_preserves_review_slots(row_side, column_side, capsys):
    from heatmap_decoder import extract_labels

    image, truth = layout_fixture(rows=30, columns=10, row_side=row_side, column_side=column_side)
    settings = GridConfig(
        heatmap_bbox=truth["heatmap_bbox"],
        rows=30,
        columns=10,
        labels=LabelsConfig.model_validate(
            {axis: {"source": "ocr", "bbox": box} for axis, box in truth["label_boxes"].items()}
        ),
    )
    result = extract_labels(image, settings)
    assert [len(a.slots) for a in result.axes] == [30, 10]
    expected = {"rows": truth["genes"], "columns": truth["samples"]}
    exact = sum(s.final_text == expected[a.axis][s.index] for a in result.axes for s in a.slots)
    raw_available = sum(bool(s.raw_text) for a in result.axes for s in a.slots)
    print(
        f"{row_side}/{column_side}: exact {exact}/40; raw {raw_available}/40; "
        f"unresolved {result.unresolved_count}/40"
    )
    assert raw_available >= 30  # Localisation sanity, never identifier verification.
    assert exact >= 20
