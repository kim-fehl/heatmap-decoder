"""Unexpected-script suggestions cannot become silently accepted identifiers."""

from collections import deque

import numpy as np
import pytest

from heatmap_decoder.extract import extract_labels
from heatmap_decoder.labels import filter_identifier_characters
from heatmap_decoder.models import GridConfig, OCRRun, OCRToken


@pytest.mark.parametrize(
    "text",
    [
        "IL-1β / TNFα_2 (control)+",
        "EsIκBζ",
        "Échantillon-2",
        "α\u0301",
        "H₂O",
    ],
)
def test_scientific_names_keep_greek_accents_spaces_and_punctuation(text):
    assert filter_identifier_characters(text) == text


@pytest.mark.parametrize(
    "text, cleaned",
    [
        ("東BRCA1", "BRCA1"),
        ("株Sample 1😀", "Sample 1"),
        ("株\u0301", ""),
    ],
)
def test_other_scripts_are_only_stripped_from_the_suggestion(text, cleaned):
    assert filter_identifier_characters(text) == cleaned


@pytest.mark.parametrize("policy", ["latin_greek", "unrestricted"])
def test_both_axes_preserve_raw_tokens_and_apply_review_to_cleaned_suggestions(monkeypatch, policy):
    from heatmap_decoder import ocr

    raw = deque(["東BRCA1", "IL-1β", "株Sample 1", "Sample α-2"])
    config = GridConfig(
        heatmap_bbox=(0, 0, 20, 20),
        rows=2,
        columns=2,
        labels=dict(
            character_policy=policy,
            rows=dict(source="ocr", recognition="slots", bbox=(20, 0, 40, 20)),
            columns=dict(source="ocr", recognition="slots", bbox=(0, 20, 20, 40)),
        ),
    )

    def recognize(image, settings, **kwargs):
        return OCRRun(
            (OCRToken(raw.popleft(), (0, 0, image.width, image.height), 99),),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "text_line",
            geometry_kind="source_field",
        )

    monkeypatch.setattr(ocr, "recognize", recognize)
    result = extract_labels(np.full((40, 40, 3), 255, dtype=np.uint8), config)
    rows, columns = result.axes
    assert [s.raw_text for s in rows.slots] == ["東BRCA1", "IL-1β"]
    assert [s.raw_text for s in columns.slots] == ["株Sample 1", "Sample α-2"]
    assert [run.tokens[0].text for _, _, _, run in result.slot_ocr_runs] == [
        "東BRCA1",
        "IL-1β",
        "株Sample 1",
        "Sample α-2",
    ]
    assert rows.slots[1].final_text == "IL-1β"
    assert columns.slots[1].final_text == "Sample α-2"
    for slot, suggestion in ((rows.slots[0], "BRCA1"), (columns.slots[0], "Sample 1")):
        if policy == "latin_greek":
            assert slot.final_text is None and slot.suggested_text == suggestion
            assert slot.status == "unreviewed"
            assert "unexpected_label_characters" in slot.issues
        else:
            assert slot.final_text == slot.raw_text
            assert slot.suggested_text is None and slot.status == "accepted"
