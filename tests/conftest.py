"""Assign execution tiers and opt-in engine policy from the test layout."""

import os
from pathlib import Path

import pytest

TIERS = {"unit", "pipeline", "ocr", "qualification"}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    ocr_enabled = os.environ.get("HEATMAP_DECODER_OCR_TESTS") == "1"
    for item in items:
        tier = Path(item.path).parent.name
        if tier not in TIERS:
            continue
        item.add_marker(getattr(pytest.mark, tier))
        if tier == "ocr":
            item.add_marker(pytest.mark.ocr_integration)
            if not ocr_enabled:
                item.add_marker(pytest.mark.skip(reason="opt in with HEATMAP_DECODER_OCR_TESTS=1"))
