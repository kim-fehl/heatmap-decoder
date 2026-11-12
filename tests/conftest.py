"""Assign execution tiers and opt-in engine policy from the test layout."""

import os
from pathlib import Path

import pytest
from PIL import ImageFont

TIERS = {"unit", "pipeline", "ocr", "qualification"}


@pytest.fixture
def bundled_fixture_fonts(monkeypatch):
    """Resolve legacy renderer font names without depending on installed fonts."""
    font_directory = Path(__file__).resolve().parents[1] / "benchmarks" / "fonts"
    truetype = ImageFont.truetype

    def load_font(font, *args, **kwargs):
        if isinstance(font, str) and Path(font).name == font:
            bundled = font_directory / font
            if bundled.is_file():
                font = str(bundled)
        return truetype(font, *args, **kwargs)

    monkeypatch.setattr(ImageFont, "truetype", load_font)


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
