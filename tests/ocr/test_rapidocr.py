"""Rapidocr contracts and fixtures."""

import pytest
from PIL import Image

from heatmap_decoder import ocr
from heatmap_decoder.models import LabelsConfig, OCRConfig
from tests.support import ROOT as REPO_ROOT


def test_real_default_engine_recovers_suffix_with_existing_equal_slots(tmp_path):
    import json

    from heatmap_decoder import GridConfig, extract_labels_file

    root = REPO_ROOT
    config = GridConfig.model_validate_json((root / "examples/m16/labels-default.json").read_text())
    assert config.labels.ocr.engine == "rapidocr" and config.labels.ocr.upscale_factor == 1
    result = extract_labels_file(
        root / "examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg",
        config,
        tmp_path / "default",
    )
    assert result.axes[0].slots[20].raw_text == "EsGalectin1"
    assert len(result.slot_ocr_runs) == 41
    assert all(
        run.engine_name == "RapidOCR" and run.model_sha256 for _, _, _, run in result.slot_ocr_runs
    )
    provenance = json.loads((tmp_path / "default/provenance.json").read_text())
    assert provenance["ocr"]["slot_runs"][20]["run"]["geometry_kind"] == "source_field"


def test_english_worker_preserves_spaces_and_model_switches_are_explicit():

    from PIL import ImageDraw, ImageFont

    image = Image.new("RGB", (200, 40), "white")
    ImageDraw.Draw(image).text(
        (8, 8), "Sample_1 long 51", font=ImageFont.truetype("DejaVuSans.ttf", 16), fill="black"
    )
    english = ocr.recognize(image, LabelsConfig().ocr, mode="text_line")
    assert [t.text for t in english.tokens] == ["Sample_1 long 51"]
    assert any(name.startswith("en_") for name, _ in english.model_sha256)
    assert not any(name == "ch_PP-OCRv4_rec_infer.onnx" for name, _ in english.model_sha256)
    chinese = ocr.recognize(
        image, OCRConfig(engine="rapidocr", language="chinese"), mode="text_line"
    )
    assert any(name == "ch_PP-OCRv4_rec_infer.onnx" for name, _ in chinese.model_sha256)
    again = ocr.recognize(image, LabelsConfig().ocr, mode="text_line")
    assert again.tokens == english.tokens and again.model_sha256 == english.model_sha256


def test_default_legend_runs_without_tesseract_and_reads_independent_tick_fields(monkeypatch):

    import numpy as np

    from heatmap_decoder.legend import inspect_legend
    from heatmap_decoder.models import LegendConfig

    monkeypatch.setattr(
        ocr, "_wrapper", lambda: pytest.fail("default legend must not load Tesseract")
    )
    root = REPO_ROOT
    pixels = np.asarray(Image.open(root / "examples/diagnostic/figure.png").convert("RGB"))
    result = inspect_legend(
        pixels,
        LegendConfig(
            colourbar_bbox=(40, 360, 54, 541),
            colourbar_orientation="vertical",
            legend={"search_pixels": 64},
        ),
    )
    physical = [t for t in result.ticks if t.position_source == "stroke"]
    assert [t.parsed_value for t in physical] == [-3, 0, 3]
    assert all(r.run.engine_name == "RapidOCR" for r in result.ocr_records)
    assert all("tessedit_char_whitelist" not in r.run.mode for r in result.ocr_records)
    assert all(
        r.run.geometry_kind == "source_field"
        for r in result.ocr_records
        if r.purpose == "numeric_line"
    )
    assert not result.automatic_export_allowed
