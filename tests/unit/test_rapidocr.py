"""Rapidocr contracts and fixtures."""

import pytest
from PIL import Image

from heatmap_decoder import ocr, rapid_ocr
from heatmap_decoder.models import LabelsConfig, LegendOptions, OCRConfig


def payload(result):
    return dict(
        result=result, version="1.4.4", runtime="test-runtime", models=(("recognizer.onnx", "abc"),)
    )


def test_default_engines_use_rapidocr_and_native_size():
    assert LabelsConfig().ocr.engine == "rapidocr"
    assert LabelsConfig(ocr={}).ocr.upscale_factor == 1
    assert LabelsConfig(ocr={"min_confidence": 85}).ocr.engine == "rapidocr"
    assert LabelsConfig(ocr={"engine": "tesseract"}).ocr.upscale_factor == 3
    assert LabelsConfig(ocr={"upscale_factor": 4}).ocr.upscale_factor == 4
    assert LegendOptions().ocr.engine == "rapidocr"
    assert LegendOptions().ocr.upscale_factor == 1
    assert LegendOptions().white_padding_pixels == 2
    legacy = LegendOptions(ocr={"engine": "tesseract"})
    assert legacy.ocr.upscale_factor == 3 and legacy.white_padding_pixels == 0
    assert OCRConfig().engine == "tesseract"


def test_default_dispatch_preserves_suffix_and_identifies_field_geometry(monkeypatch):
    seen = []
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda _: "installed")

    def request(pixels, detection, timeout, language):
        seen.append((pixels.shape, detection, timeout, language))
        return payload([["EsGalectin1", 0.894], [" HSP9O ", 0.88]])

    monkeypatch.setattr(rapid_ocr, "_request", request)
    run = ocr.recognize(Image.new("L", (128, 23)), LabelsConfig().ocr, mode="text_line")
    assert seen == [((23, 128, 3), False, 30, "eng")]
    assert [t.text for t in run.tokens] == ["EsGalectin1", " HSP9O "]
    assert run.tokens[0].confidence == pytest.approx(89.4)
    assert run.tokens[0].bbox == (0, 0, 128, 23)
    assert run.engine_name == "RapidOCR"
    assert run.geometry_kind == "source_field"
    assert run.score_kind == "uncalibrated_mean_character_score_x100"
    assert run.model_sha256 == (("recognizer.onnx", "abc"),)


def test_sparse_detection_returns_observed_text_boxes(monkeypatch):
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda _: "installed")
    monkeypatch.setattr(
        rapid_ocr,
        "_request",
        lambda *args: payload([[[[2, 3], [20, 2], [21, 12], [3, 13]], "Sample A", 0.92]]),
    )
    run = ocr.recognize(Image.new("RGB", (30, 20)), LabelsConfig().ocr)
    assert run.tokens[0].bbox == (2, 2, 21, 13)
    assert run.tokens[0].text == "Sample A"
    assert run.geometry_kind == "detected_text"


@pytest.mark.parametrize("result", [[["name", float("nan")]], [["name", 1.1]], [[42, 0.9]]])
def test_invalid_raw_engine_results_are_rejected(monkeypatch, result):
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda _: "installed")
    monkeypatch.setattr(rapid_ocr, "_request", lambda *args: payload(result))
    with pytest.raises(ValueError, match="invalid text/score"):
        ocr.recognize(Image.new("L", (10, 10)), LabelsConfig().ocr, mode="text_line")


def test_inference_timeout_terminates_worker_and_clears_cache(monkeypatch):
    import subprocess
    import sys
    import time

    import numpy as np

    process = subprocess.Popen(
        [sys.executable, "-c", "import sys,time; sys.stdin.buffer.read(1); time.sleep(30)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    monkeypatch.setattr(rapid_ocr, "_worker", process)
    started = time.monotonic()
    with pytest.raises(ValueError, match="timeout; worker terminated"):
        rapid_ocr._request(np.zeros((10, 10, 3), dtype=np.uint8), False, 0.1)
    assert time.monotonic() - started < 3
    assert process.poll() is not None
    assert rapid_ocr._worker is None


def test_unavailable_languages_do_not_silently_fall_back():
    with pytest.raises(ocr.OCRUnavailableError, match="engine='tesseract'"):
        ocr.recognize(Image.new("L", (10, 10)), OCRConfig(engine="rapidocr", language="deu"))


def test_engine_settings_alone_do_not_request_identifiers_in_numeric_only_work():
    from pydantic import ValidationError

    from heatmap_decoder.automation import AutomationRequest

    hints = {
        "rows": 2,
        "columns": 1,
        "requested": ["heatmap", "colourbar"],
        "labels": {"ocr": {"engine": "tesseract"}},
    }
    request = AutomationRequest(hints=hints, numeric_only=True)
    assert request.hints.labels.ocr.engine == "tesseract"
    hints["labels"]["rows"] = {"source": "file", "path": "genes.txt"}
    with pytest.raises(ValidationError, match="numeric_only conflicts"):
        AutomationRequest(hints=hints, numeric_only=True)


def test_english_selection_uses_packaged_recognizer_and_embedded_dictionary():
    import hashlib
    import importlib

    paths = rapid_ocr._model_paths(importlib.import_module("rapidocr_onnxruntime"), "eng")
    assert paths[-1].name == "en_PP-OCRv4_rec_infer.onnx"
    assert hashlib.sha256(paths[-1].read_bytes()).hexdigest() == (
        "e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318"
    )
    legacy = rapid_ocr._model_paths(importlib.import_module("rapidocr_onnxruntime"), "chinese")
    assert legacy[-1].name == "ch_PP-OCRv4_rec_infer.onnx"


@pytest.mark.parametrize("language", ["eng", "en", "english", "English"])
def test_english_aliases_select_english_model(monkeypatch, language):
    seen = []

    def request(pixels, detection, timeout, selected):
        seen.append(selected)
        return payload([["Sample_1 long 51", 0.99]])

    monkeypatch.setattr(rapid_ocr, "_request", request)
    run = ocr.recognize(
        Image.new("L", (160, 30)), OCRConfig(engine="rapidocr", language=language), mode="text_line"
    )
    assert seen == ["eng"]
    assert run.tokens[0].text == "Sample_1 long 51"
    assert "PP-OCRv4 eng" in run.engine_version


def test_numeric_field_preserves_unconstrained_raw_text_for_the_strict_parser(monkeypatch):
    from heatmap_decoder.legend import parse_numeric_tick

    monkeypatch.setattr(rapid_ocr, "_request", lambda *args: payload([["O.1", 0.99]]))
    run = ocr.recognize(Image.new("L", (20, 20)), LegendOptions().ocr, mode="numeric_line")
    assert run.tokens[0].text == "O.1"
    assert run.mode == "rapidocr-recognize-numeric-field-unconstrained"
    assert parse_numeric_tick(run.tokens[0].text).value is None
    assert run.geometry_kind == "source_field"
