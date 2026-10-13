"""Default dispatch, raw observations, field geometry and hard worker timeouts."""

import pytest
from PIL import Image

from heatmap_decoder import ocr, rapid_ocr
from heatmap_decoder.models import LabelsConfig, LegendOptions, OCRConfig


def payload(result):
    return dict(
        result=result, version="1.4.4", runtime="test-runtime", models=(("recognizer.onnx", "abc"),)
    )


def test_default_engine_is_scoped_to_labels_and_native_size():
    assert LabelsConfig().ocr.engine == "rapidocr"
    assert LabelsConfig(ocr={}).ocr.upscale_factor == 1
    assert LabelsConfig(ocr={"min_confidence": 85}).ocr.engine == "rapidocr"
    assert LabelsConfig(ocr={"engine": "tesseract"}).ocr.upscale_factor == 3
    assert LabelsConfig(ocr={"upscale_factor": 4}).ocr.upscale_factor == 4
    assert LegendOptions().ocr.engine == "tesseract"
    assert OCRConfig().engine == "tesseract"


def test_default_dispatch_preserves_suffix_and_identifies_field_geometry(monkeypatch):
    seen = []
    monkeypatch.setattr(rapid_ocr.importlib.metadata, "version", lambda _: "installed")

    def request(pixels, detection, timeout):
        seen.append((pixels.shape, detection, timeout))
        return payload([["EsGalectin1", 0.894], [" HSP9O ", 0.88]])

    monkeypatch.setattr(rapid_ocr, "_request", request)
    run = ocr.recognize(Image.new("L", (128, 23)), LabelsConfig().ocr, mode="text_line")
    assert seen == [((23, 128, 3), False, 30)]
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


def test_unavailable_languages_and_numeric_whitelist_do_not_silently_fall_back():
    with pytest.raises(ocr.OCRUnavailableError, match="engine='tesseract'"):
        ocr.recognize(Image.new("L", (10, 10)), OCRConfig(engine="rapidocr", language="deu"))
    with pytest.raises(ValueError, match="constrained numeric OCR"):
        ocr.recognize(Image.new("L", (10, 10)), LabelsConfig().ocr, mode="numeric_line")


def test_source_field_boxes_exclude_artificial_white_padding(monkeypatch):
    import numpy as np

    from heatmap_decoder.extract import extract_labels
    from heatmap_decoder.models import GridConfig, OCRRun, OCRToken

    config = GridConfig.model_validate(
        {
            "heatmap_bbox": [0, 0, 10, 20],
            "rows": 2,
            "columns": 1,
            "labels": {
                "rows": {
                    "source": "ocr",
                    "bbox": [10, 0, 20, 20],
                    "recognition": "slots",
                    "white_padding_pixels": 2,
                }
            },
        }
    )

    def field(image, settings, **kwargs):
        return OCRRun(
            (OCRToken("Raw1", (0, 0, image.width, image.height), 95),),
            "test",
            "test",
            "eng",
            ("eng",),
            "rapidocr-recognize-field",
            geometry_kind="source_field",
        )

    monkeypatch.setattr(ocr, "recognize", field)
    result = extract_labels(np.full((20, 20, 3), 255, dtype=np.uint8), config)
    assert [s.bbox for s in result.axes[0].slots] == [(10, 0, 20, 10), (10, 10, 20, 20)]
    assert [s.status for s in result.axes[0].slots] == ["accepted", "accepted"]
    assert all(run.tokens[0].bbox == (2, 2, 12, 12) for _, _, _, run in result.slot_ocr_runs)


def test_real_default_engine_recovers_suffix_with_existing_equal_slots(tmp_path):
    import json
    import os
    from pathlib import Path

    from heatmap_decoder import GridConfig, extract_labels_file

    if os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1":
        pytest.skip("opt in with HEATMAP_DECODER_OCR_TESTS=1")
    root = Path(__file__).resolve().parents[1]
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


test_real_default_engine_recovers_suffix_with_existing_equal_slots = pytest.mark.ocr_integration(
    test_real_default_engine_recovers_suffix_with_existing_equal_slots
)


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
