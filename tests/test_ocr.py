"""Deterministic adapter tests with fake engine responses; no optional install needed."""

from types import SimpleNamespace

import pytest
from PIL import Image

from heatmap_decoder import ocr
from heatmap_decoder.models import OCRConfig


def data():
    return dict(
        text=["", " a-0_I1 "],
        left=[0, 3],
        top=[0, 6],
        width=[30, 18],
        height=[30, 9],
        conf=["-1", "87.5"],
        block_num=[0, 1],
        par_num=[0, 2],
        line_num=[0, 3],
    )


def wrapper(**changes):
    values = dict(
        __version__="fake-wrapper",
        get_tesseract_version=lambda: "fake-engine",
        get_languages=lambda **kwargs: ["eng"],
        Output=SimpleNamespace(DICT="dict"),
        image_to_data=lambda image, **kwargs: data(),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_parse_preserves_structural_records_and_raw_spelling():
    tokens = ocr.tokens_from_data(data())
    assert len(tokens) == 2
    assert tokens[0].confidence == -1
    assert tokens[1].text == " a-0_I1 "
    assert tokens[1].bbox == (3, 6, 21, 15)
    assert tokens[1].line_id == (1, 2, 3)
    assert tokens[1].confidence == 87.5


@pytest.mark.parametrize("change", [lambda d: d.pop("conf"), lambda d: d["text"].append("extra")])
def test_malformed_engine_data_is_rejected(change):
    payload = data()
    change(payload)
    with pytest.raises(ValueError, match="OCR returned"):
        ocr.tokens_from_data(payload)


def test_exact_single_mode_and_runtime_metadata(monkeypatch):
    calls = []

    def image_to_data(image, **kwargs):
        calls.append((image, kwargs))
        return data()

    monkeypatch.setattr(ocr, "_wrapper", lambda: wrapper(image_to_data=image_to_data))
    image = Image.new("L", (30, 30))
    result = ocr.recognize(image, OCRConfig())
    assert calls == [
        (image, dict(lang="eng", config="--oem 1 --psm 11", output_type="dict", timeout=30))
    ]
    assert result.engine_version == "fake-engine"
    assert result.wrapper_version == "fake-wrapper"
    assert result.available_languages == ("eng",)


@pytest.mark.parametrize("field,value", [("conf", "nan"), ("conf", 101), ("width", -2)])
def test_invalid_token_values_are_rejected(field, value):
    payload = data()
    payload[field][1] = value
    with pytest.raises(ValueError, match="invalid token"):
        ocr.tokens_from_data(payload)


def test_missing_wrapper_is_actionable(monkeypatch):
    def missing(name):
        assert name == "pytesseract"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(ocr.importlib, "import_module", missing)
    with pytest.raises(ocr.OCRUnavailableError, match=r"heatmap-decoder\[ocr\]"):
        ocr.recognize(Image.new("L", (10, 10)), OCRConfig())


def test_missing_engine_is_actionable(monkeypatch):
    def unavailable():
        raise FileNotFoundError("tesseract missing")

    monkeypatch.setattr(ocr, "_wrapper", lambda: wrapper(get_tesseract_version=unavailable))
    with pytest.raises(ocr.OCRUnavailableError, match="engine on PATH"):
        ocr.recognize(Image.new("L", (10, 10)), OCRConfig())


def test_missing_language_and_timeout_do_not_fall_back(monkeypatch):
    monkeypatch.setattr(ocr, "_wrapper", lambda: wrapper())
    with pytest.raises(ocr.OCRUnavailableError, match="language data unavailable: deu"):
        ocr.recognize(Image.new("L", (10, 10)), OCRConfig(language="eng+deu"))

    def timeout(*args, **kwargs):
        raise RuntimeError("process timeout")

    monkeypatch.setattr(ocr, "_wrapper", lambda: wrapper(image_to_data=timeout))
    with pytest.raises(ValueError, match="OCR failed: process timeout"):
        ocr.recognize(Image.new("L", (10, 10)), OCRConfig())
