"""Small optional adapter for local Tesseract. No imports or probes until requested."""

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from PIL import Image

from .models import OCRConfig, OCRRun, OCRToken

OCR_MODE = "--oem 1 --psm 11"
OCR_TIMEOUT_SECONDS = 30


class OCRUnavailableError(ValueError):
    """An explicit OCR request cannot run in the current environment."""


def _wrapper() -> Any:
    try:
        return importlib.import_module("pytesseract")
    except ImportError as error:
        raise OCRUnavailableError(
            'OCR requires the optional wrapper: pip install "heatmap-decoder[ocr]" '
            "(or uv sync --extra ocr). Install the Tesseract engine separately."
        ) from error


def tokens_from_data(data: Mapping[str, Sequence[Any]]) -> tuple[OCRToken, ...]:
    """Parse engine TSV fields, retaining structural (-1) records as raw evidence."""
    fields = ("text", "left", "top", "width", "height", "conf", "block_num", "par_num", "line_num")
    if any(name not in data for name in fields):
        raise ValueError("OCR returned incomplete token data")
    count = len(data["text"])
    if any(len(data[name]) != count for name in fields):
        raise ValueError("OCR returned inconsistent token counts")
    tokens = []
    for index in range(count):
        left, top, width, height = (float(data[name][index]) for name in fields[1:5])
        confidence = float(data["conf"][index])
        if (
            not all(math.isfinite(v) for v in (left, top, width, height, confidence))
            or min(left, top, width, height) < 0
            or not -1 <= confidence <= 100
        ):
            raise ValueError("OCR returned invalid token geometry/confidence")
        tokens.append(
            OCRToken(
                text=str(data["text"][index]),
                bbox=(left, top, left + width, top + height),
                confidence=confidence,
                line_id=tuple(int(data[name][index]) for name in fields[6:]),
            )
        )
    return tuple(tokens)


def recognize(
    image: Image.Image,
    settings: OCRConfig,
    *,
    mode: Literal["localisation", "numeric_line", "text_line"] = "localisation",
    timeout_seconds: float = OCR_TIMEOUT_SECONDS,
) -> OCRRun:
    """Sparse localisation or one bounded numeric line; never auto-search OCR variants."""
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= OCR_TIMEOUT_SECONDS:
        raise ValueError("OCR timeout must be positive and at most 30 seconds")
    if mode not in {"localisation", "numeric_line", "text_line"}:
        raise ValueError("unsupported OCR mode")
    engine_mode = OCR_MODE if mode == "localisation" else "--oem 1 --psm 7"
    wrapper = _wrapper()
    try:
        engine_version = str(wrapper.get_tesseract_version())
        languages = tuple(sorted(wrapper.get_languages(config="")))
    except (OSError, RuntimeError) as error:
        raise OCRUnavailableError(
            "OCR requires a local Tesseract engine on PATH. "
            "Linux: install tesseract-ocr and tesseract-ocr-eng; "
            "macOS: brew install tesseract; Windows: install Tesseract and add it to PATH. "
            f"Engine check failed: {error}"
        ) from error
    missing = [language for language in settings.language.split("+") if language not in languages]
    if missing:
        raise OCRUnavailableError(
            f"Tesseract language data unavailable: {', '.join(missing)}. "
            "Install the requested traineddata and check TESSDATA_PREFIX; "
            f"available languages: {', '.join(languages) or '(none)'}"
        )
    try:
        data = wrapper.image_to_data(
            image,
            lang=settings.language,
            config=engine_mode,
            output_type=wrapper.Output.DICT,
            timeout=timeout_seconds,
        )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Local Tesseract OCR failed: {error}") from error
    return OCRRun(
        tokens_from_data(data),
        engine_version,
        str(wrapper.__version__),
        settings.language,
        languages,
        engine_mode,
    )
