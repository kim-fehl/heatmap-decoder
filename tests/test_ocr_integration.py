"""Opt-in tests against checked-in images and a documented local engine environment."""

import json
import os
from pathlib import Path

import pytest

from heatmap_decoder import GridConfig, extract_labels_file

pytestmark = [
    pytest.mark.ocr_integration,
    pytest.mark.skipif(
        os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1",
        reason="opt in with HEATMAP_DECODER_OCR_TESTS=1 and install [ocr]/Tesseract",
    ),
]
FIXTURES = Path(__file__).resolve().parents[1] / "examples/labels"


def run(variant, tmp_path):
    config = GridConfig.model_validate_json((FIXTURES / "config.json").read_text())
    result = extract_labels_file(FIXTURES / variant, config, tmp_path / variant)
    truth = json.loads((FIXTURES / "truth.json").read_text())
    total = sum(len(axis.slots) for axis in result.axes)
    correct = sum(
        slot.final_text == truth[axis.axis][slot.index]
        for axis in result.axes
        for slot in axis.slots
    )
    raw_correct = sum(
        slot.raw_text == truth[axis.axis][slot.index] for axis in result.axes for slot in axis.slots
    )
    print(
        f"{variant}: exact accepted strings {correct}/{total}; "
        f"raw exact strings {raw_correct}/{total}; unresolved {result.unresolved_count}/{total}"
    )
    for axis, engine in result.ocr_runs:
        print(
            f"{axis}: Tesseract={engine.engine_version}, pytesseract={engine.wrapper_version}, "
            f"language={engine.language}, mode={engine.mode}"
        )
    return result, correct, total


def test_fixed_clean_fixture_exact_strings_and_slots(tmp_path):
    result, correct, total = run("clean.png", tmp_path)
    assert correct == total == 7
    assert result.unresolved_count == 0
    sample = result.axes[1].slots[2]
    assert sample.final_text == "Sample A"
    assert len(result.axes[1].candidates[sample.candidate_indices[0]].token_indices) == 2


def test_fixed_unreadable_fixture_stays_unresolved(tmp_path):
    result, _, _ = run("unreadable.png", tmp_path)
    assert result.unresolved_count > 0
    assert any(s.final_text is None for axis in result.axes for s in axis.slots)


@pytest.mark.parametrize("variant", ["blur.png", "compressed.jpg"])
def test_report_degraded_variants_without_universal_accuracy_assertions(tmp_path, variant):
    result, _, total = run(variant, tmp_path)
    assert sum(len(axis.slots) for axis in result.axes) == total == 7
    assert len(result.ocr_runs) == 2
