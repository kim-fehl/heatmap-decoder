"""Validation contracts and fixtures."""

import copy
import json

import pytest

from benchmarks.m16 import validate_current as validation
from heatmap_decoder import (
    AutomationRequest,
    DetectionHints,
    ScaleAnchor,
    ScaleConfig,
    automate,
)
from tests.support import load_script
from tests.support.scale_fixtures import FONT_AVAILABLE, ROOT, case_for
from tests.support.synthetic import scale_cases, scale_fixture


def test_partition_keeps_derivatives_with_fresh_base_seeds():
    plan = validation.plan(validation.DEFAULT_SEED)
    assert len(plan["cases"]) == 16
    assert len({c["seed"] for c in plan["cases"]}) == 16
    assert all(c["seed"] > 166217 for c in plan["cases"])
    assert sum(len(c["derivatives"]) for c in plan["cases"]) == 48
    assert {c["derivatives"][0] for c in plan["cases"]} == {"clean"}
    assert len(plan["negatives"]) == 6
    assert len({c["seed"] for c in plan["negatives"]}) == 6
    assert not {c["seed"] for c in plan["negatives"]} & {c["seed"] for c in plan["cases"]}
    assert plan["label_modes"] == ["sparse", "slots"]


@pytest.fixture
def registered(tmp_path, monkeypatch):
    fingerprint = ({"src/file.py": "sha"}, {"models": {"model.onnx": "model-sha"}})
    monkeypatch.setattr(validation, "fingerprint", lambda: copy.deepcopy(fingerprint))
    path = tmp_path / "freeze.json"
    validation.freeze(path, validation.DEFAULT_SEED)
    return path, fingerprint


def test_freeze_is_read_only_and_matches_before_evaluation(registered):
    path, _ = registered
    frozen = json.loads(path.read_text())
    validation.verify(frozen)
    with pytest.raises(FileExistsError):
        validation.freeze(path, validation.DEFAULT_SEED)


@pytest.mark.parametrize("change", ["source", "added_source", "runtime", "threshold", "plan"])
def test_changed_source_runtime_or_case_plan_cannot_enter_validation(registered, change):
    path, fingerprint = registered
    frozen = json.loads(path.read_text())
    if change == "source":
        frozen["source_sha256"]["src/file.py"] = "different"
    elif change == "added_source":
        fingerprint[0]["src/new.py"] = "new"
    elif change == "runtime":
        frozen["runtime"]["models"]["model.onnx"] = "different"
    elif change == "threshold":
        frozen["policy"]["mae_fraction"] = 0.03
    else:
        frozen["plan"]["cases"].pop()
    with pytest.raises(ValueError, match="mismatch"):
        validation.verify(frozen)


def test_old_partition_cannot_be_registered_as_fresh(tmp_path):
    with pytest.raises(ValueError, match="historical"):
        validation.freeze(tmp_path / "freeze.json", 166200)


def test_split_and_derivatives_are_fixed_by_base():
    groups = {}
    for split in ("development", "validation", "test"):
        for case in scale_cases(split):
            groups.setdefault(case["seed"], set()).add(split)
    assert len(groups) == 75
    assert all(len(splits) == 1 for splits in groups.values())
    prereg = json.loads((ROOT / "benchmarks/m12/preregister.json").read_text())
    assert prereg["ranges"] == [[-4, 4], [-3, 3], [-2, 2], [-4, 2], [-3, 4]]


@pytest.mark.skipif(not FONT_AVAILABLE, reason="synthetic renderer requires DejaVuSans")
def test_selective_evaluator_reconstructs_with_runtime_ocr_anchors(monkeypatch):
    evaluator = load_script(str(ROOT / "benchmarks/m12/evaluate.py"))
    case = case_for(0)
    image, truth = scale_fixture(case)
    result = automate(
        image,
        AutomationRequest(
            hints=DetectionHints(
                requested=["heatmap", "colourbar"],
                rows=truth["rows"],
                columns=truth["columns"],
                colourbar_start_value=truth["start_value"],
                colourbar_end_value=truth["end_value"],
            ),
            numeric_only=True,
        ),
    )
    assert result.diagnostic_config is not None
    anchors = tuple(
        ScaleAnchor(position=t["position"], value=t["value"], origin="ocr", status="suggested")
        for t in truth["ticks"]
    )
    config = result.diagnostic_config.model_copy(update={"scale": ScaleConfig(source="ocr_ticks")})
    result = result.model_copy(update={"diagnostic_config": config, "ocr_anchors": anchors})
    attempt = evaluator["attempt"]
    monkeypatch.setitem(attempt.__globals__, "automate", lambda *_: result)
    report = attempt(case, image, truth, 3)
    assert report["numeric"] is not None
    assert report["numeric"]["mae"] < 0.02
