"""Validation pre-registration must reject altered source/runtime and retain denominators."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.m16 import validate_current as validation  # noqa: E402


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
