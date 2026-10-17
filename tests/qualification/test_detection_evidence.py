"""Detection evidence contracts and fixtures."""

import json
import shutil

import numpy as np
import pytest

from heatmap_decoder.detection_policy import threshold_snapshot
from heatmap_decoder.models import ROICandidate
from tests.support import load_script
from tests.support.detection_evidence import CORPUS

BENCH = load_script(str(CORPUS / "evaluate.py"))


def test_frozen_corpus_and_complete_report_include_every_attempt():
    manifest, digest = BENCH["load_frozen"]()
    report = json.loads((CORPUS / "report.json").read_text())
    assert len(manifest["entries"]) == 36
    assert report["manifest_sha256"] == digest
    assert report["thresholds"] == threshold_snapshot()
    assert report["automation_policy"]["version"] == "m9-review-only.1"
    assert not report["automation_policy"]["automatic_roi_enabled"]
    assert [r["id"] for r in report["attempts"]] == [e["id"] for e in manifest["entries"]]
    assert report["summary"]["attempted_figures"] == 36
    assert report["summary"]["evaluation_errors"] == 0
    assert report["summary"]["negative_roi_attempts"] == 5
    assert len({e["family"] for e in manifest["entries"] if not e["negative"]}) >= 3
    assert report["summary"] == BENCH["summarise"](report["attempts"])
    assert not report["automation_gates"]["complete_layout_gate_passed"]
    assert report["automation_gates"]["accepted_complete_layouts"] == 0
    assert not any(report["automation_gates"]["automatic_count_axes"].values())
    assert report["published_real_figure_count"] == 0


@pytest.mark.parametrize("kind", ["image", "manifest"])
def test_changed_frozen_inputs_rejected(tmp_path, kind):
    shutil.copytree(CORPUS, tmp_path / "corpus")
    if kind == "manifest":
        target = tmp_path / "corpus/manifest.json"
        target.write_bytes(target.read_bytes() + b" ")
        match = "manifest hash mismatch"
    else:
        target = tmp_path / "corpus/figures/grayscale_left-2-clean.png"
        target.write_bytes(target.read_bytes() + b"changed")
        match = "image hash mismatch"
    with pytest.raises(ValueError, match=match):
        BENCH["load_frozen"](tmp_path / "corpus")


def test_edge_metric_rejects_one_cell_offset_despite_high_iou():
    box = (0, 0, 200, 200)
    candidate = ROICandidate(id="shifted", bbox=(10, 0, 210, 200), score=1, score_components={})
    metric = BENCH["region_metric"](
        candidate, box, np.array([10, 10, 10, 10]), np.array([1, 1, 1, 1])
    )
    assert metric["iou"] > 0.9
    assert metric["edge_error_cell_pitch"][0] == 1
    assert not metric["within_tolerance"]


def test_zero_acceptance_does_not_pass_a_vacuous_gate(monkeypatch):
    report = json.loads((CORPUS / "report.json").read_text())
    monkeypatch.setitem(
        BENCH["gate_report"].__globals__, "policy_snapshot", lambda: report["automation_policy"]
    )
    gate = BENCH["gate_report"](report["summary"])
    assert gate["decision"] == "ship_review_only"
    assert gate["required_complete_layouts"] == 10
    assert gate["required_layout_families"] == 3
    assert not gate["complete_layout_gate_passed"]
    assert gate["accepted_exact_count_examples"] == {"rows": 0, "columns": 0}


def test_ocr_error_report_separates_false_acceptance_from_geometry():
    report = json.loads((CORPUS / "report.json").read_text())
    summary = report["summary"]["labels"]
    assert summary["accepted_texts"] == summary["exact_accepted"] + summary["false_accepted_texts"]
    assert summary["slots"] == summary["accepted_texts"] + summary["unresolved"]
    assert summary["false_accepted_texts"] > 0
    wrong = [
        s
        for r in report["attempts"]
        for a in r.get("labels", {}).get("axes", {}).values()
        for s in a["slot_evidence"]
        if s["status"] == "accepted" and s["final_text"] != s["expected_text"]
    ]
    assert any(s["geometry_association_correct"] for s in wrong)
