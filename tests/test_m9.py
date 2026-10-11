"""M9 frozen benchmark integrity and enforced review-only publication behaviour."""

import json
import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from heatmap_decoder import DetectionHints, load_raster
from heatmap_decoder.cli import app
from heatmap_decoder.detection_policy import policy_snapshot, threshold_snapshot
from heatmap_decoder.io import detect_file, review_detection
from heatmap_decoder.models import ROICandidate

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks/m9"
BENCH = runpy.run_path(str(CORPUS / "evaluate.py"))


def test_frozen_corpus_and_complete_report_include_every_attempt():
    manifest, digest = BENCH["load_frozen"]()
    report = json.loads((CORPUS / "report.json").read_text())
    assert len(manifest["entries"]) == 36
    assert report["manifest_sha256"] == digest
    assert report["thresholds"] == threshold_snapshot()
    assert report["automation_policy"] == policy_snapshot()
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


def test_zero_acceptance_does_not_pass_a_vacuous_gate():
    report = json.loads((CORPUS / "report.json").read_text())
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


def test_thresholds_and_enforced_policy_survive_saved_review(tmp_path):
    figure = CORPUS / "figures/grayscale_left-2-clean.png"
    hints = DetectionHints.model_validate_json((CORPUS / "review_hints.json").read_text())
    saved = detect_file(figure, hints, tmp_path / "roi")
    assert saved.detector_thresholds == json.loads(json.dumps(threshold_snapshot()))
    assert saved.automation_policy["mode"] == "review_only"
    raw = saved.model_dump(mode="json")
    # Saved evidence is not authority to enable runtime automation.
    raw["automation_policy"]["automatic_roi_enabled"] = True
    config = json.loads((tmp_path / "roi/config_proposed.json").read_text())
    config.update(
        heatmap_bbox=[160, 100, 424, 220],
        colourbar_bbox=[90, 100, 106, 220],
        colourbar_orientation="vertical",
    )
    _, metadata = review_detection(
        json.dumps(raw).encode(), load_raster(figure), saved.image_sha256, config, None
    )
    assert metadata["enforced_automation_policy"] == policy_snapshot()
    assert not metadata["enforced_automation_policy"]["automatic_roi_enabled"]
    assert all(r["origin"] == "manual" for r in metadata["selected_regions"].values())


@pytest.mark.parametrize(
    "fixture", ["challenge-uniform", "challenge-multipanel", "challenge-absent_bar"]
)
def test_unreviewed_detection_retains_diagnostics_and_blocks_matrix(tmp_path, fixture):
    runner = CliRunner()
    figure = CORPUS / "figures" / f"{fixture}.png"
    args = [
        "detect",
        str(figure),
        "--config",
        str(CORPUS / "review_hints.json"),
        "--out",
        str(tmp_path / "roi"),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 5, result.output
    assert (tmp_path / "roi/roi_diagnostic.png").exists()
    assert not (tmp_path / "roi/matrix.csv").exists()
    result = runner.invoke(
        app,
        [
            "extract",
            str(figure),
            "--config",
            str(tmp_path / "roi/config_proposed.json"),
            "--roi-review",
            str(tmp_path / "roi/roi_candidates.json"),
            "--out",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 2, result.output
    assert not (tmp_path / "results/matrix.csv").exists()
    assert not (tmp_path / "results/quality.csv").exists()


def test_automatic_hand_off_flag_is_not_exposed_before_gate_passes(tmp_path):
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(CORPUS / "figures/grayscale_left-2-clean.png"),
            "--config",
            str(CORPUS / "review_hints.json"),
            "--auto-roi",
            "--out",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert not (tmp_path / "matrix.csv").exists()
