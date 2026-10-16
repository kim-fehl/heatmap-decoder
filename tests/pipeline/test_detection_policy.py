"""Detection policy contracts and fixtures."""

import json

import pytest
from typer.testing import CliRunner

from heatmap_decoder import DetectionHints, load_raster
from heatmap_decoder.cli import app
from heatmap_decoder.detection_policy import policy_snapshot, threshold_snapshot
from heatmap_decoder.io import detect_file, review_detection
from tests.support.detection_evidence import CORPUS


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
