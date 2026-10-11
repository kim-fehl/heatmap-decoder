"""M7 fixed-coordinate regression gates, independent of OCR availability."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder.cli import app
from heatmap_decoder.detection import ScaleTransform, analysis_image, detect
from heatmap_decoder.io import detect_file, extract_file, load_raster, review_detection
from heatmap_decoder.models import DetectionHints, DetectorOptions, ExtractionConfig

ROOT = Path(__file__).resolve().parents[1]
fixtures = runpy.run_path(str(ROOT / "examples/roi/generate.py"))


@pytest.mark.parametrize("name", fixtures["FIXTURES"])
def test_fixed_supported_top_three(name):
    image, box, rows, columns = fixtures["make_fixture"](name)
    hints = DetectionHints(rows=rows, columns=columns)
    result = detect(image, hints)
    pitch = np.array([(box[2] - box[0]) / columns, (box[3] - box[1]) / rows] * 2)
    errors = [np.abs(np.array(c.bbox) - box) / pitch for c in result.candidates[:3]]
    assert any(np.max(error) <= 0.1 for error in errors), (name, result)
    assert result.status in {"review_required", "ambiguous"}
    assert not result.selected
    assert result == detect(image, hints)


def test_round_outward_unequal_resize_ratios():
    transform = ScaleTransform(1001, 703, 512, 360)
    assert transform.original_box((1, 1, 511, 359)) == (1, 1, 1000, 702)
    image = np.zeros((703, 1001, 3), dtype=np.uint8)
    small, actual = analysis_image(image, 512)
    assert small.shape[:2] == (360, 512)
    assert actual == transform
    assert actual.original_box((0, 0, 512, 360)) == (0, 0, 1001, 703)


@pytest.mark.parametrize("kind", ["blank", "uniform", "noise", "gradient"])
def test_negative_unresolved(kind):
    image = np.full((240, 300, 3), 245, dtype=np.uint8)
    if kind == "uniform":
        image[30:210, 30:270] = [55, 90, 110]
    elif kind == "noise":
        image[30:210, 30:270] = np.random.default_rng(44).integers(
            0, 256, (180, 240, 3), dtype=np.uint8
        )
    elif kind == "gradient":
        image[30:210, 30:270] = np.linspace(20, 220, 240, dtype=np.uint8)[None, :, None]
    result = detect(image, DetectionHints(rows=8, columns=10))
    assert result.status == "unresolved"
    assert result.candidates == ()
    assert result.unresolved_requirements


def test_multiple_panels_never_selected():
    image, box, rows, columns = fixtures["make_fixture"]("sequential")
    canvas = np.concatenate((image, image), axis=1)
    result = detect(canvas, DetectionHints(rows=rows, columns=columns))
    assert result.status == "ambiguous"
    assert len(result.candidates) >= 2
    assert not result.selected
    assert box in [c.bbox for c in result.candidates[:3]]


def test_manual_bypass_and_bounds(monkeypatch):
    image, box, rows, columns = fixtures["make_fixture"]("border")
    monkeypatch.setattr(
        "heatmap_decoder.detection.analysis_image",
        lambda *args: pytest.fail("manual bounds must bypass detection"),
    )
    hints = DetectionHints(rows=rows, columns=columns, heatmap_bbox=box, requested=())
    result = detect(image, hints)
    assert result.selected["heatmap"].bbox == box
    assert result.selected["heatmap"].origin == "manual"
    assert result.status == "manual"
    assert result.candidates == ()
    with pytest.raises(ValueError, match="inside"):
        detect(image, hints.model_copy(update={"heatmap_bbox": (0, 0, 9999, 9999)}))
    with pytest.raises(ValueError, match="at least one pixel"):
        detect(image, hints.model_copy(update={"rows": 9999}))


@pytest.mark.parametrize(
    "payload",
    [
        {"rows": 0, "columns": 2},
        {"rows": 1.1, "columns": 2},
        {"rows": 2, "columns": 2, "infer_rows": True},
        {"rows": 2, "columns": 2, "requested": ["colourbar"]},
        {"rows": 2, "columns": 2, "heatmap_bbox": [0, 0, 20, 20]},
        {"rows": 2, "columns": 2, "requested": []},
        {"rows": 2, "columns": 2, "requested": ["heatmap", "heatmap"]},
        {"rows": 2, "columns": 2, "detector": {"min_score": float("nan")}},
    ],
)
def test_strict_hints(payload):
    with pytest.raises(ValueError):
        DetectionHints.model_validate(payload)


def test_candidate_cap():
    image, _, rows, columns = fixtures["make_fixture"]("sequential")
    result = detect(
        np.concatenate((image, image), axis=1),
        DetectionHints(rows=rows, columns=columns, detector=DetectorOptions(max_candidates=1)),
    )
    assert len(result.candidates) == 1
    assert result.unresolved_requirements


def reference_hints():
    payload = json.loads((ROOT / "examples/config.json").read_text())
    payload.pop("heatmap_bbox")
    payload.pop("sampling_inset_fraction")
    return DetectionHints.model_validate(payload)


def test_exports_review_and_unchanged_extraction(tmp_path):
    figure = ROOT / "examples/figure.png"
    saved = detect_file(figure, reference_hints(), tmp_path / "roi")
    assert saved.analysis_transforms[0] == {
        "original_width": 350,
        "original_height": 576,
        "analysis_width": 311,
        "analysis_height": 512,
    }
    assert not saved.proposed_config_complete
    assert saved.missing_extraction_fields == ("heatmap_bbox",)
    assert saved.image_sha256 == hashlib.sha256(figure.read_bytes()).hexdigest()
    proposed = json.loads((tmp_path / "roi/config_proposed.json").read_text())
    assert "heatmap_bbox" not in proposed
    data = (tmp_path / "roi/roi_candidates.json").read_bytes()
    config, evidence = review_detection(
        data, load_raster(figure), saved.image_sha256, proposed, saved.result.candidates[0].id
    )
    assert evidence["origin"] == "reviewed_detection"
    result = extract_file(
        figure,
        config,
        tmp_path / "reviewed",
        roi_review=tmp_path / "roi/roi_candidates.json",
        heatmap_candidate="heatmap-001",
    )
    manual_config = ExtractionConfig.model_validate_json(
        (ROOT / "examples/config.json").read_text()
    )
    manual = extract_file(figure, manual_config, tmp_path / "manual")
    np.testing.assert_array_equal(result.values, manual.values)
    truth = np.loadtxt(ROOT / "examples/truth.csv", delimiter=",")
    assert np.mean(np.abs(result.values - truth)) < 0.02 * 4
    metadata = json.loads((tmp_path / "reviewed/provenance.json").read_text())
    assert metadata["roi_review"]["evidence"]["result"]["candidates"]
    assert metadata["roi_review"]["review_sha256"] == hashlib.sha256(data).hexdigest()
    # Editing the config makes it usable without any detection dependency at extraction time.
    config_file = tmp_path / "reviewed.json"
    config_file.write_text(config.model_dump_json())
    cli = CliRunner().invoke(
        app,
        ["extract", str(figure), "--config", str(config_file), "--out", str(tmp_path / "ordinary")],
    )
    assert cli.exit_code == 0, cli.output


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("hash", "hash/dimensions"),
        ("dimensions", "hash/dimensions"),
        ("schema", "schema_version"),
        ("duplicate", "duplicate"),
        ("box", "inside"),
        ("id", "unknown"),
        ("count", "count"),
        ("conflict", "conflicts"),
    ],
)
def test_saved_review_rejects_invalid(tmp_path, mutation, match):
    figure = ROOT / "examples/figure.png"
    saved = detect_file(figure, reference_hints(), tmp_path)
    payload = saved.model_dump(mode="json")
    config = json.loads((tmp_path / "config_proposed.json").read_text())
    candidate = "heatmap-001"
    if mutation == "hash":
        payload["image_sha256"] = "0" * 64
    elif mutation == "dimensions":
        payload["image_dimensions"][0] += 1
    elif mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "duplicate":
        payload["result"]["candidates"] *= 2
    elif mutation == "box":
        payload["result"]["candidates"][0]["bbox"][2] = 9999
    elif mutation == "id":
        candidate = "unknown"
    elif mutation == "count":
        config["rows"] += 1
    elif mutation == "conflict":
        config["heatmap_bbox"] = [1, 1, 20, 20]
    with pytest.raises(ValueError, match=match):
        review_detection(
            json.dumps(payload).encode(), load_raster(figure), saved.image_sha256, config, candidate
        )


def test_cli_review_exit_overwrite_and_no_matrices(tmp_path):
    config = tmp_path / "hints.json"
    config.write_text(reference_hints().model_dump_json())
    args = [
        "detect",
        str(ROOT / "examples/figure.png"),
        "--config",
        str(config),
        "--out",
        str(tmp_path / "roi"),
    ]
    runner = CliRunner()
    first = runner.invoke(app, args)
    assert first.exit_code == 5, first.output
    assert {p.name for p in (tmp_path / "roi").iterdir()} == {
        "roi_candidates.json",
        "config_proposed.json",
        "roi_diagnostic.png",
    }
    with Image.open(tmp_path / "roi/roi_diagnostic.png") as overlay:
        assert overlay.size == (350, 576)
    assert runner.invoke(app, args).exit_code == 2
    assert runner.invoke(app, args + ["--overwrite"]).exit_code == 5
    extract = runner.invoke(
        app,
        [
            "extract",
            str(ROOT / "examples/figure.png"),
            "--config",
            str(tmp_path / "roi/config_proposed.json"),
            "--roi-review",
            str(tmp_path / "roi/roi_candidates.json"),
            "--heatmap-candidate",
            "heatmap-001",
            "--out",
            str(tmp_path / "results"),
        ],
    )
    assert extract.exit_code == 0, extract.output


def test_blank_exports_no_guessed_boxes_or_endpoints(tmp_path):
    figure = tmp_path / "blank.png"
    Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8)).save(figure)
    saved = detect_file(figure, DetectionHints(rows=5, columns=5), tmp_path / "out")
    assert saved.result.status == "unresolved"
    proposed = json.loads((tmp_path / "out/config_proposed.json").read_text())
    assert proposed == {"rows": 5, "columns": 5}
    assert not saved.proposed_config_complete


def test_manual_cli_complete_and_input_protection(tmp_path):
    payload = json.loads((ROOT / "examples/config.json").read_text())
    payload.pop("sampling_inset_fraction")
    payload["requested"] = []
    config = tmp_path / "hints.json"
    config.write_text(json.dumps(payload))
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "detect",
            str(ROOT / "examples/figure.png"),
            "--config",
            str(config),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    proposed = ExtractionConfig.model_validate_json(
        (tmp_path / "out/config_proposed.json").read_text()
    )
    assert proposed.heatmap_bbox == tuple(payload["heatmap_bbox"])
    protected = tmp_path / "config_proposed.json"
    protected.write_text(json.dumps(payload))
    result = runner.invoke(
        app,
        [
            "detect",
            str(ROOT / "examples/figure.png"),
            "--config",
            str(protected),
            "--out",
            str(tmp_path),
            "--overwrite",
        ],
    )
    assert result.exit_code == 2
    assert "overwrite an input" in result.output


def test_relative_file_labels_survive_proposed_config_location(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "genes.txt").write_text("\n".join(f"Gene{i}" for i in range(20)))
    payload = reference_hints().model_dump(mode="json")
    payload["labels"] = {"rows": {"source": "file", "path": "genes.txt"}}
    hints = DetectionHints.model_validate(payload)
    hint_path = inputs / "hints.json"
    hint_path.write_text(hints.model_dump_json())
    figure = ROOT / "examples/figure.png"
    detect_file(figure, hints, tmp_path / "roi", config_path=hint_path)
    proposed = json.loads((tmp_path / "roi/config_proposed.json").read_text())
    assert (tmp_path / "roi" / proposed["labels"]["rows"]["path"]).resolve() == inputs / "genes.txt"
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(figure),
            "--config",
            str(tmp_path / "roi/config_proposed.json"),
            "--roi-review",
            str(tmp_path / "roi/roi_candidates.json"),
            "--heatmap-candidate",
            "heatmap-001",
            "--out",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Gene0" in (tmp_path / "results/matrix.csv").read_text()


def test_edited_config_review_metadata(tmp_path):
    figure = ROOT / "examples/figure.png"
    saved = detect_file(figure, reference_hints(), tmp_path / "roi")
    config = ExtractionConfig.model_validate_json((ROOT / "examples/config.json").read_text())
    config_path = tmp_path / "edited.json"
    config_path.write_text(config.model_dump_json())
    extract_file(
        figure,
        config,
        tmp_path / "out",
        roi_review=tmp_path / "roi/roi_candidates.json",
        config_path=config_path,
    )
    metadata = json.loads((tmp_path / "out/provenance.json").read_text())["roi_review"]
    assert metadata["config_sha256"] == hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert metadata["selected_regions"]["heatmap"]["origin"] == "manual"
    assert metadata["evidence"]["image_sha256"] == saved.image_sha256


def test_fixed_label_bounds_checked_without_ocr():
    hints = DetectionHints(
        rows=2, columns=2, labels={"rows": {"source": "ocr", "bbox": [20, 20, 300, 400]}}
    )
    with pytest.raises(ValueError, match="labels.rows.*inside"):
        detect(np.full((50, 50, 3), 255, dtype=np.uint8), hints)


def test_manual_disagreement_warns_without_moving_bounds():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    hints = DetectionHints(rows=5, columns=5, heatmap_bbox=(10, 10, 90, 90), requested=())
    result = detect(image, hints)
    assert result.selected["heatmap"].bbox == hints.heatmap_bbox
    assert result.warnings
