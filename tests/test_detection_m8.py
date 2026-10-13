"""Review-only bar/label/count contracts, including deliberately confounded inputs."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder import DetectionHints, detect, load_raster
from heatmap_decoder.cli import app
from heatmap_decoder.io import detect_file, review_detection
from heatmap_decoder.models import OCRRun, OCRToken, ROIReview
from heatmap_decoder.ocr import OCRUnavailableError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = runpy.run_path(str(ROOT / "examples/roi_m8/generate.py"))


@pytest.mark.parametrize("name", FIXTURES["LAYOUTS"])
def test_fixed_bars_association_without_ocr(name, monkeypatch):
    monkeypatch.setattr(
        "heatmap_decoder.ocr.recognize", lambda *a: pytest.fail("OCR must be optional")
    )
    image, annotation = FIXTURES["make_layout"](name)
    hints = DetectionHints(
        rows=6, columns=8, heatmap_bbox=annotation["heatmap_bbox"], requested=("colourbar",)
    )
    result = detect(image, hints)
    assert any(
        np.max(np.abs(np.asarray(c.bbox) - annotation["colourbar_bbox"])) <= 1
        for c in result.candidates[:3]
    )
    correct = [c for c in result.candidates if c.bbox == annotation["colourbar_bbox"]]
    assert len(correct) == 1
    assert correct[0].orientation == annotation["orientation"]
    assert any(
        a.colourbar_id == correct[0].id and a.heatmap_id == "heatmap-manual"
        for a in result.associations
    )
    assert "colourbar" not in result.selected
    assert result.unresolved_requirements


def test_competing_bars_and_absence(tmp_path):
    image, annotation = FIXTURES["make_layout"]("vertical_sequential")
    image = image.copy()
    image[100:280, 90:104] = image[100:280, 410:424]
    hints = DetectionHints(
        rows=6, columns=8, heatmap_bbox=annotation["heatmap_bbox"], requested=("colourbar",)
    )
    result = detect(image, hints)
    assert result.status == "ambiguous"
    assert len(result.associations) >= 2
    assert "colourbar" not in result.selected
    image[100:280, 90:104] = 250
    image[100:280, 410:424] = 250
    figure = tmp_path / "absent.png"
    Image.fromarray(image).save(figure)
    saved = detect_file(figure, hints, tmp_path / "roi")
    assert saved.result.status == "unresolved"
    assert saved.result.unresolved_requirements
    assert (tmp_path / "roi/roi_diagnostic.png").exists()
    assert "colourbar_bbox" not in json.loads((tmp_path / "roi/config_proposed.json").read_text())


def test_categorical_track_and_clipped_bar():
    image, annotation = FIXTURES["make_layout"]("vertical_sequential")
    hints = DetectionHints(
        rows=6, columns=8, heatmap_bbox=annotation["heatmap_bbox"], requested=("colourbar",)
    )
    result = detect(image, hints)
    assert not any(c.bbox[1] == 80 for c in result.candidates)
    image = image.copy()
    image[100:280, 410:424] = 250
    image[100:280, :14] = FIXTURES["palette"](np.linspace(0, 0.5, 180), "sequential")[:, None]
    result = detect(image, hints)
    assert result.unresolved_requirements
    assert all("touches_image_edge_possible_clipping" in c.reasons for c in result.candidates)


@pytest.mark.parametrize("kind", ["complete", "omitted", "merged"])
@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_count_true_and_harmonic_retained(kind, axis):
    image, box, rows, columns = FIXTURES["count_fixture"](kind)
    hints = DetectionHints(heatmap_bbox=box, requested=(), infer_rows=True, infer_columns=True)
    result = detect(image, hints)
    count = rows if axis == "rows" else columns
    proposals = [p for p in result.count_proposals if p.axis == axis]
    assert count in [p.count for p in proposals]
    assert any(
        p.count != count and (p.count % count == 0 or count % p.count == 0) for p in proposals
    )
    assert all("integer_multiple_ambiguity" in p.reasons for p in proposals)
    assert all(p.origin is None for p in proposals)
    assert result.status == "ambiguous"
    if kind == "complete":
        assert result.selected_counts[axis].count == count
        assert result.selected_counts[axis].origin == "measured_inference"
    else:
        assert result.selected_counts == {}
    if kind == "complete":
        true = next(p for p in proposals if p.count == count)
        assert {"gridline_strokes", "colour_transitions"} <= true.signals.keys()


def test_uniform_counts_abstain_and_never_export_guesses(tmp_path):
    image, box, _, _ = FIXTURES["count_fixture"]("uniform")
    figure = tmp_path / "uniform.png"
    Image.fromarray(image).save(figure)
    hints = DetectionHints(heatmap_bbox=box, requested=(), infer_rows=True, infer_columns=True)
    saved = detect_file(figure, hints, tmp_path / "roi")
    assert saved.result.count_proposals == ()
    assert saved.result.status == "unresolved"
    assert "rows" not in json.loads((tmp_path / "roi/config_proposed.json").read_text())
    assert "columns" not in json.loads((tmp_path / "roi/config_proposed.json").read_text())


def test_supplied_counts_authoritative_even_when_signals_disagree():
    image, box, _, _ = FIXTURES["count_fixture"]("complete")
    hints = DetectionHints(rows=7, columns=8, heatmap_bbox=box, requested=(), infer_rows=True)
    result = detect(image, hints)
    assert result.selected_counts["rows"].count == 7
    assert all(p.count != 7 and p.conflicts for p in result.count_proposals)


@pytest.mark.parametrize(
    "payload",
    [
        {"rows": None, "columns": 2},
        {"rows": 2, "columns": 2, "infer_rows": "yes"},
        {
            "rows": 2,
            "columns": 2,
            "colourbar_bbox": [1, 1, 20, 100],
            "requested": ["heatmap", "colourbar"],
        },
        {
            "rows": 2,
            "columns": 2,
            "heatmap_bbox": [10, 10, 100, 100],
            "requested": ["labels_rows"],
            "labels": {"rows": {"source": "file", "path": "a.txt"}},
        },
    ],
)
def test_conflicting_hints(payload):
    with pytest.raises(ValueError):
        DetectionHints.model_validate(payload)


def fake_ocr(monkeypatch, tokens_by_call):
    calls = iter(tokens_by_call)

    def recognize(image, settings):
        tokens = next(calls)
        factor = settings.upscale_factor
        return OCRRun(
            tuple(
                OCRToken(t.text, tuple(v * factor for v in t.bbox), t.confidence) for t in tokens
            ),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "--psm 11",
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)


def test_unavailable_ocr_saves_unresolved_without_losing_bar(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OCRUnavailableError('Install "heatmap-decoder[ocr]" and local Tesseract')

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", unavailable)
    image, annotation = FIXTURES["make_layout"]("vertical_sequential")
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    hints = DetectionHints(
        rows=6,
        columns=8,
        heatmap_bbox=annotation["heatmap_bbox"],
        requested=("colourbar", "labels_rows"),
    )
    saved = detect_file(figure, hints, tmp_path / "roi")
    assert any(c.kind == "colourbar" for c in saved.result.candidates)
    assert not any(c.kind == "labels_rows" for c in saved.result.candidates)
    assert saved.result.status == "unresolved"
    assert any("Tesseract" in w for w in saved.result.warnings)
    assert (tmp_path / "roi/roi_candidates.json").is_file()


def test_mock_label_positions_order_partial_and_misleading_caption(monkeypatch):
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    # Crop-relative row boxes: heatmap y=50..150, five slots of pitch 20.
    tokens = [
        OCRToken("Gapdh", (20, 5, 70, 15), 95),
        OCRToken("HLA-DRA", (20, 45, 90, 55), 95),
        OCRToken("BRCA2", (20, 85, 80, 95), 75),
        OCRToken("Caption", (5, 18, 80, 30), 99),
    ]
    fake_ocr(monkeypatch, [tokens, ()])
    hints = DetectionHints(
        rows=5, columns=4, heatmap_bbox=(100, 50, 260, 150), requested=("labels_rows",)
    )
    result = detect(image, hints)
    candidate = next(c for c in result.candidates if c.kind == "labels_rows")
    assert "incomplete_label_axis" in candidate.reasons
    assert "low_confidence_text" in candidate.reasons
    assignment = result.label_evidence[0].assignment
    assert len(assignment.slots) == 5
    assert assignment.slots[0].status == "ambiguous"
    assert "Gapdh" in assignment.slots[0].raw_text
    assert assignment.slots[2].final_text == "HLA-DRA"
    assert assignment.slots[4].final_text is None
    assert assignment.slots[4].raw_text == "BRCA2"
    assert not any(slot.final_text == "Caption" for slot in assignment.slots)


def test_unknown_count_ocr_keeps_unassigned_positions_and_missing_labels(monkeypatch):
    image, box, rows, columns = FIXTURES["count_fixture"]("merged")
    # Search left crop spans y=25..145; missing a middle row is deliberate.
    tokens = [OCRToken(f"Gene{i}", (1, i * 20 + 5, 20, i * 20 + 15), 95) for i in (0, 1, 3, 4, 5)]
    fake_ocr(monkeypatch, [tokens, ()])
    hints = DetectionHints(
        columns=columns, infer_rows=True, heatmap_bbox=box, requested=("labels_rows",)
    )
    result = detect(image, hints)
    assert len(result.label_evidence[0].unassigned_candidates) == 5
    assert result.label_evidence[0].assignment is None
    true = next(p for p in result.count_proposals if p.count == rows)
    assert "ocr_left" in true.signals
    assert true.signals["ocr_left"].coverage == pytest.approx(5 / 6)
    assert not any(c.kind == "labels_rows" for c in result.candidates)


def test_region_and_count_id_review_and_conflicts(tmp_path):
    image, annotation = FIXTURES["make_layout"]("vertical_sequential")
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    hints = DetectionHints(
        heatmap_bbox=annotation["heatmap_bbox"],
        requested=("colourbar",),
        infer_rows=True,
        infer_columns=True,
        colourbar_start_value=1,
        colourbar_end_value=0,
    )
    saved = detect_file(figure, hints, tmp_path / "roi")
    payload = json.loads((tmp_path / "roi/config_proposed.json").read_text())
    bar = next(c for c in saved.result.candidates if c.bbox == annotation["colourbar_bbox"])
    count_ids = {
        axis: next(
            p.id
            for p in saved.result.count_proposals
            if p.axis == axis and p.count == annotation[axis]
        )
        for axis in ("rows", "columns")
    }
    selections = ROIReview(regions={"colourbar": bar.id}, counts=count_ids)
    data = (tmp_path / "roi/roi_candidates.json").read_bytes()
    config, evidence = review_detection(data, image, saved.image_sha256, payload, None, selections)
    assert config.rows == 6 and config.columns == 8
    assert config.colourbar_bbox == bar.bbox
    assert evidence["selected_counts"]["rows"]["origin"] == "reviewed_detection"
    selection_path = tmp_path / "selections.json"
    selection_path.write_text(selections.model_dump_json())
    cli = CliRunner().invoke(
        app,
        [
            "extract",
            str(figure),
            "--config",
            str(tmp_path / "roi/config_proposed.json"),
            "--roi-review",
            str(tmp_path / "roi/roi_candidates.json"),
            "--roi-selections",
            str(selection_path),
            "--out",
            str(tmp_path / "result"),
        ],
    )
    assert cli.exit_code == 0, cli.output
    metadata = json.loads((tmp_path / "result/provenance.json").read_text())
    assert (
        metadata["roi_review"]["selections_sha256"]
        == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    )
    payload["rows"] = 7
    with pytest.raises(ValueError, match="conflicts"):
        review_detection(data, image, saved.image_sha256, payload, None, selections)
    with pytest.raises(ValueError, match="unknown colourbar"):
        review_detection(
            data, image, saved.image_sha256, {}, None, ROIReview(regions={"colourbar": "x"})
        )


@pytest.mark.parametrize("side", ["right", "below"])
def test_opposite_label_sides_keep_slot_order(monkeypatch, side):
    image = np.full((300, 400, 3), 255, dtype=np.uint8)
    if side == "right":
        tokens = [OCRToken(f"Gene{i}", (5, i * 40 + 10, 60, i * 40 + 30), 95) for i in range(3)]
        fake_ocr(monkeypatch, [(), tokens])
        requested = ("labels_rows",)
        index = 1
    else:
        tokens = [OCRToken(f"Sample{i}", (i * 80 + 20, 5, i * 80 + 60, 25), 95) for i in range(3)]
        fake_ocr(monkeypatch, [(), tokens])
        requested = ("labels_columns",)
        index = 1
    result = detect(
        image,
        DetectionHints(rows=3, columns=3, heatmap_bbox=(100, 80, 340, 200), requested=requested),
    )
    candidate = next(c for c in result.candidates if c.id.endswith(side))
    assert candidate.score == 1
    slots = result.label_evidence[index].assignment.slots
    assert [s.index for s in slots] == [0, 1, 2]
    assert [s.final_text for s in slots] == (
        [f"Gene{i}" for i in range(3)] if side == "right" else [f"Sample{i}" for i in range(3)]
    )


def test_competing_label_sides_ambiguous(monkeypatch):
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    tokens = [OCRToken(f"Gene{i}", (5, i * 30 + 10, 35, i * 30 + 20), 95) for i in range(3)]
    fake_ocr(monkeypatch, [tokens, tokens])
    result = detect(
        image,
        DetectionHints(
            rows=3, columns=3, heatmap_bbox=(100, 50, 200, 140), requested=("labels_rows",)
        ),
    )
    assert result.status == "ambiguous"
    assert len(result.candidates) == 2
    assert "labels_rows" not in result.selected


def test_requested_labels_cannot_silently_become_numeric_only(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OCRUnavailableError("install OCR")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", unavailable)
    figure = ROOT / "examples/labels/reconstruction.png"
    payload = json.loads((ROOT / "examples/labels/numeric_config.json").read_text())
    hints = DetectionHints.model_validate({**payload, "requested": ["labels_rows"]})
    saved = detect_file(figure, hints, tmp_path / "roi")
    with pytest.raises(ValueError, match="requested labels_rows is unresolved"):
        review_detection(
            (tmp_path / "roi/roi_candidates.json").read_bytes(),
            load_raster(figure),
            saved.image_sha256,
            payload,
            None,
        )


def test_old_m7_saved_schema_still_imports(tmp_path):
    figure = ROOT / "examples/figure.png"
    config = json.loads((ROOT / "examples/config.json").read_text())
    config.pop("sampling_inset_fraction")
    hints = DetectionHints.model_validate({**config, "requested": []})
    saved = detect_file(figure, hints, tmp_path)
    old = saved.model_dump(mode="json")
    old["schema_version"] = 1
    old["detector_version"] = "m7.1"
    for name in ("count_proposals", "selected_counts", "label_evidence", "associations"):
        old["result"].pop(name)
    for name in ("infer_rows", "infer_columns"):
        old["hints"].pop(name)
    result, _ = review_detection(
        json.dumps(old).encode(), load_raster(figure), saved.image_sha256, config, None
    )
    assert result.heatmap_bbox == tuple(config["heatmap_bbox"])


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("association", "unknown region"),
        ("pitch", "pitch/phase"),
        ("count_id", "duplicate"),
        ("selected_count", "contradicts"),
        ("signal_position", "outside"),
        ("image_hash", "hash/dimensions"),
    ],
)
def test_m8_saved_evidence_integrity(tmp_path, mutation, match):
    image, annotation = FIXTURES["make_layout"]("vertical_sequential")
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    hints = DetectionHints(
        rows=6,
        columns=8,
        heatmap_bbox=annotation["heatmap_bbox"],
        requested=("colourbar",),
        infer_rows=True,
        colourbar_start_value=1,
        colourbar_end_value=0,
    )
    saved = detect_file(figure, hints, tmp_path / "roi")
    payload = saved.model_dump(mode="json")
    config = json.loads((tmp_path / "roi/config_proposed.json").read_text())
    if mutation == "association":
        payload["result"]["associations"][0]["colourbar_id"] = "unknown"
    elif mutation == "pitch":
        payload["result"]["count_proposals"][0]["pitch"] += 1
    elif mutation == "count_id":
        payload["result"]["count_proposals"].append(payload["result"]["count_proposals"][0])
    elif mutation == "selected_count":
        payload["result"]["selected_counts"]["rows"]["count"] = 7
    elif mutation == "signal_position":
        payload["result"]["count_proposals"][0]["signals"]["colour_transitions"]["positions"][0] = (
            999
        )
    elif mutation == "image_hash":
        payload["image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=match):
        review_detection(json.dumps(payload).encode(), image, saved.image_sha256, config, None)


def test_no_joint_box_count_refinement_when_one_axis_unknown():
    image, annotation = FIXTURES["make_layout"]("vertical_sequential")
    result = detect(image, DetectionHints(columns=8, infer_rows=True, requested=("heatmap",)))
    assert result.candidates
    assert any(
        p.count == 6 and p.bbox == annotation["heatmap_bbox"] for p in result.count_proposals
    )
    assert all(p.bbox in [c.bbox for c in result.candidates] for p in result.count_proposals)
    assert result.selected_counts["columns"].count == 8
    assert result.selected_counts["rows"].count == 6
    assert result.selected_counts["rows"].origin == "measured_inference"
    assert result.acceptance.decisions[0].status == "accepted"


def test_conflicting_count_observations_are_retained(monkeypatch):
    image, box, _, columns = FIXTURES["count_fixture"]("complete")
    # Four unassigned row text positions disagree with the six-cell image grid.
    tokens = [OCRToken(f"Gene{i}", (1, i * 30 + 10, 20, i * 30 + 20), 95) for i in range(4)]
    fake_ocr(monkeypatch, [tokens, ()])
    result = detect(
        image,
        DetectionHints(
            columns=columns, infer_rows=True, heatmap_bbox=box, requested=("labels_rows",)
        ),
    )
    proposal = next(p for p in result.count_proposals if p.count == 6)
    assert proposal.signals["ocr_left"].alignment < 0.75
    assert any("ocr_left" in conflict for conflict in proposal.conflicts)
    assert "conflicting_observations" in proposal.reasons


def test_selection_file_cannot_be_overwritten_by_outputs(tmp_path):
    image, annotation = FIXTURES["make_layout"]("vertical_sequential")
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    hints = DetectionHints(
        rows=6,
        columns=8,
        heatmap_bbox=annotation["heatmap_bbox"],
        requested=("colourbar",),
        colourbar_start_value=1,
        colourbar_end_value=0,
    )
    saved = detect_file(figure, hints, tmp_path / "roi")
    choice = ROIReview(regions={"colourbar": saved.result.candidates[0].id})
    selection = tmp_path / "result/provenance.json"
    selection.parent.mkdir()
    selection.write_text(choice.model_dump_json())
    before = selection.read_bytes()
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(figure),
            "--config",
            str(tmp_path / "roi/config_proposed.json"),
            "--roi-review",
            str(tmp_path / "roi/roi_candidates.json"),
            "--roi-selections",
            str(selection),
            "--out",
            str(selection.parent),
            "--overwrite",
        ],
    )
    assert result.exit_code == 2
    assert "overwrite an input" in result.output
    assert selection.read_bytes() == before
