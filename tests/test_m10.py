"""Measured M10 decisions, adversarial evidence and realistic omics layout sizes."""

import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder import DetectionHints, ExtractionConfig, detect, extract
from heatmap_decoder.acceptance import distinct_candidates, evaluate_acceptance
from heatmap_decoder.cli import app
from heatmap_decoder.detection import _count_signals, bar_measurements
from heatmap_decoder.detection_policy import policy_snapshot
from heatmap_decoder.io import detect_file, extract_file, review_detection
from heatmap_decoder.labels import labels_from_tokens
from heatmap_decoder.models import (
    GridConfig,
    LabelsConfig,
    OCRRun,
    OCRToken,
    ROICandidate,
    ROIReview,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = runpy.run_path(str(ROOT / "examples/m10/generate.py"))
LAYOUTS = list(FIXTURE["layouts"]())


def hints(annotation, **options):
    return DetectionHints(
        rows=annotation["rows"],
        columns=annotation["columns"],
        requested=("heatmap", "colourbar"),
        colourbar_start_value=2,
        colourbar_end_value=-2,
        **options,
    )


def config(annotation):
    return ExtractionConfig(
        heatmap_bbox=annotation["heatmap_bbox"],
        colourbar_bbox=annotation["colourbar_bbox"],
        colourbar_orientation=annotation["orientation"],
        rows=annotation["rows"],
        columns=annotation["columns"],
        colourbar_start_value=2,
        colourbar_end_value=-2,
    )


@pytest.mark.parametrize("options", LAYOUTS, ids=[f"layout-{i}" for i in range(32)])
def test_omics_scale_all_label_sides_and_bar_orientations(options):
    image, truth = FIXTURE["make_layout"](**options)
    settings = hints(truth)
    result = detect(image, settings)
    # Runtime selects first, before numerical truth is consulted. This is a
    # development regression assertion, never an M12 release benchmark.
    assert all(
        d.status == "accepted"
        for d in result.acceptance.decisions
        if d.requirement != "numeric_quality"
    )
    assert not result.acceptance.automatic_export_allowed
    choices = {d.requirement: d.candidate_ids for d in result.acceptance.decisions}
    heatmap = next(c for c in result.candidates if c.id == choices["heatmap"][0])
    bar = next(c for c in result.candidates if c.id == choices["colourbar"][0])
    pitches = np.array([72, 22] * 2)
    assert np.max(np.abs(np.array(heatmap.bbox) - truth["heatmap_bbox"]) / pitches) <= 0.1
    assert np.max(np.abs(np.array(bar.bbox) - truth["colourbar_bbox"])) <= 1
    reconstruction = extract(
        image,
        config(truth).model_copy(
            update={
                "heatmap_bbox": heatmap.bbox,
                "colourbar_bbox": bar.bbox,
            }
        ),
    )
    expected = 2 - 4 * np.array(truth["values"])
    finite = np.isfinite(reconstruction.values)
    assert finite.any()
    assert np.mean(np.abs(reconstruction.inferred_values - expected)) / 4 < 0.02
    assert np.mean(np.abs(reconstruction.values[finite] - expected[finite])) / 4 < 0.02
    assert np.max(np.abs(reconstruction.values[finite] - expected[finite])) / 4 < 0.03
    qc = next(d for d in result.acceptance.decisions if d.requirement == "numeric_quality")
    if not finite.all():
        assert qc.status == "review_required"
        assert result.acceptance.overall.status == "review_required"
    # Assignment of original-coordinate mock tokens is independent of OCR.
    grid = GridConfig(
        heatmap_bbox=truth["heatmap_bbox"],
        rows=truth["rows"],
        columns=truth["columns"],
        labels=LabelsConfig(),
    )
    for axis, texts in (("rows", truth["genes"]), ("columns", truth["samples"])):
        tokens = tuple(OCRToken(t["text"], t["bbox"], 99) for t in truth["tokens"][axis])
        assigned = labels_from_tokens(tokens, grid, axis)
        assert [s.index for s in assigned.slots] == list(range(len(texts)))
        assert [s.final_text for s in assigned.slots] == list(texts)
        assert all(s.status == "accepted" for s in assigned.slots)


@pytest.mark.parametrize(
    "family", ["sequential_right", "diverging_top", "grayscale_left", "gridlines_below"]
)
@pytest.mark.parametrize("degradation", ["blur", "jpeg"])
def test_ordinary_m9_bar_failures_now_offer_reviewable_bounds(family, degradation):
    suffix = ".jpg" if degradation == "jpeg" else ".png"
    image_path = ROOT / "benchmarks/m9/figures" / f"{family}-1-{degradation}{suffix}"
    manifest = json.loads((ROOT / "benchmarks/m9/manifest.json").read_text())
    truth = next(e for e in manifest["entries"] if e["id"] == f"{family}-1-{degradation}")
    from heatmap_decoder import load_raster

    result = detect(load_raster(image_path), hints(truth))
    bars = [c for c in result.candidates if c.kind == "colourbar"]
    assert bars
    assert min(np.max(np.abs(np.array(c.bbox) - truth["colourbar_bbox"])) for c in bars) <= 1
    assert not result.acceptance.automatic_export_allowed
    # Robust boundary profiles may now accept a correctly measured degraded case.


@pytest.mark.parametrize(
    "case", ["categorical", "partial", "clipped", "second_bar", "wrong_counts"]
)
def test_adversarial_layouts_do_not_pass_supplied_count_decisions(case):
    image, truth = FIXTURE["make_layout"]()
    image = image.copy()
    settings = hints(truth)
    x, y, r, s = truth["colourbar_bbox"]
    if case == "categorical":
        image[y:s, x:r] = np.repeat(
            np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8), (s - y) // 2, axis=0
        )[:, None]
    elif case == "partial":
        image[y : (y + s) // 2, x:r] = 255
    elif case == "clipped":
        image[y:s, :14] = image[y:s, x:r]
        image[y:s, x:r] = 255
    elif case == "second_bar":
        image[y:s, 40:54] = image[y:s, x:r]
    else:
        settings = settings.model_copy(update={"rows": truth["rows"] + 1})
    result = detect(image, settings)
    assert result.acceptance.overall.status != "accepted"
    assert not result.acceptance.automatic_export_allowed


def test_high_score_and_edited_saved_evidence_cannot_enable_exports(tmp_path):
    image, truth = FIXTURE["make_layout"]()
    image = image.copy()
    image[300:340, 320:896] = 127  # Break within-cell piecewise constancy.
    result = detect(image, hints(truth))
    edited = result.model_copy(
        update={
            "candidates": tuple(
                c.model_copy(update={"score": 1, "score_components": {"equal_grid": 1}})
                for c in result.candidates
            )
        }
    )
    report = evaluate_acceptance(image, hints(truth), edited)
    assert report.overall.status != "accepted"
    assert not report.automatic_export_allowed
    assert not policy_snapshot()["automatic_roi_enabled"]


def test_near_identical_boxes_deduplicated_without_averaging():
    image, truth = FIXTURE["make_layout"]()
    box = truth["heatmap_bbox"]
    candidates = (
        ROICandidate(id="a", bbox=box, score=0.9, score_components={}),
        ROICandidate(id="b", bbox=(box[0] + 1, *box[1:]), score=0.95, score_components={}),
        ROICandidate(id="c", bbox=(box[0] + 72, *box[1:]), score=0.8, score_components={}),
    )
    retained = distinct_candidates(candidates, hints(truth))
    assert [c.id for c in retained] == ["a", "c"]
    assert retained[0].bbox == box


def test_gridline_pixels_cannot_be_count_colour_support():
    image, truth = FIXTURE["make_layout"](gridlines=True)
    box = truth["heatmap_bbox"]
    x, y, r, s = box
    image = image.copy()
    image[y:s, x:r] = 100
    image[y + 21 : s : 22, x:r] = 0
    signals = _count_signals(image, box, "rows")
    assert signals["gridline_strokes"]
    assert not signals["colour_transitions"]
    result = detect(
        image,
        DetectionHints(heatmap_bbox=box, requested=(), infer_rows=True, columns=truth["columns"]),
    )
    row = next(d for d in result.acceptance.decisions if d.requirement == "rows")
    assert row.status == "review_required"
    assert all(not p["agreement"] for p in row.evidence["hypotheses"])


def test_inference_failure_does_not_block_supplied_count_roi_validation():
    image, truth = FIXTURE["make_layout"]()
    result = detect(image, hints(truth, infer_rows=True, infer_columns=True))
    assert result.acceptance.overall.status == "accepted"
    assert result.selected_counts["rows"].origin == "manual"
    assert result.selected_counts["rows"].count == truth["rows"]
    inferred = detect(
        image,
        DetectionHints(
            requested=("heatmap", "colourbar"),
            infer_rows=True,
            infer_columns=True,
            colourbar_start_value=2,
            colourbar_end_value=-2,
        ),
    )
    assert inferred.acceptance.overall.status != "accepted"
    assert inferred.selected_counts == {}


def test_categorical_jump_not_hidden_by_average_continuity():
    crop = np.full((600, 14, 3), 50.0)
    crop[300:] = 200
    measured = bar_measurements(crop)
    assert measured["continuity"] > 0.99
    assert measured["maximum_step_rgb"] > measured["step_limit_rgb"]
    assert measured["longest_plateau_fraction"] > 0.4


def test_auto_cli_retains_diagnostics_and_existing_matrices(tmp_path):
    image, truth = FIXTURE["make_layout"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    hint_path = tmp_path / "hints.json"
    hint_path.write_text(hints(truth).model_dump_json())
    out = tmp_path / "result"
    out.mkdir()
    (out / "matrix.csv").write_text("existing\n")
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(figure),
            "--auto-roi",
            "--numeric-only",
            "--config",
            str(hint_path),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 5, result.output
    assert "M12 release gate is closed" in result.output
    assert (out / "matrix.csv").read_text() == "existing\n"
    assert not (out / "quality.csv").exists()
    saved = json.loads((out / "roi_review/roi_candidates.json").read_text())
    assert saved["schema_version"] == 3
    assert saved["result"]["acceptance"]["overall"]["status"] == "accepted"
    assert not saved["result"]["acceptance"]["automatic_export_allowed"]


def test_reviewed_large_matrix_exports_unchanged_values_and_matching_labels(tmp_path):
    image, truth = FIXTURE["make_layout"](
        rows=30, columns=10, row_side="right", column_side="bottom", bar_side="top"
    )
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    saved = detect_file(figure, hints(truth), tmp_path / "roi")
    decisions = {d.requirement: d.candidate_ids for d in saved.result.acceptance.decisions}
    choices = ROIReview(regions={k: decisions[k][0] for k in ("heatmap", "colourbar")})
    payload = json.loads((tmp_path / "roi/config_proposed.json").read_text())
    settings, _ = review_detection(
        saved.model_dump_json().encode(), image, saved.image_sha256, payload, None, choices
    )
    (tmp_path / "genes.txt").write_text("\n".join(truth["genes"]) + "\n")
    (tmp_path / "samples.txt").write_text("\n".join(truth["samples"]) + "\n")
    settings = settings.model_copy(
        update={
            "labels": LabelsConfig.model_validate(
                {
                    "rows": {"source": "file", "path": "genes.txt"},
                    "columns": {"source": "file", "path": "samples.txt"},
                }
            )
        }
    )
    result = extract_file(figure, settings, tmp_path / "results", base_directory=tmp_path)
    np.testing.assert_array_equal(result.values, extract(image, config(truth)).values)
    import csv

    with (tmp_path / "results/matrix.csv").open() as stream:
        matrix = list(csv.reader(stream))
    with (tmp_path / "results/quality.csv").open() as stream:
        quality = list(csv.reader(stream))
    assert matrix[0][1:] == truth["samples"]
    assert [row[0] for row in matrix[1:]] == list(truth["genes"])
    records = [dict(zip(quality[0], row, strict=True)) for row in quality[1:]]
    assert len(records) == 300
    assert all(
        row["row_label"] == truth["genes"][int(row["row"])]
        and row["column_label"] == truth["samples"][int(row["column"])]
        for row in records
    )


@pytest.mark.parametrize("rows,columns", [(20, 6), (25, 8), (30, 10)])
def test_staged_localisation_uses_shortlist_and_retains_raw_positions(monkeypatch, rows, columns):
    image, truth = FIXTURE["make_layout"](rows=rows, columns=columns)
    # Fake adapter returns crop-relative boxes for the known four search strips.
    from heatmap_decoder.detection import _label_searches

    searches = iter(_label_searches(truth["heatmap_bbox"], image.shape, 256))

    def recognize(crop, settings):
        side, region = next(searches)
        axis = "rows" if side in {"left", "right"} else "columns"
        records = truth["tokens"][axis] if side in {"left", "above"} else []
        f = settings.upscale_factor
        tokens = tuple(
            OCRToken(t["text"], tuple((v - region[i % 2]) * f for i, v in enumerate(t["bbox"])), 99)
            for t in records
        )
        return OCRRun(tokens, "fake", "fake", "eng", ("eng",), "--psm 11")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    request = hints(truth).model_copy(
        update={"requested": ("heatmap", "colourbar", "labels_rows", "labels_columns")}
    )
    result = detect(image, request)
    assert result.label_evidence
    assert all(e.heatmap_id == "heatmap-001" for e in result.label_evidence)
    row_evidence = next(e for e in result.label_evidence if e.side == "left")
    column_evidence = next(e for e in result.label_evidence if e.side == "above")
    assert len(row_evidence.unassigned_candidates) == rows
    assert len(row_evidence.assignment.slots) == rows
    assert len(column_evidence.unassigned_candidates) == columns
    assert len(column_evidence.assignment.slots) == columns
    assert result.acceptance.identifier_verification == "review_required"
    assert result.acceptance.overall.status == "review_required"


@pytest.mark.ocr_integration
@pytest.mark.skipif(os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1", reason="opt-in OCR")
@pytest.mark.parametrize("row_side,column_side", [("left", "top"), ("right", "bottom")])
def test_real_ocr_30_genes_10_samples_preserves_review_slots(row_side, column_side, capsys):
    from heatmap_decoder import extract_labels

    image, truth = FIXTURE["make_layout"](
        rows=30, columns=10, row_side=row_side, column_side=column_side
    )
    settings = GridConfig(
        heatmap_bbox=truth["heatmap_bbox"],
        rows=30,
        columns=10,
        labels=LabelsConfig.model_validate(
            {axis: {"source": "ocr", "bbox": box} for axis, box in truth["label_boxes"].items()}
        ),
    )
    result = extract_labels(image, settings)
    assert [len(a.slots) for a in result.axes] == [30, 10]
    expected = {"rows": truth["genes"], "columns": truth["samples"]}
    exact = sum(s.final_text == expected[a.axis][s.index] for a in result.axes for s in a.slots)
    raw_available = sum(bool(s.raw_text) for a in result.axes for s in a.slots)
    print(
        f"{row_side}/{column_side}: exact {exact}/40; raw {raw_available}/40; "
        f"unresolved {result.unresolved_count}/40"
    )
    assert raw_available >= 30  # Localisation sanity, never identifier verification.
    assert exact >= 20


def test_candidate_limit_cannot_hide_competing_bars():
    from heatmap_decoder.models import DetectorOptions

    image, truth = FIXTURE["make_layout"]()
    image = image.copy()
    x, y, r, s = truth["colourbar_bbox"]
    image[y:s, 40:54] = image[y:s, x:r]
    result = detect(image, hints(truth, detector=DetectorOptions(max_candidates=1)))
    bar = next(d for d in result.acceptance.decisions if d.requirement == "colourbar")
    assert bar.status == "review_required"
    assert result.acceptance.overall.status != "accepted"


def test_unstable_refinement_requires_review(monkeypatch):

    image, truth = FIXTURE["make_layout"]()
    result = detect(image, hints(truth))
    monkeypatch.setattr(
        "heatmap_decoder.edge_profiles.refine_measured_box",
        lambda image, box: (box[0], box[1] + 3, box[2], box[3] + 3),
    )
    report = evaluate_acceptance(image, hints(truth), result)
    heatmap = next(d for d in report.decisions if d.requirement == "heatmap")
    assert "unstable_heatmap_bounds" in heatmap.reasons
    assert report.overall.status != "accepted"


def test_resource_limit_fails_before_processing(monkeypatch):
    from dataclasses import replace

    import heatmap_decoder.detection as detection

    monkeypatch.setattr(detection, "A", replace(detection.A, max_image_pixels=100))
    with pytest.raises(ValueError, match="pixel cap"):
        detect(np.zeros((11, 11, 3), dtype=np.uint8), DetectionHints(rows=2, columns=2))


@pytest.mark.parametrize("schema", [1, 2])
def test_legacy_saved_schema_is_readable_and_review_only(tmp_path, schema):
    image, truth = FIXTURE["make_layout"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    saved = detect_file(figure, hints(truth), tmp_path / "roi")
    legacy = saved.model_dump(mode="json")
    legacy["schema_version"] = schema
    legacy["result"].pop("acceptance")
    legacy["automation_policy"] = {"automatic_roi_enabled": True}
    payload = config(truth).model_dump(mode="json")
    _, metadata = review_detection(
        json.dumps(legacy).encode(), image, saved.image_sha256, payload, None
    )
    assert not metadata["enforced_automation_policy"]["automatic_roi_enabled"]
    assert metadata["origin"] == "manual"


def test_numeric_only_never_drops_requested_identifiers(tmp_path):
    image, truth = FIXTURE["make_layout"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    request = hints(truth).model_copy(update={"requested": ("heatmap", "colourbar", "labels_rows")})
    hint_path = tmp_path / "hints.json"
    hint_path.write_text(request.model_dump_json())
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(figure),
            "--auto-roi",
            "--numeric-only",
            "--config",
            str(hint_path),
            "--out",
            str(tmp_path / "result"),
        ],
    )
    assert result.exit_code == 2
    assert "conflicts with requested gene/sample identities" in result.output
    assert not (tmp_path / "result/matrix.csv").exists()


def test_staged_label_review_cannot_switch_its_heatmap(tmp_path):
    image, truth = FIXTURE["make_layout"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    saved = detect_file(figure, hints(truth), tmp_path / "roi")
    raw = saved.model_dump(mode="json")
    box = truth["heatmap_bbox"]
    raw["result"]["candidates"].extend(
        [
            ROICandidate(
                id="other-panel",
                bbox=(box[0], box[1] + 22, box[2], box[3] + 22),
                score=1,
                score_components={},
            ).model_dump(mode="json"),
            ROICandidate(
                id="labels-other-panel",
                kind="labels_rows",
                bbox=truth["label_boxes"]["rows"],
                score=1,
                score_components={},
                associations=("other-panel",),
            ).model_dump(mode="json"),
        ]
    )
    with pytest.raises(ValueError, match="label region conflicts with selected heatmap"):
        review_detection(
            json.dumps(raw).encode(),
            image,
            saved.image_sha256,
            config(truth).model_dump(mode="json"),
            None,
            ROIReview(regions={"labels_rows": "labels-other-panel"}),
        )
