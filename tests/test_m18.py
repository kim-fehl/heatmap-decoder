"""Affine and attachment geometry independent of OCR spelling/engine confidence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from pydantic import ValidationError

from benchmarks.m18.fixtures import fixture
from heatmap_decoder.angle_search import estimate_angle
from heatmap_decoder.extract import UnresolvedLabelsError, extract, extract_labels
from heatmap_decoder.geometry import resolve_grid
from heatmap_decoder.io import label_provenance
from heatmap_decoder.labels import CropTransform
from heatmap_decoder.models import LabelsConfig, OCRLabelSource, OCRRun, OCRToken
from heatmap_decoder.oriented_labels import (
    LabelBudget,
    associate_lines,
    recognise_lines,
    segment_lines,
)

ANGLES = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]


@pytest.mark.parametrize("angle", ANGLES + [37.25, -22.7, 150, -150, 180, 370])
@pytest.mark.parametrize("factor", [1, 3])
def test_affine_point_polygon_round_trips(angle, factor):
    transform = CropTransform((17, 23, 142, 78), factor, angle, 2)
    polygon = ((17.5, 23.5), (100, 25.2), (130.6, 60), (20, 70.2))
    assert np.allclose(
        transform.polygon_to_original(transform.polygon_to_crop(polygon)), polygon, atol=1e-10
    )
    for p in polygon:
        assert transform.point_to_original(transform.point_to_crop(p)) == pytest.approx(
            p, abs=1e-10
        )
    metadata = transform.metadata()
    assert len(metadata["forward_affine"]) == len(metadata["inverse_affine"]) == 6
    if angle not in (0, 90, -90):
        assert "expanded" in metadata["interpolation"]


@pytest.mark.parametrize(
    "value", [True, False, float("nan"), float("inf"), -float("inf"), "45", None]
)
def test_angles_reject_invalid(value):
    with pytest.raises(ValidationError):
        OCRLabelSource(source="ocr", bbox=(0, 0, 10, 20), text_rotation_degrees=value)
    with pytest.raises(ValueError):
        CropTransform((0, 0, 10, 20), text_rotation_degrees=value)


@pytest.mark.parametrize(
    "value,expected", [(90.0, 90), (370, 10), (-190, 170), (180, -180), (720, 0), (37.25, 37.25)]
)
def test_angle_normalisation(value, expected):
    assert (
        OCRLabelSource(
            source="ocr", bbox=(0, 0, 10, 20), text_rotation_degrees=value
        ).text_rotation_degrees
        == expected
    )


def test_gene_angles_remain_unsupported():
    with pytest.raises(ValidationError, match="columns only"):
        LabelsConfig(rows={"source": "ocr", "bbox": (0, 0, 20, 30), "text_rotation_degrees": 15})


@pytest.mark.parametrize("angle", ANGLES + [37.25, -37.25, 150, -150])
@pytest.mark.parametrize("below", [False, True])
@pytest.mark.parametrize("blocked", [False, True])
def test_explicit_line_attachment_order(angle, below, blocked):
    image, c, _, _, truth = fixture(angle, below, blocked)
    _, _, lines, _ = segment_lines(image, c.labels.columns)
    lines, geometry = associate_lines(lines, c, c.labels.columns)
    assert geometry["status"] == "accepted"
    assert len(lines) == 4
    assert sorted(line.slot_index for line in lines) == [0, 1, 2, 3]
    for line in lines:
        expected = truth[line.slot_index]["attachment"]
        assert np.abs(np.array(line.attachment) - expected).max() < 6
        assert len(line.polygon) == 4 and len(line.baseline) == 2
    assert geometry["grid_sha256"] == resolve_grid(c).metadata()["sha256"]


@pytest.mark.parametrize("derivative", ["jpeg", "blur"])
@pytest.mark.parametrize("angle", [15, -30, 45, -60, 75, -90])
def test_ordinary_derivatives_preserve_associations(derivative, angle):
    im, c, *_ = fixture(angle, True, True, derivative=derivative)
    _, _, ls, _ = segment_lines(im, c.labels.columns)
    ls, g = associate_lines(ls, c, c.labels.columns)
    assert g["resolved_lines"] == 4
    assert sorted(line.slot_index for line in ls if not line.issues) == [0, 1, 2, 3]


def fake_engine(monkeypatch, readings=("Sample A_1",)):
    calls = []

    def recognize(image, settings, **kwargs):
        text = readings[len(calls) % len(readings)]
        calls.append(kwargs)
        return OCRRun(
            (OCRToken(text, (0, 0, image.width, image.height), 99),),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "text_line",
            geometry_kind="source_field",
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    return calls


def test_line_ocr_retains_polygons_raw_readings_and_unreviewed_identifiers(monkeypatch):
    calls = fake_engine(monkeypatch)
    im, c, *_ = fixture(30, blocked=True)
    result = extract_labels(im, c)
    columns = result.axes[1]
    assert len(calls) == 4
    assert all(s.status == "unreviewed" and s.final_text is None for s in columns.slots)
    assert all(s.raw_text == "Sample A_1" and s.duplicate for s in columns.slots)
    assert result.unresolved_count == 4
    assert all(
        candidate.polygon and candidate.baseline and candidate.attachment
        for candidate in columns.candidates
    )
    p = label_provenance(result, input_filename="f.png", input_sha256="x", image_shape=im.shape)
    assert len(p["ocr"]["oriented_lines"]) == 4
    assert all(
        r["heuristic_text_acceptance"] and r["identifier_status"] == "unreviewed"
        for r in p["ocr"]["oriented_lines"]
    )
    assert p["grid_sha256"] == columns.geometry_evidence["grid_sha256"]


def test_numeric_invariance_and_review_gate(monkeypatch):
    fake_engine(monkeypatch)
    im, _, c, _, _ = fixture(-45, True, True)
    numeric = extract(im, c.model_copy(update={"labels": None}))
    with pytest.raises(UnresolvedLabelsError):
        extract(im, c)
    labelled = extract(im, c, reviewed_labels={("columns", i): f"Exact_{i}" for i in range(4)})
    assert np.array_equal(numeric.values, labelled.values)
    assert np.array_equal(numeric.flags, labelled.flags)
    assert resolve_grid(c).metadata()["sha256"] == resolve_grid(numeric.config).metadata()["sha256"]


@pytest.mark.parametrize("cap", ["calls", "pixels", "seconds"])
def test_shared_budgets_keep_skipped_trials(monkeypatch, cap):
    calls = fake_engine(monkeypatch)
    im, c, *_ = fixture(45)
    budget = LabelBudget(
        calls=1 if cap == "calls" else 128,
        pixels=0 if cap == "pixels" else 16000000,
        seconds=0 if cap == "seconds" else 60,
    )
    _, records, skipped = recognise_lines(im, c, c.labels.columns, budget)
    assert len(calls) == (1 if cap == "calls" else 0)
    if cap == "calls":
        assert skipped and all(r["reason"] == "shared_budget_exhausted" for r in skipped)
    else:
        assert not records


def test_conflicting_retries_are_separate_readings(monkeypatch):
    fake_engine(monkeypatch, ("Raw A_1", "Raw B 7"))
    im, c, *_ = fixture(30)
    labels = c.labels.model_copy(
        update={"ocr": c.labels.ocr.model_copy(update={"retry_variants": ("contrast",)})}
    )
    result = extract_labels(im, c.model_copy(update={"labels": labels}))
    assert all(
        s.status == "ambiguous" and s.raw_text == "Raw A_1\nRaw B 7" for s in result.axes[1].slots
    )
    assert len(result.oriented_ocr_records) == 8


def test_missing_slots_and_headers_do_not_shift_ownership():
    im, c, *_ = fixture(-45, True, True, missing=(1,), header=True)
    _, _, ls, _ = segment_lines(im, c.labels.columns)
    ls, g = associate_lines(ls, c, c.labels.columns)
    assert sorted(
        line.slot_index for line in ls if line.slot_index is not None and not line.issues
    ) == [
        0,
        2,
        3,
    ]
    assert g["missing_slots"] == [1]
    assert any(line.slot_index is None and line.issues for line in ls)


def test_wrong_alignment_and_overlapping_attachments_require_review():
    im, c, *_ = fixture(45)
    source = c.labels.columns.model_copy(update={"attachment_endpoint": "end"})
    _, _, ls, _ = segment_lines(im, source)
    ls, g = associate_lines(ls, c, source)
    assert g["status"] == "review_required"
    assert any("configured_endpoint_faces_away_from_matrix" in line.issues for line in ls)
    source = c.labels.columns
    _, _, ls, _ = segment_lines(im, source)
    ls, g = associate_lines(ls + (ls[0],), c, source)
    assert g["status"] == "review_required"
    assert any("overlapping_attachments" in line.issues for line in ls)


@pytest.mark.parametrize("angle", [15, -30, 45, -60, 75, -90, 37.25])
def test_geometric_angle_search_is_bounded_and_preserves_flip_alternative(angle):
    im, c, *_ = fixture(angle)
    r = estimate_angle(im, c.labels.columns.bbox, LabelBudget())
    assert r["status"] == "unique_axis"
    error = abs((r["angle"] - angle + 90) % 180 - 90)
    assert error <= 1
    assert len(r["trials"]) <= 21
    assert len(r["angle_alternatives"]) == 2
    assert r["policy"]["reading_direction_from_confidence"] is False


def test_automatic_reading_direction_uses_attachment_assumption_not_confidence(monkeypatch):
    from heatmap_decoder.models import DetectionHints
    from heatmap_decoder.orientation import OrientationBudget, arbitrary_column_regions

    fake_engine(monkeypatch)
    for endpoint, expected in [("nearest", "ambiguous"), ("start", "unique")]:
        im, c, *_ = fixture(30, names=["SAMPLE1234"] * 4)
        h = DetectionHints(
            heatmap_bbox=c.heatmap_bbox,
            rows=4,
            columns=4,
            requested=("labels_columns",),
            labels={"ocr": {"engine": "tesseract", "upscale_factor": 1}},
            detector={
                "propose_arbitrary_column_angles": True,
                "column_attachment_endpoint": endpoint,
            },
        )
        proposals, records, _ = arbitrary_column_regions(
            im, h, "h", (("above", c.labels.columns.bbox),), [16000000], OrientationBudget(12, 60)
        )
        supported = [r for r in records if r.orientation_status in ("unique", "ambiguous")]
        assert supported and all(r.orientation_status == expected for r in supported)
        assert len(proposals) == (2 if endpoint == "nearest" else 1)
        assert all(not r.count_geometry_eligible for r in records)
        assert all("text_identity_unverified" in p.reasons for p in proposals)


def test_incomplete_angle_search_cannot_become_unique(monkeypatch):
    from heatmap_decoder.models import DetectionHints
    from heatmap_decoder.orientation import OrientationBudget, arbitrary_column_regions

    fake_engine(monkeypatch)
    im, c, *_ = fixture(30)
    h = DetectionHints(
        heatmap_bbox=c.heatmap_bbox,
        rows=4,
        columns=4,
        requested=("labels_columns",),
        labels={"ocr": {"engine": "tesseract", "upscale_factor": 1}},
        detector={"propose_arbitrary_column_angles": True, "column_attachment_endpoint": "start"},
    )
    proposals, records, _ = arbitrary_column_regions(
        im, h, "h", (("above", c.labels.columns.bbox),), [50000], OrientationBudget(1, 60)
    )
    assert not any(r.orientation_status == "unique" for r in records)
    assert not proposals


@pytest.mark.parametrize("angle", [45, -45, 60, -60])
def test_tight_but_resolvable_long_lines_cross_multiple_bands(angle):
    im, c, *_ = fixture(angle, pitch=48, blocked=True)
    _, _, lines, _ = segment_lines(im, c.labels.columns)
    lines, g = associate_lines(lines, c, c.labels.columns)
    assert g["complete"] and g["status"] == "accepted"
    assert sorted(line.slot_index for line in lines) == [0, 1, 2, 3]


def test_truncated_name_keeps_unresolved_geometry():
    im, c, *_ = fixture(45)
    source = c.labels.columns.model_copy(update={"bbox": (300, *c.labels.columns.bbox[1:])})
    _, _, lines, _ = segment_lines(im, source)
    lines, g = associate_lines(lines, c, source)
    assert not g["complete"] or g["status"] == "review_required"
    assert any("source_region_truncation_possible" in line.issues for line in lines)


def test_glyph_baseline_geometry_resolves_flip_without_spelling_or_confidence(monkeypatch):
    from heatmap_decoder.models import DetectionHints
    from heatmap_decoder.orientation import OrientationBudget, arbitrary_column_regions

    fake_engine(monkeypatch)
    im, c, *_ = fixture(30)
    h = DetectionHints(
        heatmap_bbox=c.heatmap_bbox,
        rows=4,
        columns=4,
        requested=("labels_columns",),
        labels={"ocr": {"engine": "tesseract", "upscale_factor": 1}},
        detector={"propose_arbitrary_column_angles": True},
    )
    proposals, records, _ = arbitrary_column_regions(
        im, h, "h", (("above", c.labels.columns.bbox),), [16000000], OrientationBudget(12, 60)
    )
    assert len(proposals) == 1 and abs(proposals[0].text_rotation_degrees - 30) <= 1
    assert any("geometric_upside_down_baseline" in r.orientation_reasons for r in records)
    assert all(
        s.status == "unreviewed"
        for s in next(r.assignment for r in records if r.orientation_status == "unique").slots
    )
