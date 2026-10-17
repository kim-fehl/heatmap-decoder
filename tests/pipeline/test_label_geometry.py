"""Label geometry contracts and fixtures."""

from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.m18.fixtures import fixture
from heatmap_decoder.angle_search import estimate_angle
from heatmap_decoder.extract import UnresolvedLabelsError, extract, extract_labels
from heatmap_decoder.geometry import resolve_grid
from heatmap_decoder.io import label_provenance
from heatmap_decoder.labels import (
    CropTransform,
    assign_candidates,
    group_tokens,
    transform_tokens,
)
from heatmap_decoder.models import GridConfig, OCRRun, OCRToken
from heatmap_decoder.oriented_labels import (
    LabelBudget,
    associate_lines,
    recognise_lines,
    segment_lines,
)
from tests.support.angle_fixtures import ANGLES
from tests.support.rotation_fixtures import config


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


@pytest.mark.parametrize("angle", [90, -90])
@pytest.mark.parametrize("recognition", ["sparse", "slots"])
def test_group_upright_assign_original_x_and_keep_run_links(monkeypatch, angle, recognition):
    cfg = config(angle, recognition)
    original = []
    # Two separated words in each original vertical label; +/- signs reverse upright line order.
    for i in range(3):
        x = 24 + 40 * i
        boxes = [(x, 25, x + 10, 45), (x, 49, x + 10, 75)]
        if angle == 90:
            boxes.reverse()
        original.extend(
            [
                OCRToken(f"Sample{i}", boxes[0], 95, (1, 1, 1)),
                OCRToken("A_b-C", boxes[1], 94, (1, 1, 1)),
            ]
        )
    calls = []

    def recognize(image, settings, *, mode="localisation", timeout_seconds=30):
        index = len(calls)
        bbox = cfg.labels.columns.bbox
        if recognition == "slots":
            bbox = (10 + 40 * index, 20, 50 + 40 * index, 100)
            selected = original[index * 2 : index * 2 + 2]
        else:
            selected = original
        transform = CropTransform(bbox, settings.upscale_factor, angle)
        assert image.size == transform.crop_size
        calls.append(mode)
        tokens = tuple(replace(t, bbox=transform.to_crop(t.bbox)) for t in selected)
        return OCRRun(tokens, "fake", "fake", "eng", ("eng",), mode)

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    result = extract_labels(np.full((160, 140, 3), 255, dtype=np.uint8), cfg)
    columns = result.axes[1]
    assert [s.final_text for s in columns.slots] == [f"Sample{i} A_b-C" for i in range(3)]
    assert len(columns.tokens) == 6
    assert all(len(c.token_indices) == 2 for c in columns.candidates)
    assert columns.tokens == tuple(original)
    evidence = label_provenance(
        result, input_filename="fake.png", input_sha256="x", image_shape=(160, 140, 3)
    )
    transforms = (
        list(evidence["ocr"]["transforms"].values())
        if recognition == "sparse"
        else [r["transform"] for r in evidence["ocr"]["slot_runs"]]
    )
    assert all(t["text_rotation_degrees"] == angle for t in transforms)
    assert all(t["grouping_frame"] == "upright OCR pixels" for t in transforms)


@pytest.mark.parametrize("angle", [90, -90])
def test_rotated_missing_duplicate_and_span_behaviour(angle):
    cfg = config(angle)
    transform = CropTransform(cfg.labels.columns.bbox, 3, angle)
    originals = [
        OCRToken("Duplicate", (24, 25, 34, 70), 95, (1, 1, 1)),
        OCRToken("Duplicate", (104, 25, 114, 70), 95, (1, 1, 1)),
    ]
    tokens = tuple(replace(t, bbox=transform.to_crop(t.bbox)) for t in originals)
    candidates = group_tokens(tokens, cfg, "columns", transform=transform)
    result = assign_candidates(
        candidates, cfg, "columns", tokens=transform_tokens(tokens, transform)
    )
    assert result.slots[1].status == "missing"
    assert result.slots[0].duplicate and result.slots[2].duplicate
    spanning = OCRToken("span", transform.to_crop((40, 25, 65, 70)), 95, (1, 1, 1))
    result = assign_candidates(
        group_tokens([spanning], cfg, "columns", transform=transform), cfg, "columns"
    )
    assert result.slots[0].status == result.slots[1].status == "ambiguous"


def test_trimmed_rotated_slot_preserves_original_coordinates_and_review(monkeypatch):
    image = np.full((80, 50, 3), 255, np.uint8)
    image[40:60, 15:25] = 0
    config = GridConfig.model_validate(
        {
            "heatmap_bbox": [10, 0, 30, 20],
            "rows": 1,
            "columns": 1,
            "labels": {
                "columns": {
                    "source": "ocr",
                    "recognition": "slots",
                    "bbox": [10, 30, 30, 80],
                    "text_rotation_degrees": 90,
                    "trim_whitespace": True,
                    "require_review": True,
                },
                "ocr": {"engine": "rapidocr", "upscale_factor": 2},
            },
        }
    )
    seen = []

    def recognize(crop, settings, **kwargs):
        seen.append(crop.size)
        return OCRRun(
            (OCRToken("NC1", (0, 0, crop.width, crop.height), 99, (1, 1, 1)),),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "fake",
            geometry_kind="source_field",
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    result = extract_labels(image, config)
    assert seen == [(44, 24)]
    assert result.slot_ocr_runs[0][2] == (14, 39, 26, 61)
    slot = result.axes[1].slots[0]
    assert slot.raw_text == "NC1"
    assert slot.final_text is None and slot.status == "unreviewed"
    assert slot.bbox == (14, 39, 26, 61)
    assert "configured_identifier_review_required" in slot.issues


@pytest.mark.parametrize("policy", ["latin_greek", "unrestricted"])
def test_both_axes_preserve_raw_tokens_and_apply_review_to_cleaned_suggestions(monkeypatch, policy):
    from heatmap_decoder import ocr

    raw = deque(["東BRCA1", "IL-1β", "株Sample 1", "Sample α-2"])
    config = GridConfig(
        heatmap_bbox=(0, 0, 20, 20),
        rows=2,
        columns=2,
        labels=dict(
            character_policy=policy,
            rows=dict(source="ocr", recognition="slots", bbox=(20, 0, 40, 20)),
            columns=dict(source="ocr", recognition="slots", bbox=(0, 20, 20, 40)),
        ),
    )

    def recognize(image, settings, **kwargs):
        return OCRRun(
            (OCRToken(raw.popleft(), (0, 0, image.width, image.height), 99),),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "text_line",
            geometry_kind="source_field",
        )

    monkeypatch.setattr(ocr, "recognize", recognize)
    result = extract_labels(np.full((40, 40, 3), 255, dtype=np.uint8), config)
    rows, columns = result.axes
    assert [s.raw_text for s in rows.slots] == ["東BRCA1", "IL-1β"]
    assert [s.raw_text for s in columns.slots] == ["株Sample 1", "Sample α-2"]
    assert [run.tokens[0].text for _, _, _, run in result.slot_ocr_runs] == [
        "東BRCA1",
        "IL-1β",
        "株Sample 1",
        "Sample α-2",
    ]
    assert rows.slots[1].final_text == "IL-1β"
    assert columns.slots[1].final_text == "Sample α-2"
    for slot, suggestion in ((rows.slots[0], "BRCA1"), (columns.slots[0], "Sample 1")):
        if policy == "latin_greek":
            assert slot.final_text is None and slot.suggested_text == suggestion
            assert slot.status == "unreviewed"
            assert "unexpected_label_characters" in slot.issues
        else:
            assert slot.final_text == slot.raw_text
            assert slot.suggested_text is None and slot.status == "accepted"
