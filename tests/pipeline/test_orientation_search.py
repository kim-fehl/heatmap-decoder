"""Orientation search contracts and fixtures."""

import time

import numpy as np
import pytest

from heatmap_decoder import DetectionHints, detect
from heatmap_decoder.detection import _label_searches, count_proposals
from heatmap_decoder.labels import CropTransform
from heatmap_decoder.models import LabelsConfig, OCRRun, OCRToken
from heatmap_decoder.orientation import OrientationBudget, column_orientation_regions

BOX = (40, 300, 360, 460)


IMAGE = np.full((600, 420, 3), 255, np.uint8)


def hints(**changes):
    return DetectionHints(
        heatmap_bbox=BOX,
        rows=4,
        columns=4,
        requested=("labels_columns",),
        detector={"propose_column_rotations": True},
        **changes,
    )


def runner(monkeypatch, adequate=(90,), missing=(), confidence=1):
    searches = _label_searches(BOX, IMAGE.shape, 256)
    call = [0]

    def recognize(image, settings, **options):
        side_index, angle_index = divmod(call[0], 3)
        call[0] += 1
        angle = (0, 90, -90)[angle_index]
        tokens = []
        if side_index == 0 and angle in adequate:
            box = next(b for side, b in searches if side == "above")
            transform = CropTransform(box, settings.upscale_factor, angle)
            for i in range(4):
                if i in missing:
                    continue
                original = (
                    (70 + i * 80, 80, 90 + i * 80, 280)
                    if angle
                    else (60 + i * 80, 270, 108 + i * 80, 290)
                )
                tokens.append(OCRToken(f"Name {i}", transform.to_crop(original), confidence))
        return OCRRun(tuple(tokens), "test", "test", "eng", ("eng",), "test")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    return call


@pytest.mark.parametrize("angle", [0, 90, -90])
def test_unique_geometry_selects_angle_without_confidence_selecting_it(monkeypatch, angle):
    runner(monkeypatch, (angle,), confidence=1)
    proposals, records, _ = column_orientation_regions(
        IMAGE,
        hints(),
        "heatmap-manual",
        _label_searches(BOX, IMAGE.shape, 256),
        [16000000],
        OrientationBudget(12, 60),
    )
    assert len(records) == 6 and len(proposals) == 1
    assert proposals[0].text_rotation_degrees == angle
    unique = [r for r in records if r.orientation_status == "unique"]
    assert len(unique) == 1 and unique[0].count_geometry_eligible
    assert [s.raw_text for s in unique[0].assignment.slots] == [
        "Name 0",
        "Name 1",
        "Name 2",
        "Name 3",
    ]
    assert unique[0].raw_tokens != unique[0].tokens
    assert "text_identity_unverified" in proposals[0].reasons


def test_conflicting_signs_remain_review_required_and_do_not_feed_counts(monkeypatch):
    runner(monkeypatch, (90, -90), confidence=99)
    p, rs, _ = column_orientation_regions(
        IMAGE,
        hints(),
        "heatmap-manual",
        _label_searches(BOX, IMAGE.shape, 256),
        [16000000],
        OrientationBudget(12, 60),
    )
    assert len(p) == 2
    assert sum(r.orientation_status == "ambiguous" for r in rs) == 2
    assert not any(r.count_geometry_eligible for r in rs)


def test_missing_label_hole_retains_original_indices(monkeypatch):
    runner(monkeypatch, missing=(1,))
    _, rs, _ = column_orientation_regions(
        IMAGE,
        hints(),
        "heatmap-manual",
        _label_searches(BOX, IMAGE.shape, 256),
        [16000000],
        OrientationBudget(12, 60),
    )
    r = next(r for r in rs if r.orientation_status == "unique")
    assert r.assignment.slots[1].raw_text is None
    assert r.assignment.slots[2].raw_text == "Name 2"
    assert tuple(s.index for s in r.assignment.slots) == (0, 1, 2, 3)


@pytest.mark.parametrize("cap", ["pixels", "calls", "seconds"])
def test_resource_caps_record_skipped_trials(monkeypatch, cap):
    calls = runner(monkeypatch)
    pixels = [0 if cap == "pixels" else 16000000]
    budget = OrientationBudget(1 if cap == "calls" else 12, 60)
    if cap == "seconds":
        budget.started = time.monotonic() - 61
    _, rs, _ = column_orientation_regions(
        IMAGE, hints(), "heatmap-manual", _label_searches(BOX, IMAGE.shape, 256), pixels, budget
    )
    assert len(rs) == 6 and any(r.orientation_status == "capped" for r in rs)
    assert calls[0] == (1 if cap == "calls" else 0)


def test_unknown_count_uses_unassigned_centres_and_links_orientation(monkeypatch):
    runner(monkeypatch)
    h = DetectionHints(
        heatmap_bbox=BOX,
        rows=4,
        infer_columns=True,
        requested=("labels_columns",),
        detector={"propose_column_rotations": True, "count_label_geometry": True},
    )
    _, rs, _ = column_orientation_regions(
        IMAGE,
        h,
        "heatmap-manual",
        _label_searches(BOX, IMAGE.shape, 256),
        [16000000],
        OrientationBudget(12, 60),
    )
    assert all(r.assignment is None for r in rs)
    p = count_proposals(IMAGE, BOX, "columns", h, "heatmap-manual", rs)
    expected = next(p for p in p if p.count == 4)
    signal = next((k, v) for k, v in expected.signals.items() if k.startswith("ocr_"))
    assert "angle90" in signal[0]
    assert signal[1].positions == (80, 160, 240, 320)
    assert signal[1].observation == "unassigned_text_geometry"
    # Text alone is not sufficient independent support for a count acceptance.
    assert detect(IMAGE, h).acceptance.overall.status != "accepted"


def test_explicit_angle_runs_only_authoritative_trial(monkeypatch):
    calls = []

    def empty(image, settings, **options):
        calls.append(options)
        return OCRRun((), "test", "test", "eng", ("eng",), "test")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", empty)
    h = DetectionHints(
        heatmap_bbox=BOX,
        rows=4,
        infer_columns=True,
        requested=(),
        labels=LabelsConfig(
            columns={"source": "ocr", "bbox": (40, 44, 360, 300), "text_rotation_degrees": -90}
        ),
        detector={"propose_column_rotations": True, "count_label_geometry": True},
    )
    _, rs, _ = column_orientation_regions(
        IMAGE, h, "heatmap-manual", (), [16000000], OrientationBudget(12, 60)
    )
    assert len(calls) == 1 and len(rs) == 1 and rs[0].text_rotation_degrees == -90


def test_grouping_unassigned_centres_never_counts_recognised_strings(monkeypatch):
    runner(monkeypatch, missing=(1,))
    h = DetectionHints(
        heatmap_bbox=BOX,
        rows=4,
        infer_columns=True,
        requested=("labels_columns",),
        detector={"propose_column_rotations": True, "count_label_geometry": True},
    )
    _, rs, _ = column_orientation_regions(
        IMAGE,
        h,
        "heatmap-manual",
        _label_searches(BOX, IMAGE.shape, 256),
        [16000000],
        OrientationBudget(12, 60),
    )
    assert not any(r.count_geometry_eligible for r in rs)
    assert sum(len(r.unassigned_candidates) for r in rs) == 3


def test_incomplete_search_cannot_claim_unique_orientation(monkeypatch):
    runner(monkeypatch)
    proposals, records, _ = column_orientation_regions(
        IMAGE,
        hints(),
        "heatmap-manual",
        _label_searches(BOX, IMAGE.shape, 256),
        [16000000],
        OrientationBudget(2, 60),
    )
    assert len(proposals) == 1
    assert not any(r.orientation_status == "unique" for r in records)
    assert not any(r.count_geometry_eligible for r in records)
    assert "orientation_search_incomplete" in proposals[0].reasons


def test_opposite_angles_are_not_deduplicated_as_same_region(monkeypatch):
    runner(monkeypatch, (90, -90))
    result = detect(IMAGE, hints())
    region = next(d for d in result.acceptance.decisions if d.requirement == "labels_columns")
    assert region.status == "review_required"
    assert len(region.candidate_ids) == 2


def test_roi_review_cannot_override_an_explicit_configured_angle(monkeypatch):
    from heatmap_decoder.io import review_detection
    from heatmap_decoder.models import ROIReview, SavedDetection

    runner(monkeypatch)
    h = hints()
    detected = detect(IMAGE, h)
    candidate = next(c for c in detected.candidates if c.text_rotation_degrees == 90)
    saved = SavedDetection(
        schema_version=3,
        image_sha256="0" * 64,
        image_dimensions=(IMAGE.shape[1], IMAGE.shape[0]),
        detector_version="test",
        hints=h,
        result=detected,
        proposed_config_complete=False,
        missing_extraction_fields=("labels.columns",),
    )
    payload = {
        "heatmap_bbox": BOX,
        "rows": 4,
        "columns": 4,
        "colourbar_bbox": (380, 300, 396, 460),
        "colourbar_orientation": "vertical",
        "colourbar_start_value": -3,
        "colourbar_end_value": 5,
        "labels": {
            "columns": {"source": "ocr", "bbox": candidate.bbox, "text_rotation_degrees": -90}
        },
    }
    with pytest.raises(ValueError, match="configured labels.columns angle"):
        review_detection(
            saved.model_dump_json().encode(),
            IMAGE,
            "0" * 64,
            payload,
            None,
            ROIReview(regions={"labels_columns": candidate.id}),
        )
    payload["labels"]["columns"]["text_rotation_degrees"] = 90
    config, _ = review_detection(
        saved.model_dump_json().encode(),
        IMAGE,
        "0" * 64,
        payload,
        None,
        ROIReview(regions={"labels_columns": candidate.id}),
    )
    assert config.labels.columns.text_rotation_degrees == 90
