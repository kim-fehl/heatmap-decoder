"""Bounded column-text proposals; layout evidence never verifies spelling."""

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from .labels import (
    CropTransform,
    assign_candidates,
    group_tokens,
    group_unassigned_tokens,
    transform_tokens,
)
from .models import (
    BoundingBox,
    DetectionHints,
    GridConfig,
    LabelLocalisationEvidence,
    LabelsConfig,
    OCRLabelSource,
    OCRRun,
    RGBArray,
    ROICandidate,
)


@dataclass
class OrientationBudget:
    calls: int
    seconds: float
    started: float = field(default_factory=time.monotonic)


def column_orientation_regions(
    image: RGBArray,
    hints: DetectionHints,
    heatmap_id: str,
    searches: tuple[tuple[str, BoundingBox], ...],
    pixels_left: list[int],
    budget: OrientationBudget,
) -> tuple[tuple[ROICandidate, ...], tuple[LabelLocalisationEvidence, ...], tuple[str, ...]]:
    from .ocr import recognize

    if hints.detector.propose_arbitrary_column_angles or (
        hints.labels
        and isinstance(hints.labels.columns, OCRLabelSource)
        and hints.labels.columns.text_rotation_degrees not in (0, 90, -90)
    ):
        return arbitrary_column_regions(image, hints, heatmap_id, searches, pixels_left, budget)
    assert hints.heatmap_bbox is not None
    options = hints.detector
    settings = hints.labels.ocr if hints.labels else LabelsConfig().ocr
    source = hints.labels.columns if hints.labels else None
    explicit = isinstance(source, OCRLabelSource)
    angles: tuple[float, ...]
    if isinstance(source, OCRLabelSource):
        searches = (("above" if source.bbox[3] <= hints.heatmap_bbox[1] else "below", source.bbox),)
        angles = (source.text_rotation_degrees,)
    else:
        angles = (0, 90, -90)
    records, warnings = [], []
    for side, box in searches:
        if side not in ("above", "below"):
            continue
        for angle in angles:
            transform = CropTransform(box, settings.upscale_factor, angle)
            pixels = math.prod(transform.crop_size)
            identifier = f"labels_columns-{heatmap_id}-{side}-angle{angle}"
            remaining = budget.seconds - (time.monotonic() - budget.started)
            used, started = 0, time.monotonic()
            reasons = []
            raw = OCRRun((), "not_run", "not_run", settings.language, (), "not_run")
            if (
                budget.calls <= 0
                or remaining <= 0
                or pixels > pixels_left[0]
                or pixels > options.max_ocr_pixels
            ):
                reasons.append("orientation_resource_cap_exceeded")
            else:
                budget.calls -= 1
                pixels_left[0] -= pixels
                used = pixels
                try:
                    raw = recognize(
                        transform.prepare(Image.fromarray(image[:, :, :3])),
                        settings,
                        timeout_seconds=min(30.0, remaining),
                    )
                except (ValueError, OSError) as error:
                    reasons.append(str(error))
            original = transform_tokens(raw.tokens, transform)
            upright = group_unassigned_tokens(raw.tokens)
            grouped = tuple(replace(c, bbox=transform.to_original(c.bbox)) for c in upright)
            assignment = None
            adequate = False
            if hints.columns is not None:
                grid = GridConfig(
                    heatmap_bbox=hints.heatmap_bbox,
                    rows=hints.rows or 1,
                    columns=hints.columns,
                    labels=LabelsConfig(ocr=settings),
                    rows_segments=hints.rows_segments,
                    columns_segments=hints.columns_segments,
                )
                assignment = assign_candidates(
                    group_tokens(raw.tokens, grid, "columns", transform=transform),
                    grid,
                    "columns",
                    tokens=original,
                )
                aligned = [
                    s
                    for s in assignment.slots
                    if s.bbox is not None and s.status in ("accepted", "low_confidence")
                ]
                adequate = len(aligned) >= max(
                    min(2, hints.columns), math.ceil(0.75 * hints.columns)
                ) and not any(s.status == "ambiguous" for s in assignment.slots)
                boxes = [s.bbox for s in aligned if s.bbox]
            else:
                boxes = [c.bbox for c in grouped]
                centres = np.sort([(b[0] + b[2]) / 2 for b in boxes])
                gaps = np.diff(centres)
                adequate = (
                    len(centres) >= 2
                    and len(gaps) > 0
                    and np.min(gaps) > 0
                    and np.max(np.abs(gaps - np.median(gaps))) <= 0.1 * np.median(gaps)
                )
                if adequate:
                    adequate = bool(
                        abs(centres[0] - hints.heatmap_bbox[0] - np.median(gaps) / 2)
                        <= 0.25 * np.median(gaps)
                        and abs(hints.heatmap_bbox[2] - centres[-1] - np.median(gaps) / 2)
                        <= 0.25 * np.median(gaps)
                    )
            # Upright runs must yield text-shaped boxes in the trial's frame.
            shaped = [
                b[2] - b[0] >= 1.5 * (b[3] - b[1])
                if angle == 0
                else b[3] - b[1] >= 1.5 * (b[2] - b[0])
                for b in boxes
            ]
            adequate = adequate and bool(shaped) and sum(shaped) / len(shaped) >= 0.75
            if not adequate:
                reasons.append("insufficient_layout_consistency")
            records.append(
                LabelLocalisationEvidence(
                    axis="columns",
                    side=cast(Literal["above", "below"], side),
                    search_bbox=box,
                    preprocessing=settings,
                    tokens=original,
                    raw_tokens=raw.tokens,
                    unassigned_candidates=grouped,
                    assignment=assignment,
                    engine=raw,
                    heatmap_id=heatmap_id,
                    trial_id=identifier,
                    text_rotation_degrees=cast(Literal[0, 90, -90], angle),
                    transform={
                        "bbox": box,
                        "upscale_factor": settings.upscale_factor,
                        "text_rotation_degrees": angle,
                        "crop_size": transform.crop_size,
                    },
                    ocr_pixels=used,
                    ocr_seconds=time.monotonic() - started,
                    orientation_status="capped" if used == 0 else "unsupported",
                    orientation_reasons=tuple(reasons),
                    count_geometry_eligible=False,
                )
            )
    supported = [r for r in records if not r.orientation_reasons]
    # Competing signs/sides are retained, even if they produce similar spelling.
    exhaustive = all(r.ocr_pixels > 0 and r.engine.engine_version != "not_run" for r in records)
    unique = len(supported) == 1 and exhaustive
    proposals = []
    final = []
    for record in records:
        eligible = record in supported
        status = (
            "unique"
            if eligible and unique
            else "ambiguous"
            if eligible
            else record.orientation_status
        )
        final_reasons = record.orientation_reasons
        if eligible and not unique:
            final_reasons += (
                "contradictory_orientation_or_side_hypotheses"
                if exhaustive
                else "orientation_search_incomplete",
            )
        if eligible:
            final_reasons += (
                "explicit_angle_authoritative" if explicit else "layout_orientation_only",
                "text_identity_unverified",
            )
        final.append(
            record.model_copy(
                update={
                    "orientation_status": status,
                    "orientation_reasons": final_reasons,
                    "count_geometry_eligible": eligible and unique,
                }
            )
        )
        if eligible:
            boxes = (
                [
                    s.bbox
                    for s in record.assignment.slots
                    if s.bbox is not None and s.status in ("accepted", "low_confidence")
                ]
                if record.assignment
                else [c.bbox for c in record.unassigned_candidates]
            )
            padding = options.label_padding_pixels
            box = record.search_bbox
            crop = (
                max(box[0], math.floor(min(b[0] for b in boxes)) - padding),
                max(box[1], math.floor(min(b[1] for b in boxes)) - padding),
                min(box[2], math.ceil(max(b[2] for b in boxes)) + padding),
                min(box[3], math.ceil(max(b[3] for b in boxes)) + padding),
            )
            proposals.append(
                ROICandidate(
                    id=record.trial_id or "",
                    kind="labels_columns",
                    bbox=crop,
                    score=1 if unique else 0.5,
                    score_components={"unique_layout_hypothesis": float(unique)},
                    reasons=("review_required", *final_reasons),
                    associations=(heatmap_id,),
                    text_rotation_degrees=record.text_rotation_degrees,
                )
            )
        elif final_reasons:
            warnings.append(f"{record.trial_id}: " + "; ".join(final_reasons))
    return tuple(proposals), tuple(final), tuple(warnings)


def arbitrary_column_regions(
    image: RGBArray,
    hints: DetectionHints,
    heatmap_id: str,
    searches: tuple[tuple[str, BoundingBox], ...],
    pixels_left: list[int],
    budget: OrientationBudget,
) -> tuple[tuple[ROICandidate, ...], tuple[LabelLocalisationEvidence, ...], tuple[str, ...]]:
    """M18 path: geometric angle evidence precedes bounded isolated line OCR."""
    from .angle_search import estimate_angle
    from .models import OCRRun
    from .oriented_labels import LabelBudget, recognise_lines

    assert hints.heatmap_bbox is not None
    settings = hints.labels.ocr if hints.labels else LabelsConfig().ocr
    configured = hints.labels.columns if hints.labels else None
    if isinstance(configured, OCRLabelSource):
        searches = (
            ("above" if configured.bbox[3] <= hints.heatmap_bbox[1] else "below", configured.bbox),
        )
    if not isinstance(configured, OCRLabelSource):
        padding = hints.detector.label_search_pixels
        searches = tuple(
            (
                side,
                (max(0, box[0] - padding), box[1], min(image.shape[1], box[2] + padding), box[3]),
            )
            for side, box in searches
            if side in ("above", "below")
        )
    shared = LabelBudget(budget.calls, pixels_left[0], budget.seconds, budget.started)
    records = []
    for side, box in searches:
        if side not in ("above", "below"):
            continue
        if isinstance(configured, OCRLabelSource):
            estimation: dict[str, Any] = {
                "status": "explicit",
                "angle": configured.text_rotation_degrees,
                "angle_alternatives": (configured.text_rotation_degrees,),
                "reading_direction": "explicit_angle",
            }
        else:
            estimation = estimate_angle(image, box, shared)
        angles = estimation.get("angle_alternatives", ())
        if not angles:
            # Keep exhausted/unsupported geometric trials as an auditable skipped record.
            records.append(
                LabelLocalisationEvidence(
                    axis="columns",
                    side=cast(Literal["above", "below"], side),
                    search_bbox=box,
                    preprocessing=settings,
                    tokens=(),
                    unassigned_candidates=(),
                    engine=OCRRun((), "not_run", "not_run", settings.language, (), "not_run"),
                    heatmap_id=heatmap_id,
                    orientation_status="capped"
                    if shared.remaining_seconds <= 0 or shared.pixels <= 0
                    else "unsupported",
                    orientation_reasons=(estimation.get("reason", "unresolved_orientation"),),
                    transform={"angle_estimation": estimation},
                    count_geometry_eligible=False,
                )
            )
        for angle in angles:
            source = OCRLabelSource(
                source="ocr",
                bbox=box,
                recognition="lines",
                text_rotation_degrees=angle,
                attachment_endpoint=configured.attachment_endpoint
                if isinstance(configured, OCRLabelSource)
                else hints.detector.column_attachment_endpoint,
                max_attachment_distance_pixels=configured.max_attachment_distance_pixels
                if isinstance(configured, OCRLabelSource)
                else 40,
            )
            identifier = f"labels_columns-{heatmap_id}-{side}-angle{angle:g}"
            reasons = []
            if (
                math.prod(CropTransform(box, settings.upscale_factor, angle).crop_size)
                > hints.detector.max_ocr_pixels
            ):
                reasons.append("orientation_resource_cap_exceeded")
            if hints.columns is None or reasons:
                reasons.append("resolved_columns_required_for_attachment")
                assignment = None
                line_records: tuple[dict[str, Any], ...] = ()
                skipped: tuple[dict[str, Any], ...] = ()
            else:
                config = GridConfig(
                    heatmap_bbox=hints.heatmap_bbox,
                    rows=hints.rows or 1,
                    columns=hints.columns,
                    rows_segments=hints.rows_segments,
                    columns_segments=hints.columns_segments,
                    labels=LabelsConfig(columns=source, ocr=settings),
                )
                assignment, line_records, skipped = recognise_lines(image, config, source, shared)
                geometry = assignment.geometry_evidence or {}
                if geometry.get("status") != "accepted" or geometry.get("resolved_lines", 0) < max(
                    2, math.ceil(0.75 * hints.columns)
                ):
                    reasons.append("insufficient_attachment_geometry")
                if (
                    not isinstance(configured, OCRLabelSource)
                    and source.attachment_endpoint == "nearest"
                    and geometry.get("reading_direction_evidence", {}).get("status") == "inverted"
                ):
                    reasons.append("geometric_upside_down_baseline")
                if any(
                    s["reason"] != "geometry_requires_review" or s.get("slot") is not None
                    for s in skipped
                ):
                    reasons.append("orientation_search_incomplete")
            if estimation["status"] not in ("unique_axis", "explicit"):
                reasons.append("orientation_geometry_requires_review")
            records.append(
                LabelLocalisationEvidence(
                    axis="columns",
                    side=cast(Literal["above", "below"], side),
                    search_bbox=box,
                    preprocessing=settings,
                    tokens=assignment.tokens if assignment else (),
                    unassigned_candidates=assignment.candidates if assignment else (),
                    assignment=assignment,
                    engine=OCRRun(
                        (),
                        "oriented_line_aggregate",
                        "see_line_records",
                        settings.language,
                        (),
                        "isolated_lines",
                    ),
                    heatmap_id=heatmap_id,
                    trial_id=identifier,
                    text_rotation_degrees=source.text_rotation_degrees,
                    transform={
                        "angle_estimation": estimation,
                        "line_records": line_records,
                        "skipped_attempts": skipped,
                        "source_transform": CropTransform(
                            box, settings.upscale_factor, angle
                        ).metadata(),
                    },
                    ocr_pixels=sum(r["ocr_pixels"] for r in line_records),
                    ocr_seconds=sum(r["ocr_seconds"] for r in line_records),
                    orientation_status="unsupported" if reasons else "unique",
                    orientation_reasons=tuple(reasons),
                    count_geometry_eligible=False,
                )
            )
    supported = [r for r in records if not r.orientation_reasons]
    exhaustive = all(
        not any(
            t.get("status") == "skipped"
            for t in r.transform.get("angle_estimation", {}).get("trials", [])
        )
        and "orientation_search_incomplete" not in r.orientation_reasons
        for r in records
    )
    unique = len(supported) == 1 and exhaustive
    proposals = []
    final = []
    for record in records:
        if record not in supported:
            final.append(record)
            continue
        final_reasons = (
            ()
            if unique
            else (
                "contradictory_orientation_or_side_hypotheses",
                "upside_down_or_side_alternative_unresolved",
            )
        )
        final_record = record.model_copy(
            update={
                "orientation_status": "unique" if unique else "ambiguous",
                "orientation_reasons": final_reasons + ("text_identity_unverified",),
            }
        )
        final.append(final_record)
        proposals.append(
            ROICandidate(
                id=record.trial_id or "",
                kind="labels_columns",
                bbox=record.search_bbox,
                score=1 if unique else 0.5,
                score_components={"unique_attachment_geometry": float(unique)},
                reasons=("review_required", *final_record.orientation_reasons),
                associations=(heatmap_id,),
                text_rotation_degrees=record.text_rotation_degrees,
            )
        )
    budget.calls = shared.calls
    pixels_left[0] = shared.pixels
    return (
        tuple(proposals),
        tuple(final),
        tuple(f"{r.trial_id}: {r.orientation_reasons}" for r in final if r.orientation_reasons),
    )
