"""Bounded column-text proposals; layout evidence never verifies spelling."""

import math
import time
from dataclasses import dataclass, field, replace
from typing import Literal, cast

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
    OCRConfig,
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

    assert hints.heatmap_bbox is not None
    options = hints.detector
    settings = hints.labels.ocr if hints.labels else OCRConfig()
    source = hints.labels.columns if hints.labels else None
    explicit = isinstance(source, OCRLabelSource)
    angles: tuple[int, ...]
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
