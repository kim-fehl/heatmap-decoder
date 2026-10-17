"""Whole-region oriented lines, matrix-side attachments and bounded recognition.

Spelling never influences grouping, attachment or column ownership. The numeric
grid stays in source coordinates. New line OCR is evidence requiring identifier
review, regardless of engine confidence.
"""

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage  # type: ignore[import-untyped]

from .geometry import axis_slots, resolve_grid
from .labels import CropTransform
from .models import (
    AxisLabelResults,
    GridConfig,
    LabelCandidate,
    LabelResult,
    LabelStatus,
    OCRLabelSource,
    OCRToken,
    RGBArray,
    TextBox,
)

LINE_POLICY: dict[str, Any] = {
    "version": "oriented-attachments.1",
    "maximum_pixels": 16000000,
    "maximum_calls": 128,
    "maximum_seconds": 60.0,
    "maximum_lines": 128,
    "ink_luminance": 175,
    "component_minimum_pixels": 3,
    "glyph_gap_height_fraction": 2.0,
    "line_padding_pixels": 2,
}


@dataclass
class LabelBudget:
    calls: int = 128
    pixels: int = 16000000
    seconds: float = 60.0
    started: float = field(default_factory=time.monotonic)

    @property
    def remaining_seconds(self) -> float:
        return self.seconds - (time.monotonic() - self.started)

    def reserve(self, pixels: int, *, ocr: bool = False) -> bool:
        if pixels > self.pixels or self.remaining_seconds <= 0 or ocr and self.calls <= 0:
            return False
        self.pixels -= pixels
        if ocr:
            self.calls -= 1
        return True


@dataclass(frozen=True)
class OrientedLine:
    index: int
    normalised_bbox: tuple[int, int, int, int]
    polygon: tuple[tuple[float, float], ...]
    baseline: tuple[tuple[float, float], ...]
    attachment: tuple[float, float] | None = None
    attachment_interpretation: str = "not_associated"
    slot_index: int | None = None
    issues: tuple[str, ...] = ()


def bounds(polygon: tuple[tuple[float, float], ...]) -> TextBox:
    return (
        min(x for x, y in polygon),
        min(y for x, y in polygon),
        max(x for x, y in polygon),
        max(y for x, y in polygon),
    )


def segment_lines(
    image: RGBArray, source: OCRLabelSource, factor: int = 1, budget: LabelBudget | None = None
) -> tuple[Image.Image | None, CropTransform, tuple[OrientedLine, ...], dict[str, Any]]:
    """Deskew the complete region before any narrow crop or column assignment."""
    transform = CropTransform(
        source.bbox, factor, source.text_rotation_degrees, source.white_padding_pixels
    )
    pixels = math.prod(transform.crop_size)
    metadata: dict[str, Any] = {
        "policy": LINE_POLICY,
        "transform": transform.metadata(),
        "source_bbox": source.bbox,
        "segmentation_pixels": pixels,
    }
    if pixels > LINE_POLICY["maximum_pixels"] or budget is not None and not budget.reserve(pixels):
        return (
            None,
            transform,
            (),
            {**metadata, "status": "capped", "reason": "segmentation_budget_exhausted"},
        )
    prepared = transform.prepare(Image.fromarray(image[:, :, :3]))
    ink = np.asarray(prepared) < LINE_POLICY["ink_luminance"]
    components, _ = ndimage.label(ink)
    slices = ndimage.find_objects(components)
    heights = [
        s[0].stop - s[0].start for s in slices if s is not None and np.count_nonzero(ink[s]) >= 3
    ]
    glyph_height = (
        float(np.median([h for h in heights if h >= 3])) if any(h >= 3 for h in heights) else 3.0
    )
    gap = max(3, min(65, round(glyph_height * LINE_POLICY["glyph_gap_height_fraction"])))
    # Connectivity is spatial, independent of recognised strings or dictionaries.
    joined = ndimage.binary_dilation(
        ink, structure=np.ones((max(3, factor * 2 + 1), gap), dtype=bool)
    )
    groups, n = ndimage.label(joined)
    regions = ndimage.find_objects(groups)
    lines: list[OrientedLine] = []
    for number, region in enumerate(regions, 1):
        if region is None:
            continue
        ys, xs = np.nonzero(ink[region] & (groups[region] == number))
        if len(xs) < LINE_POLICY["component_minimum_pixels"]:
            continue
        x, y, r, b = (
            int(xs.min()) + region[1].start,
            int(ys.min()) + region[0].start,
            int(xs.max()) + region[1].start + 1,
            int(ys.max()) + region[0].start + 1,
        )
        if b - y < 2 or r - x < 2:
            continue
        polygon = transform.polygon_to_original(((x, y), (r, y), (r, b), (x, b)))
        baseline = transform.polygon_to_original(((x, (y + b) / 2), (r, (y + b) / 2)))
        original = bounds(polygon)
        left, top, right, bottom = source.bbox
        issues: tuple[str, ...] = ()
        if b - y > 2.2 * glyph_height:
            issues = ("overlapping_or_mixed_orientation_line_structure",)
        if (
            original[0] <= left + 0.5
            or original[1] <= top + 0.5
            or original[2] >= right - 0.5
            or original[3] >= bottom - 0.5
        ):
            issues += ("source_region_truncation_possible",)
        lines.append(OrientedLine(len(lines), (x, y, r, b), polygon, baseline, issues=issues))
    if len(lines) > LINE_POLICY["maximum_lines"]:
        return (
            prepared,
            transform,
            (),
            {
                **metadata,
                "status": "capped",
                "reason": "line_limit_exceeded",
                "observed_lines": len(lines),
            },
        )
    return (
        prepared,
        transform,
        tuple(lines),
        {
            **metadata,
            "status": "measured",
            "glyph_height": glyph_height,
            "join_width": gap,
            "observed_lines": len(lines),
        },
    )


def associate_lines(
    lines: tuple[OrientedLine, ...], config: GridConfig, source: OCRLabelSource
) -> tuple[tuple[OrientedLine, ...], dict[str, Any]]:
    """Own a column by the text endpoint nearest the selected matrix boundary."""
    _, top, _, bottom = config.heatmap_bbox
    if source.bbox[3] <= top:
        side, boundary = "above", top
    elif source.bbox[1] >= bottom:
        side, boundary = "below", bottom
    else:
        return tuple(
            replace(line, issues=line.issues + ("label_region_overlaps_matrix",)) for line in lines
        ), {"status": "review_required", "reason": "label_region_overlaps_matrix"}
    slots = axis_slots(config, "columns")
    maximum = config.labels.ocr.max_axis_offset_fraction if config.labels else 0.4
    associated = []
    for line in lines:
        a, b = line.baseline
        distance_a, distance_b = abs(a[1] - boundary), abs(b[1] - boundary)
        issues = list(line.issues)
        interpretation: str = source.attachment_endpoint
        if interpretation == "start":
            attachment = a
        elif interpretation == "end":
            attachment = b
        elif abs(math.sin(math.radians(source.text_rotation_degrees))) < 1e-9:
            attachment = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            interpretation = "centred_horizontal"
        elif abs(distance_a - distance_b) < 1:
            attachment = None
            issues.append("ambiguous_matrix_side_endpoint")
        else:
            attachment = a if distance_a < distance_b else b
            interpretation = "matrix_side_start" if distance_a < distance_b else "matrix_side_end"
        close = []
        if (
            attachment is not None
            and source.attachment_endpoint in ("start", "end")
            and abs(attachment[1] - boundary) > min(distance_a, distance_b) + 1
        ):
            issues.append("configured_endpoint_faces_away_from_matrix")
        if attachment is not None:
            if abs(attachment[1] - boundary) > source.max_attachment_distance_pixels:
                issues.append("attachment_too_far_from_matrix_or_group_header")
            else:
                close = [
                    s.index
                    for s in slots
                    if s.start <= attachment[0] < s.end
                    and abs(attachment[0] - s.centre) <= maximum * (s.end - s.start)
                ]
                if len(close) != 1:
                    issues.append("attachment_outside_unique_column")
        associated.append(
            replace(
                line,
                attachment=attachment,
                attachment_interpretation=interpretation,
                slot_index=close[0] if len(close) == 1 else None,
                issues=tuple(issues),
            )
        )
    counts = Counter(line.slot_index for line in associated if line.slot_index is not None)
    associated = [
        replace(line, issues=line.issues + ("overlapping_attachments",))
        if line.slot_index is not None and counts[line.slot_index] > 1
        else line
        for line in associated
    ]
    resolved = [line for line in associated if line.slot_index is not None and not line.issues]
    occupied = {line.slot_index for line in resolved}
    return tuple(associated), dict(
        status="accepted"
        if resolved
        and all(
            not line.issues
            or line.slot_index is None
            and any(
                issue in line.issues
                for issue in (
                    "attachment_outside_unique_column",
                    "attachment_too_far_from_matrix_or_group_header",
                )
            )
            for line in associated
        )
        else "review_required",
        side=side,
        matrix_boundary=boundary,
        method="matrix-side endpoint; original x; no spelling evidence",
        resolved_lines=len(resolved),
        missing_slots=[s.index for s in slots if s.index not in occupied],
        complete=len(occupied) == len(slots),
        lines=[asdict(line) for line in associated],
        grid_sha256=resolve_grid(config).metadata()["sha256"],
    )


def recognise_lines(
    image: RGBArray, config: GridConfig, source: OCRLabelSource, budget: LabelBudget | None = None
) -> tuple[AxisLabelResults, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    from .ocr import recognize

    assert config.labels is not None
    budget = budget or LabelBudget()
    prepared, transform, lines, segmentation = segment_lines(
        image, source, config.labels.ocr.upscale_factor, budget
    )
    lines, geometry = associate_lines(lines, config, source)
    geometry = {
        **geometry,
        "reading_direction_evidence": baseline_direction_evidence(prepared, lines)
        if prepared is not None
        else {"status": "unresolved"},
        "segmentation": segmentation,
        "text_angle": source.text_rotation_degrees,
    }
    candidates: list[LabelCandidate] = []
    tokens: list[OCRToken] = []
    records, skipped = [], []
    if segmentation.get("status") == "capped":
        skipped.append(dict(reason=segmentation.get("reason"), stage="segmentation", slot=None))
    for line in lines:
        if line.issues or line.slot_index is None or prepared is None:
            skipped.append(
                dict(
                    line=line.index,
                    slot=line.slot_index,
                    reason="geometry_requires_review",
                    issues=line.issues,
                )
            )
            continue
        x, y, r, b = line.normalised_bbox
        pad = LINE_POLICY["line_padding_pixels"] * config.labels.ocr.upscale_factor
        cropbox = (
            max(0, x - pad),
            max(0, y - pad),
            min(prepared.width, r + pad),
            min(prepared.height, b + pad),
        )
        crop = prepared.crop(cropbox)
        readings: dict[str, int] = {}
        for variant in ("original", *config.labels.ocr.retry_variants):
            pixels = math.prod(crop.size)
            if not budget.reserve(pixels, ocr=True):
                skipped.append(
                    dict(
                        line=line.index,
                        slot=line.slot_index,
                        variant=variant,
                        reason="shared_budget_exhausted",
                    )
                )
                continue
            trial = (
                ImageOps.autocontrast(crop)
                if variant == "contrast"
                else crop.point(lambda p: 255 if p >= 160 else 0)
                if variant == "threshold"
                else crop
            )
            started = time.monotonic()
            try:
                run = recognize(
                    trial,
                    config.labels.ocr,
                    mode="text_line",
                    timeout_seconds=min(30, budget.remaining_seconds),
                )
            except (ValueError, OSError) as error:
                skipped.append(
                    dict(line=line.index, slot=line.slot_index, variant=variant, reason=str(error))
                )
                continue
            offset = len(tokens)
            original_tokens = []
            for token in run.tokens:
                if not token.text.strip() or token.confidence < 0:
                    original_tokens.append(token)
                    continue
                tx, ty, tr, tb = token.bbox
                polygon = transform.polygon_to_original(
                    (
                        (tx + cropbox[0], ty + cropbox[1]),
                        (tr + cropbox[0], ty + cropbox[1]),
                        (tr + cropbox[0], tb + cropbox[1]),
                        (tx + cropbox[0], tb + cropbox[1]),
                    )
                )
                original_tokens.append(replace(token, bbox=bounds(polygon), polygon=polygon))
            tokens.extend(original_tokens)
            valid = [i for i, t in enumerate(run.tokens) if t.text.strip() and t.confidence >= 0]
            # Only one geometrically isolated line is read; preserve spaces between words.
            valid.sort(key=lambda i: run.tokens[i].bbox[0])
            text = " ".join(run.tokens[i].text.strip() for i in valid)
            records.append(
                dict(
                    line=asdict(line),
                    variant=variant,
                    run=asdict(run),
                    original_tokens=[asdict(t) for t in original_tokens],
                    normalised_line_crop=cropbox,
                    source_transform=transform.metadata(),
                    ocr_pixels=pixels,
                    ocr_seconds=time.monotonic() - started,
                    heuristic_text_acceptance=bool(
                        valid
                        and min(run.tokens[i].confidence for i in valid)
                        >= config.labels.ocr.min_confidence
                    ),
                    identifier_status="unreviewed",
                    grid_sha256=geometry.get("grid_sha256"),
                )
            )
            if not valid:
                continue
            confidence = min(run.tokens[i].confidence for i in valid)
            indices = tuple(offset + i for i in valid)
            if text in readings:
                index = readings[text]
                prev = candidates[index]
                candidates[index] = replace(
                    prev,
                    confidence=min(prev.confidence, confidence),
                    token_indices=prev.token_indices + indices,
                )
            else:
                readings[text] = len(candidates)
                candidates.append(
                    LabelCandidate(
                        text,
                        bounds(line.polygon),
                        confidence,
                        indices,
                        line.slot_index,
                        line.polygon,
                        line.baseline,
                        line.attachment,
                        line.issues,
                    )
                )
    slots = []
    for slot in axis_slots(config, "columns"):
        members = [i for i, c in enumerate(candidates) if c.slot_index == slot.index]
        related = [line for line in lines if line.slot_index == slot.index]
        issues = tuple(dict.fromkeys(issue for line in related for issue in line.issues))
        if len(members) > 1:
            issues += ("conflicting_raw_readings",)
        raw = "\n".join(candidates[i].text for i in members) or None
        slot_confidence = min((candidates[i].confidence for i in members), default=None)
        status: LabelStatus = "ambiguous" if issues else "unreviewed" if raw else "missing"
        slots.append(
            LabelResult(
                slot.index,
                raw,
                None,
                slot_confidence,
                bounds(related[0].polygon) if related else None,
                "ocr",
                status,
                tuple(members),
                issues=issues or (("identifier_review_required",) if raw else ()),
            )
        )
    duplicates = Counter(s.raw_text for s in slots if s.raw_text is not None)
    slots = [
        replace(s, duplicate=True, issues=s.issues + ("duplicate_text",))
        if s.raw_text and duplicates[s.raw_text] > 1
        else s
        for s in slots
    ]
    geometry["budget_remaining"] = dict(
        calls=budget.calls, pixels=budget.pixels, seconds=max(0, budget.remaining_seconds)
    )
    return (
        AxisLabelResults(
            "columns", tuple(slots), tuple(tokens), tuple(candidates), geometry_evidence=geometry
        ),
        tuple(records),
        tuple(skipped),
    )


def baseline_direction_evidence(
    prepared: Image.Image, lines: tuple[OrientedLine, ...]
) -> dict[str, Any]:
    """Glyph edge alignment, never recognised words or engine confidence.

    Upright single-line glyphs may share a baseline while their top edges vary.
    Uniform-height/all-capital glyphs, weak evidence and mixed baselines remain
    undecided. This bounded font/layout model is not a universal text-direction test.
    """
    records = []
    for line in lines:
        if line.slot_index is None or line.issues:
            continue
        x, y, r, b = line.normalised_bbox
        ink = np.asarray(prepared.crop((x, y, r, b))) < 175
        labels, _ = ndimage.label(ink)
        regions = ndimage.find_objects(labels)
        heights = [region[0].stop - region[0].start for region in regions if region is not None]
        typical = (
            float(np.median([h for h in heights if h >= 3])) if any(h >= 3 for h in heights) else 0
        )
        glyphs = [
            region
            for i, region in enumerate(regions, 1)
            if region is not None
            and region[0].stop - region[0].start >= 0.55 * typical
            and np.count_nonzero(labels[region] == i) >= 3
        ]
        if len(glyphs) < 6:
            continue
        tops = np.array([region[0].start for region in glyphs], float)
        bottoms = np.array([region[0].stop for region in glyphs], float)
        top = float(np.quantile(np.abs(tops - np.median(tops)), 0.8))
        bottom = float(np.quantile(np.abs(bottoms - np.median(bottoms)), 0.8))
        direction = (
            "upright"
            if bottom + 0.5 < 0.75 * (top + 0.5)
            else "inverted"
            if top + 0.5 < 0.75 * (bottom + 0.5)
            else "unresolved"
        )
        records.append(
            dict(
                line=line.index,
                glyphs=len(glyphs),
                top_dispersion=top,
                bottom_dispersion=bottom,
                direction=direction,
                typical_height=typical,
            )
        )
    upright = sum(r["direction"] == "upright" for r in records)
    inverted = sum(r["direction"] == "inverted" for r in records)
    status = (
        "upright"
        if len(records) >= 2 and upright >= 0.75 * len(records) and not inverted
        else (
            "inverted"
            if len(records) >= 2 and inverted >= 0.75 * len(records) and not upright
            else "unresolved"
        )
    )
    return dict(
        status=status,
        model="shared glyph baseline with varying top edges",
        records=records,
        score_kind="geometric edge dispersion; not spelling confidence",
    )
