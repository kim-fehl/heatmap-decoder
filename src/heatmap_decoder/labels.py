"""Deterministic horizontal-label geometry, with no OCR engine dependency.

Rows describe genes, columns describe samples, always in the displayed order.
Geometry cannot establish biological identity. All evidence survives assignment.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace

from PIL import Image

from .geometry import axis_slots, validate_grid_geometry
from .models import (
    Axis,
    AxisLabelResults,
    AxisSlot,
    BoundingBox,
    GridConfig,
    LabelCandidate,
    LabelResult,
    LabelStatus,
    OCRConfig,
    OCRLabelSource,
    OCRToken,
    TextBox,
)


def _validate_box(box: TextBox) -> None:
    left, top, right, bottom = box
    if not all(math.isfinite(value) for value in box) or right <= left or bottom <= top:
        raise ValueError("text box must have finite coordinates and positive width and height")


@dataclass(frozen=True)
class CropTransform:
    """Map resized crop pixels to original-image coordinates without rounding."""

    bbox: BoundingBox
    upscale_factor: int = 3

    def __post_init__(self) -> None:
        _validate_box(self.bbox)
        if self.bbox[0] < 0 or self.bbox[1] < 0:
            raise ValueError("crop origin must be nonnegative")
        if isinstance(self.upscale_factor, bool) or not isinstance(self.upscale_factor, int):
            raise ValueError("upscale_factor must be an integer >= 1")
        if self.upscale_factor < 1:
            raise ValueError("upscale_factor must be an integer >= 1")

    def to_original(self, box: TextBox) -> TextBox:
        _validate_box(box)
        left, top, right, bottom = self.bbox
        x0, y0, x1, y1 = box
        factor = self.upscale_factor
        if not (0 <= x0 < x1 <= (right - left) * factor):
            raise ValueError("token box must lie inside the resized crop")
        if not (0 <= y0 < y1 <= (bottom - top) * factor):
            raise ValueError("token box must lie inside the resized crop")
        return left + x0 / factor, top + y0 / factor, left + x1 / factor, top + y1 / factor

    def to_crop(self, box: TextBox) -> TextBox:
        _validate_box(box)
        left, top, right, bottom = self.bbox
        x0, y0, x1, y1 = box
        if not (left <= x0 < x1 <= right and top <= y0 < y1 <= bottom):
            raise ValueError("token box must lie inside the original crop")
        factor = self.upscale_factor
        return (
            (x0 - left) * factor,
            (y0 - top) * factor,
            (x1 - left) * factor,
            (y1 - top) * factor,
        )


def crop_label_region(
    image: Image.Image, config: GridConfig, axis: Axis
) -> tuple[Image.Image, CropTransform]:
    """Validate before cropping; deterministic grayscale, then bicubic upscale."""
    validate_grid_geometry(config, (image.height, image.width))
    source = getattr(config.labels, axis) if config.labels is not None else None
    if not isinstance(source, OCRLabelSource):
        raise ValueError(f"labels.{axis} must specify an OCR region")
    assert config.labels is not None
    transform = CropTransform(source.bbox, config.labels.ocr.upscale_factor)
    crop = image.crop(source.bbox).convert("L")
    factor = transform.upscale_factor
    return crop.resize(
        (crop.width * factor, crop.height * factor), Image.Resampling.BICUBIC
    ), transform


def transform_tokens(tokens: Sequence[OCRToken], transform: CropTransform) -> tuple[OCRToken, ...]:
    """Keep raw text/confidence/line IDs; ignore geometry of structural records."""
    return tuple(
        replace(token, bbox=transform.to_original(token.bbox))
        if token.text.strip() and token.confidence >= 0
        else token
        for token in tokens
    )


def _union_box(boxes: Sequence[TextBox]) -> TextBox:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _column_band(box: TextBox, slots: Sequence[AxisSlot]) -> int | None:
    return next((s.index for s in slots if s.start <= box[0] and box[2] <= s.end), None)


def _same_line(a: OCRToken, b: OCRToken) -> bool:
    """At least half-height vertical overlap and at most two heights of word gap.

    Line IDs never override geometry. A complete-link grouping avoids chaining
    several rows or widely separated labels into one candidate.
    """
    x0, y0, x1, y1 = a.bbox
    u0, v0, u1, v1 = b.bbox
    overlap = min(y1, v1) - max(y0, v0)
    gap = max(0.0, u0 - x1, x0 - u1)
    return overlap >= 0.5 * min(y1 - y0, v1 - v0) and gap <= 2 * max(y1 - y0, v1 - v0)


def group_tokens(
    tokens: Sequence[OCRToken], config: GridConfig, axis: Axis
) -> tuple[LabelCandidate, ...]:
    """Group upright words geometrically, preserving token membership and spelling.

    Column words are grouped only within the same full column band. A token
    crossing a band is kept intact for ambiguity review, never split to fit.
    """
    slots = axis_slots(config, axis)
    valid = []
    for i, token in enumerate(tokens):
        if not token.text.strip() or token.confidence < 0:
            continue
        _validate_box(token.bbox)
        if not math.isfinite(token.confidence) or token.confidence > 100:
            raise ValueError("text token confidence must be finite and between 0 and 100")
        valid.append(i)
    valid.sort(key=lambda i: (tokens[i].bbox[1], tokens[i].bbox[0], i))
    groups: list[list[int]] = []
    for i in valid:
        token = tokens[i]
        matches = []
        for group in groups:
            band = _column_band(token.bbox, slots)
            if axis == "columns" and (
                band is None or any(_column_band(tokens[j].bbox, slots) != band for j in group)
            ):
                continue
            if all(_same_line(token, tokens[j]) for j in group):
                matches.append(group)
        # A word compatible with multiple lines stays separate for review.
        if len(matches) == 1:
            matches[0].append(i)
        else:
            groups.append([i])
    candidates = []
    for group in groups:
        group.sort(key=lambda i: (tokens[i].bbox[0], tokens[i].bbox[1], i))
        candidates.append(
            LabelCandidate(
                text=" ".join(tokens[i].text.strip() for i in group),
                bbox=_union_box([tokens[i].bbox for i in group]),
                confidence=min(tokens[i].confidence for i in group),
                token_indices=tuple(group),
            )
        )
    return tuple(candidates)


def assign_candidates(
    candidates: Sequence[LabelCandidate],
    config: GridConfig,
    axis: Axis,
    *,
    tokens: Sequence[OCRToken] = (),
) -> AxisLabelResults:
    """Return exactly one result per slot; retain collisions, spans and unmatched evidence."""
    slots = axis_slots(config, axis)
    settings = config.labels.ocr if config.labels is not None else OCRConfig()
    coordinate = 1 if axis == "rows" else 0
    memberships: list[list[int]] = [[] for _ in slots]
    problems: dict[int, str] = {}
    unmatched: list[int] = []
    for i, candidate in enumerate(candidates):
        _validate_box(candidate.bbox)
        if not candidate.text.strip():
            raise ValueError("candidate text must be non-empty")
        if not math.isfinite(candidate.confidence) or not 0 <= candidate.confidence <= 100:
            raise ValueError("candidate confidence must be finite and between 0 and 100")
        start, end = candidate.bbox[coordinate], candidate.bbox[coordinate + 2]
        centre = (start + end) / 2
        close = [
            s.index
            for s in slots
            if abs(centre - s.centre) <= settings.max_axis_offset_fraction * (s.end - s.start)
        ]
        if len(close) == 1:
            slot = slots[close[0]]
            if slot.start <= start and end <= slot.end:
                memberships[slot.index].append(i)
                continue
        # Mark all intersected bands ambiguous, even if the centre is too far away.
        affected = [s.index for s in slots if start < s.end and end > s.start]
        problems[i] = "spans_slots" if len(affected) > 1 else "axis_offset"
        unmatched.append(i)
        for index in affected:
            memberships[index].append(i)
    results = []
    for slot, members in zip(slots, memberships, strict=True):
        if not members:
            results.append(LabelResult(slot.index, None, None, None, None, "ocr", "missing"))
            continue
        evidence = [candidates[i] for i in members]
        issues = tuple(dict.fromkeys(problems[i] for i in members if i in problems))
        if len(members) > 1:
            issues += ("multiple_candidates",)
        ambiguous = bool(issues)
        confidence = min(c.confidence for c in evidence)
        raw = "\n".join(c.text.strip() for c in evidence)
        status: LabelStatus = (
            "ambiguous"
            if ambiguous
            else ("accepted" if confidence >= settings.min_confidence else "low_confidence")
        )
        results.append(
            LabelResult(
                slot.index,
                raw,
                raw if status == "accepted" else None,
                confidence,
                _union_box([c.bbox for c in evidence]),
                "ocr",
                status,
                tuple(members),
                issues=issues,
            )
        )
    counts = Counter(r.final_text for r in results if r.status == "accepted")
    results = [
        replace(r, duplicate=True, issues=r.issues + ("duplicate_text",))
        if r.final_text is not None and counts[r.final_text] > 1
        else r
        for r in results
    ]
    return AxisLabelResults(
        axis, tuple(results), tuple(tokens), tuple(candidates), tuple(unmatched)
    )


def labels_from_tokens(
    tokens: Sequence[OCRToken], config: GridConfig, axis: Axis
) -> AxisLabelResults:
    """Engine-independent M4 entry point; input boxes must be original coordinates."""
    return assign_candidates(group_tokens(tokens, config, axis), config, axis, tokens=tokens)


def generated_labels(config: GridConfig, axis: Axis) -> AxisLabelResults:
    """Positional identifiers for an omitted axis, without compressing its slots."""
    prefix = "row" if axis == "rows" else "column"
    return AxisLabelResults(
        axis,
        tuple(
            LabelResult(s.index, None, f"{prefix}_{s.index}", None, None, "generated", "generated")
            for s in axis_slots(config, axis)
        ),
    )
