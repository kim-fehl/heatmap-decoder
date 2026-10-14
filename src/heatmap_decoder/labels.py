"""Deterministic explicit-orientation label geometry, with no OCR engine dependency.

Rows describe genes, columns describe samples, always in the displayed order.
Geometry cannot establish biological identity. All evidence survives assignment.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from PIL import Image, ImageOps

from .geometry import axis_slots, validate_grid_geometry
from .models import (
    Axis,
    AxisLabelResults,
    AxisSlot,
    BoundingBox,
    GridConfig,
    LabelCandidate,
    LabelExtractionResult,
    LabelResult,
    LabelStatus,
    OCRConfig,
    OCRLabelSource,
    OCRToken,
    TextBox,
    normalise_text_angle,
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
    text_rotation_degrees: float = 0
    white_padding_pixels: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "text_rotation_degrees", normalise_text_angle(self.text_rotation_degrees)
        )
        if (
            isinstance(self.white_padding_pixels, bool)
            or not isinstance(self.white_padding_pixels, int)
            or not 0 <= self.white_padding_pixels <= 2
        ):
            raise ValueError("white_padding_pixels must be an integer 0..2")
        _validate_box(self.bbox)
        if self.bbox[0] < 0 or self.bbox[1] < 0:
            raise ValueError("crop origin must be nonnegative")
        if isinstance(self.upscale_factor, bool) or not isinstance(self.upscale_factor, int):
            raise ValueError("upscale_factor must be an integer >= 1")
        if self.upscale_factor < 1:
            raise ValueError("upscale_factor must be an integer >= 1")

    @property
    def crop_size(self) -> tuple[int, int]:
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        if self.text_rotation_degrees in (90, -90):
            width, height = height, width
        elif self.text_rotation_degrees != 0:
            c, s, _, _ = self.affine
            width, height = (
                math.ceil(abs(c) * width + abs(s) * height - 1e-9),
                math.ceil(abs(s) * width + abs(c) * height - 1e-9),
            )
        return (
            (width + 2 * self.white_padding_pixels) * self.upscale_factor,
            (height + 2 * self.white_padding_pixels) * self.upscale_factor,
        )

    def prepare(self, image: Image.Image) -> Image.Image:
        """Inverse visual rotation, exactly, before the existing upscale."""
        crop = image.crop(self.bbox).convert("L")
        if self.text_rotation_degrees == 90:
            crop = crop.transpose(Image.Transpose.ROTATE_270)
        elif self.text_rotation_degrees == -90:
            crop = crop.transpose(Image.Transpose.ROTATE_90)
        elif self.text_rotation_degrees != 0:
            c, s, tx, ty = self.affine
            width, height = self.crop_size
            pad = self.white_padding_pixels
            size = (width // self.upscale_factor - 2 * pad, height // self.upscale_factor - 2 * pad)
            crop = crop.transform(
                size,
                Image.Transform.AFFINE,
                (c, s, -c * tx - s * ty, -s, c, s * tx - c * ty),
                resample=Image.Resampling.BICUBIC,
                fillcolor=255,
            )
        if self.white_padding_pixels:
            crop = ImageOps.expand(crop, border=self.white_padding_pixels, fill=255)
        return crop.resize(self.crop_size, Image.Resampling.BICUBIC)

    def _point(self, x: float, y: float, *, inverse: bool) -> tuple[float, float]:
        left, top, right, bottom = self.bbox
        width, height = right - left, bottom - top
        factor = self.upscale_factor
        angle = self.text_rotation_degrees
        if inverse:
            x, y = x / factor - self.white_padding_pixels, y / factor - self.white_padding_pixels
            if angle == 90:
                x, y = y, height - x
            elif angle == -90:
                x, y = width - y, x
            elif angle != 0:
                c, s, tx, ty = self.affine
                x, y = c * (x - tx) + s * (y - ty), -s * (x - tx) + c * (y - ty)
            return left + x, top + y
        x, y = x - left, y - top
        if angle == 90:
            x, y = height - y, x
        elif angle == -90:
            x, y = y, width - x
        elif angle != 0:
            c, s, tx, ty = self.affine
            x, y = c * x - s * y + tx, s * x + c * y + ty
        return ((x + self.white_padding_pixels) * factor, (y + self.white_padding_pixels) * factor)

    @property
    def affine(self) -> tuple[float, float, float, float]:
        """Forward local continuous coordinates to expanded deskewed canvas."""
        angle = math.radians(self.text_rotation_degrees)
        c, s = math.cos(angle), math.sin(angle)
        width, height = self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]
        corners = [(c * x - s * y, s * x + c * y) for x in (0, width) for y in (0, height)]
        return c, s, -min(x for x, y in corners), -min(y for x, y in corners)

    def point_to_original(self, point: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(v) for v in point):
            raise ValueError("point must be finite")
        return self._point(*point, inverse=True)

    def point_to_crop(self, point: tuple[float, float]) -> tuple[float, float]:
        if not all(math.isfinite(v) for v in point):
            raise ValueError("point must be finite")
        return self._point(*point, inverse=False)

    def polygon_to_original(
        self, polygon: Sequence[tuple[float, float]]
    ) -> tuple[tuple[float, float], ...]:
        return tuple(self.point_to_original(p) for p in polygon)

    def polygon_to_crop(
        self, polygon: Sequence[tuple[float, float]]
    ) -> tuple[tuple[float, float], ...]:
        return tuple(self.point_to_crop(p) for p in polygon)

    def metadata(self) -> dict[str, object]:
        origin = self.point_to_crop((0, 0))
        px, py = self.point_to_crop((1, 0)), self.point_to_crop((0, 1))
        inverse = self.point_to_original((0, 0))
        ix, iy = self.point_to_original((1, 0)), self.point_to_original((0, 1))
        return {
            "bbox": self.bbox,
            "angle": self.text_rotation_degrees,
            "upscale_factor": self.upscale_factor,
            "white_padding_pixels": self.white_padding_pixels,
            "crop_size": self.crop_size,
            "forward_affine": [
                px[0] - origin[0],
                py[0] - origin[0],
                origin[0],
                px[1] - origin[1],
                py[1] - origin[1],
                origin[1],
            ],
            "inverse_affine": [
                ix[0] - inverse[0],
                iy[0] - inverse[0],
                inverse[0],
                ix[1] - inverse[1],
                iy[1] - inverse[1],
                inverse[1],
            ],
            "interpolation": "exact transpose"
            if self.text_rotation_degrees in (0, 90, -90)
            else "Pillow affine bicubic; expanded white canvas",
        }

    def _box(self, box: TextBox, *, inverse: bool) -> TextBox:
        _validate_box(box)
        x0, y0, x1, y1 = box
        pad = self.white_padding_pixels
        bounds = (
            (0, 0, *self.crop_size)
            if inverse
            else (self.bbox[0] - pad, self.bbox[1] - pad, self.bbox[2] + pad, self.bbox[3] + pad)
        )
        if not (bounds[0] <= x0 < x1 <= bounds[2] and bounds[1] <= y0 < y1 <= bounds[3]):
            raise ValueError(
                "token box must lie inside the resized crop"
                if inverse
                else "token box must lie inside the original crop"
            )
        corners = [self._point(x, y, inverse=inverse) for x in (x0, x1) for y in (y0, y1)]
        return (
            min(p[0] for p in corners),
            min(p[1] for p in corners),
            max(p[0] for p in corners),
            max(p[1] for p in corners),
        )

    def to_original(self, box: TextBox) -> TextBox:
        return self._box(box, inverse=True)

    def to_crop(self, box: TextBox) -> TextBox:
        return self._box(box, inverse=False)


def crop_label_region(
    image: Image.Image, config: GridConfig, axis: Axis
) -> tuple[Image.Image, CropTransform]:
    """Validate before cropping; deterministic grayscale, then bicubic upscale."""
    validate_grid_geometry(config, (image.height, image.width))
    source = getattr(config.labels, axis) if config.labels is not None else None
    if not isinstance(source, OCRLabelSource):
        raise ValueError(f"labels.{axis} must specify an OCR region")
    assert config.labels is not None
    transform = CropTransform(
        source.bbox, config.labels.ocr.upscale_factor, source.text_rotation_degrees
    )
    if math.prod(transform.crop_size) > 16000000:
        raise ValueError("label OCR pixel cap exceeded")
    return transform.prepare(image), transform


def transform_tokens(tokens: Sequence[OCRToken], transform: CropTransform) -> tuple[OCRToken, ...]:
    """Keep raw text/confidence/line IDs; ignore geometry of structural records."""
    return tuple(
        replace(
            token,
            bbox=transform.to_original(token.bbox),
            polygon=transform.polygon_to_original(
                (
                    (token.bbox[0], token.bbox[1]),
                    (token.bbox[2], token.bbox[1]),
                    (token.bbox[2], token.bbox[3]),
                    (token.bbox[0], token.bbox[3]),
                )
            )
            if transform.text_rotation_degrees not in (0, 90, -90)
            else token.polygon,
        )
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
    tokens: Sequence[OCRToken],
    config: GridConfig,
    axis: Axis,
    *,
    transform: CropTransform | None = None,
    slot_index: int | None = None,
) -> tuple[LabelCandidate, ...]:
    """Group upright words geometrically, preserving token membership and spelling.

    With a transform, grouping uses the upright OCR frame and bands use original x.
    Column words are grouped only within the same full column band. A token
    crossing a band is kept intact for ambiguity review, never split to fit.
    """
    slots = axis_slots(config, axis)
    original = transform_tokens(tokens, transform) if transform else tokens
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
            band = _column_band(original[i].bbox, slots)
            if (
                axis == "columns"
                and slot_index is None
                and (
                    band is None
                    or any(_column_band(original[j].bbox, slots) != band for j in group)
                )
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
                bbox=_union_box([original[i].bbox for i in group]),
                confidence=min(tokens[i].confidence for i in group),
                token_indices=tuple(group),
                slot_index=slot_index,
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
        if candidate.slot_index is not None:
            index = candidate.slot_index
            if isinstance(index, bool) or not 0 <= index < len(slots):
                raise ValueError("owned OCR slot index is out of range")
            slot = slots[index]
            source = getattr(config.labels, axis) if config.labels else None
            padding = source.crop_padding_pixels if isinstance(source, OCRLabelSource) else 0
            tolerance = 1 + padding  # outward raster rounding plus recorded source padding
            memberships[index].append(i)
            if (
                abs(centre - slot.centre)
                > settings.max_axis_offset_fraction * (slot.end - slot.start)
                or start < slot.start - tolerance
                or end > slot.end + tolerance
            ):
                problems[i] = "outside_owned_slot_field"
                unmatched.append(i)
            continue
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


def requested_review_slots(config: GridConfig) -> set[tuple[Axis, int]]:
    """Every configured axis must be reviewed completely; omitted axes are excluded."""
    axes: tuple[Axis, ...] = ("rows", "columns")
    return {
        (axis, slot.index)
        for axis in axes
        if config.labels is not None and getattr(config.labels, axis) is not None
        for slot in axis_slots(config, axis)
    }


def validate_label_slots(result: LabelExtractionResult) -> None:
    """Reject incomplete or reordered precomputed results before matrix publication."""
    if tuple(axis.axis for axis in result.axes) != ("rows", "columns"):
        raise ValueError("label results must contain rows then columns")
    for axis in result.axes:
        expected = axis_slots(result.config, axis.axis)
        if tuple(slot.index for slot in axis.slots) != tuple(slot.index for slot in expected):
            raise ValueError(f"labels.{axis.axis} must preserve every slot index in order")


def apply_review(
    result: LabelExtractionResult,
    reviewed: Mapping[tuple[Axis, int], str],
) -> LabelExtractionResult:
    """Resolve by index only; preserve raw text, boxes, confidence and OCR runs."""
    expected = requested_review_slots(result.config)
    validate_label_slots(result)
    if not expected:
        raise ValueError("labels review requires at least one configured label axis")
    if set(reviewed) != expected:
        raise ValueError("labels review must contain exactly every requested axis/index slot")
    texts = {key: value.strip() for key, value in reviewed.items()}
    if any(not text for text in texts.values()):
        raise ValueError("labels review text must be non-empty for every requested slot")
    axes = []
    for axis in result.axes:
        slots = tuple(
            replace(
                slot, final_text=texts[(axis.axis, slot.index)], source="manual", status="manual"
            )
            if (axis.axis, slot.index) in texts
            else slot
            for slot in axis.slots
        )
        counts = Counter(slot.final_text for slot in slots if slot.final_text is not None)
        slots = tuple(
            replace(
                slot,
                duplicate=slot.final_text is not None and counts[slot.final_text] > 1,
                issues=tuple(issue for issue in slot.issues if issue != "duplicate_text")
                + (
                    ("duplicate_text",)
                    if slot.final_text is not None and counts[slot.final_text] > 1
                    else ()
                ),
            )
            for slot in slots
        )
        axes.append(replace(axis, slots=slots))
    return replace(result, axes=tuple(axes), unreviewed_axes=result.unreviewed_axes or result.axes)


def group_unassigned_tokens(tokens: Sequence[OCRToken]) -> tuple[LabelCandidate, ...]:
    """Upright geometric grouping without slots or counts, for localisation evidence.

    Complete-link grouping cannot chain across long text rows. This evidence can
    support a count hypothesis, but no grouping establishes biological identity.
    """
    groups: list[list[int]] = []
    valid = [i for i, token in enumerate(tokens) if token.text.strip() and token.confidence >= 0]
    for index in sorted(valid, key=lambda i: (tokens[i].bbox[1], tokens[i].bbox[0], i)):
        token = tokens[index]
        _validate_box(token.bbox)
        matched = [group for group in groups if all(_same_line(token, tokens[j]) for j in group)]
        if len(matched) == 1:
            matched[0].append(index)
        else:
            groups.append([index])
    return tuple(
        LabelCandidate(
            " ".join(
                tokens[i].text.strip() for i in sorted(group, key=lambda i: tokens[i].bbox[0])
            ),
            _union_box([tokens[i].bbox for i in group]),
            min(tokens[i].confidence for i in group),
            tuple(group),
        )
        for group in groups
    )
