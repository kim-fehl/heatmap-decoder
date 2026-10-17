"""Regular-grid geometry using exclusive bounds and pixel centres."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

from .models import (
    Axis,
    AxisSegment,
    AxisSlot,
    BoundingBox,
    ExtractionConfig,
    GridConfig,
    OCRLabelSource,
)


def validate_grid_geometry(config: GridConfig, image_shape: tuple[int, ...]) -> None:
    """Validate the shared grid and all label regions against original pixels."""
    height, width = image_shape[:2]
    boxes = [("heatmap", config.heatmap_bbox)]
    if config.labels is not None:
        for axis in ("rows", "columns"):
            source = getattr(config.labels, axis)
            if isinstance(source, OCRLabelSource):
                boxes.append((f"labels.{axis}", source.bbox))
    for name, (left, top, right, bottom) in boxes:
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"{name} bounding box must lie inside the image")
    left, top, right, bottom = config.heatmap_bbox
    if config.columns > right - left or config.rows > bottom - top:
        raise ValueError("each heatmap cell must span at least one pixel per axis")
    for axis in ("rows", "columns"):
        for slot in getattr(resolve_grid(config), axis).slots:
            if slot.end - slot.start < 1:
                raise ValueError(
                    f"insufficient resolution: {axis} cell {slot.index} spans <1 pixel"
                )
            interior_slice(slot.start, slot.end, 0)


@dataclass(frozen=True)
class ResolvedAxis:
    slots: tuple[AxisSlot, ...]
    segments: tuple[AxisSegment, ...]
    gaps: tuple[tuple[float, float], ...]
    # Global index -> (segment index, segment-local index).
    ownership: tuple[tuple[int, int], ...]
    source: str
    status: str = "resolved"
    uncertainty: str = "user assumption; not automatic detection evidence"


@dataclass(frozen=True)
class ResolvedGrid:
    bbox: BoundingBox
    rows: ResolvedAxis
    columns: ResolvedAxis
    version: str = "segmented-grid.1"

    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        for axis in ("rows", "columns"):
            payload[axis]["segments"] = [s.model_dump() for s in getattr(self, axis).segments]
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {**payload, "sha256": hashlib.sha256(canonical.encode()).hexdigest()}


@lru_cache(maxsize=512)
def _resolve_axis(
    start: float, end: float, count: int, configured: tuple[AxisSegment, ...] | None
) -> ResolvedAxis:
    segments = configured or (AxisSegment(start=start, end=end, count=count),)
    slots: list[AxisSlot] = []
    ownership: list[tuple[int, int]] = []
    for group, segment in enumerate(segments):
        pitch = (segment.end - segment.start) / segment.count
        for local in range(segment.count):
            slots.append(
                AxisSlot(
                    len(slots),
                    segment.start + (local + 0.5) * pitch,
                    segment.start + local * pitch,
                    segment.start + (local + 1) * pitch,
                )
            )
            ownership.append((group, local))
    gaps = tuple((a.end, b.start) for a, b in zip(segments, segments[1:]) if a.end < b.start)
    return ResolvedAxis(
        tuple(slots),
        segments,
        gaps,
        tuple(ownership),
        "explicit_segments" if configured is not None else "regular_counts",
    )


def resolve_grid(config: GridConfig) -> ResolvedGrid:
    """The sole resolved geometry for sampling, OCR, overlays and provenance."""
    left, top, right, bottom = config.heatmap_bbox
    return ResolvedGrid(
        config.heatmap_bbox,
        _resolve_axis(top, bottom, config.rows, config.rows_segments),
        _resolve_axis(left, right, config.columns, config.columns_segments),
    )


def axis_slots(config: GridConfig, axis: Axis) -> tuple[AxisSlot, ...]:
    """Cell intervals in original coordinates; gaps never consume matrix indices."""
    if axis not in ("rows", "columns"):
        raise ValueError("axis must be rows or columns")
    grid = resolve_grid(config)
    return grid.rows.slots if axis == "rows" else grid.columns.slots


def validate_geometry(config: ExtractionConfig, image_shape: tuple[int, ...]) -> None:
    """Reject out-of-image ROIs and grids finer than the raster."""
    validate_grid_geometry(config, image_shape)
    for axis in ("rows", "columns"):
        for slot in getattr(resolve_grid(config), axis).slots:
            interior_slice(slot.start, slot.end, config.sampling_inset_fraction)
    height, width = image_shape[:2]
    left, top, right, bottom = config.colourbar_bbox
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("colourbar bounding box must lie inside the image")
    length = bottom - top if config.colourbar_orientation == "vertical" else right - left
    if length < 2:
        raise ValueError("colourbar must contain at least two positions along its axis")


def interior_slice(start: float, end: float, inset: float) -> slice:
    """Select pixel centres within the inset interval; reject an empty interior."""
    margin = (end - start) * inset
    first = math.ceil(start + margin - 0.5)
    stop = math.ceil(end - margin - 0.5)
    if first >= stop:
        raise ValueError(
            "insufficient resolution: sampling interior contains no pixel centres; "
            "reduce inset or grid size"
        )
    return slice(first, stop)


def cell_slices(config: ExtractionConfig) -> list[tuple[slice, slice]]:
    """Return central interiors in row-major order, including fractional cell bounds."""
    return [
        (
            interior_slice(row.start, row.end, config.sampling_inset_fraction),
            interior_slice(col.start, col.end, config.sampling_inset_fraction),
        )
        for row in axis_slots(config, "rows")
        for col in axis_slots(config, "columns")
    ]


def box_slices(box: BoundingBox) -> tuple[slice, slice]:
    left, top, right, bottom = box
    return slice(top, bottom), slice(left, right)
