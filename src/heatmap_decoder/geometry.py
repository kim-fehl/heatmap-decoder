"""Regular-grid geometry using exclusive bounds and pixel centres."""

import math

from .models import Axis, AxisSlot, BoundingBox, ExtractionConfig, GridConfig, OCRLabelSource


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


def axis_slots(config: GridConfig, axis: Axis) -> tuple[AxisSlot, ...]:
    """Full equal cell bands, in displayed order; independent of ROI side."""
    left, top, right, bottom = config.heatmap_bbox
    if axis == "rows":
        start, end, count = top, bottom, config.rows
    elif axis == "columns":
        start, end, count = left, right, config.columns
    else:
        raise ValueError("axis must be rows or columns")
    pitch = (end - start) / count
    return tuple(
        AxisSlot(i, start + (i + 0.5) * pitch, start + i * pitch, start + (i + 1) * pitch)
        for i in range(count)
    )


def validate_geometry(config: ExtractionConfig, image_shape: tuple[int, ...]) -> None:
    """Reject out-of-image ROIs and grids finer than the raster."""
    validate_grid_geometry(config, image_shape)
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
        raise ValueError("sampling interior contains no pixel centres; reduce inset or grid size")
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
