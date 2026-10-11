"""Regular-grid geometry using exclusive bounds and pixel centres."""

import math

from .models import BoundingBox, ExtractionConfig


def validate_geometry(config: ExtractionConfig, image_shape: tuple[int, ...]) -> None:
    """Reject out-of-image ROIs and grids finer than the raster."""
    height, width = image_shape[:2]
    for name, box in (("heatmap", config.heatmap_bbox), ("colourbar", config.colourbar_bbox)):
        left, top, right, bottom = box
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError(f"{name} bounding box must lie inside the image")
    left, top, right, bottom = config.heatmap_bbox
    if config.columns > right - left or config.rows > bottom - top:
        raise ValueError("each heatmap cell must span at least one pixel per axis")
    left, top, right, bottom = config.colourbar_bbox
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
    left, top, right, bottom = config.heatmap_bbox
    width = (right - left) / config.columns
    height = (bottom - top) / config.rows
    return [
        (
            interior_slice(
                top + row * height, top + (row + 1) * height, config.sampling_inset_fraction
            ),
            interior_slice(
                left + col * width, left + (col + 1) * width, config.sampling_inset_fraction
            ),
        )
        for row in range(config.rows)
        for col in range(config.columns)
    ]


def box_slices(box: BoundingBox) -> tuple[slice, slice]:
    left, top, right, bottom = box
    return slice(top, bottom), slice(left, right)
