"""Deterministic, review-only heatmap proposals; no OCR, files or inferred counts."""

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import closing

from .geometry import validate_grid_geometry
from .models import (
    BoundingBox,
    DetectionHints,
    DetectionResult,
    GridConfig,
    OCRLabelSource,
    RGBArray,
    ROICandidate,
    SelectedRegion,
)
from .sampling import validate_rgb

DETECTOR_VERSION = "m7.1"


@dataclass(frozen=True)
class ScaleTransform:
    """Exact independent axis ratios, including rounded resize dimensions."""

    original_width: int
    original_height: int
    analysis_width: int
    analysis_height: int

    def original_box(self, box: BoundingBox) -> BoundingBox:
        x = self.original_width / self.analysis_width
        y = self.original_height / self.analysis_height
        return (
            math.floor(box[0] * x),
            math.floor(box[1] * y),
            min(self.original_width, math.ceil(box[2] * x)),
            min(self.original_height, math.ceil(box[3] * y)),
        )


def analysis_image(image: RGBArray, maximum: int) -> tuple[RGBArray, ScaleTransform]:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = np.asarray(Image.fromarray(image[:, :, :3]).resize(size, Image.Resampling.BOX))
    return resized, ScaleTransform(width, height, *size)


def foreground(image: RGBArray, threshold: float) -> np.ndarray:
    """Background distance uses RGB variation too, so neutral palettes remain eligible."""
    rgb = image[:, :, :3].astype(float)
    perimeter = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.median(perimeter, axis=0)
    return np.max(np.abs(rgb - background), axis=2) > threshold


def refine_box(
    image: RGBArray,
    box: BoundingBox,
    margin: int,
    threshold: float,
    full_mask: NDArray[np.bool_] | None = None,
) -> BoundingBox:
    """Find occupied original pixels near a coarse component; trim uniform outer strokes.

    No merging at original resolution. A uniform perimeter is border evidence only
    when all pixels on the entire edge agree; uncertain borders stay reviewable.
    """
    height, width = image.shape[:2]
    left, top, right, bottom = box
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(width, right + margin), min(height, bottom + margin)
    mask = (foreground(image, threshold) if full_mask is None else full_mask)[
        top:bottom, left:right
    ]
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return box
    right, bottom = left + int(xs.max()) + 1, top + int(ys.max()) + 1
    left, top = left + int(xs.min()), top + int(ys.min())
    # Bounded trim: borders are thin; never erode arbitrarily into a uniform panel.
    for _ in range(4):
        crop = image[top:bottom, left:right, :3]
        if min(crop.shape[:2]) <= 2:
            break
        trims = [
            np.max(np.ptp(edge.astype(float), axis=0)) < 2
            for edge in (crop[:, 0], crop[0], crop[:, -1], crop[-1])
        ]
        if not any(trims):
            break
        left += int(trims[0])
        top += int(trims[1])
        right -= int(trims[2])
        bottom -= int(trims[3])
    return left, top, right, bottom


def score_box(
    image: RGBArray,
    box: BoundingBox,
    hints: DetectionHints,
    full_mask: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """Compare fixed-grid interiors with between-cell variation; scores are heuristics."""
    left, top, right, bottom = box
    pitch_x, pitch_y = (right - left) / hints.columns, (bottom - top) / hints.rows
    if min(pitch_x, pitch_y) < 3:
        return {
            "occupancy": 0.0,
            "piecewise_constant": 0.0,
            "variation": 0.0,
            "boundary": 0.0,
            "equal_grid": 0.0,
        }
    # At most 32 slots per axis and 3x3 samples per cell, including both ends.
    rows = np.unique(np.linspace(0, hints.rows - 1, min(32, hints.rows)).astype(int))
    columns = np.unique(np.linspace(0, hints.columns - 1, min(32, hints.columns)).astype(int))
    colours, errors = [], []
    for row in rows:
        for column in columns:
            ys = np.minimum(
                bottom - 1, (top + (row + np.array([0.3, 0.5, 0.7])) * pitch_y).astype(int)
            )
            xs = np.minimum(
                right - 1, (left + (column + np.array([0.3, 0.5, 0.7])) * pitch_x).astype(int)
            )
            pixels = image[ys[:, None], xs[None, :], :3].astype(float).reshape(-1, 3)
            colour = np.median(pixels, axis=0)
            colours.append(colour)
            errors.append(float(np.mean(np.abs(pixels - colour))))
    variation = float(np.mean(np.std(colours, axis=0)))
    constant = max(0.0, 1 - float(np.mean(errors)) / max(variation, 1))
    grid_colours = np.asarray(colours).reshape(len(rows), len(columns), 3)
    jumps = [
        float(np.mean(np.abs(np.diff(grid_colours, axis=axis))))
        for axis in (0, 1)
        if grid_colours.shape[axis] > 1
    ]
    grid = max(0.0, 1 - 4 * float(np.mean(errors)) / max(max(jumps, default=0), 1))
    occupancy = float(
        np.mean(
            (
                foreground(image, hints.detector.background_distance)
                if full_mask is None
                else full_mask
            )[top:bottom, left:right]
        )
    )
    boundary = []
    for inside, outside in (
        (image[top:bottom, left, :3], image[top:bottom, max(0, left - 1), :3]),
        (image[top:bottom, right - 1, :3], image[top:bottom, min(image.shape[1] - 1, right), :3]),
        (image[top, left:right, :3], image[max(0, top - 1), left:right, :3]),
        (image[bottom - 1, left:right, :3], image[min(image.shape[0] - 1, bottom), left:right, :3]),
    ):
        boundary.append(min(1.0, float(np.mean(np.abs(inside.astype(float) - outside))) / 30))
    return {
        "occupancy": occupancy,
        "piecewise_constant": constant,
        "variation": min(1.0, variation / 25),
        "boundary": float(np.mean(boundary)),
        "equal_grid": grid,
    }


def detect(image: RGBArray, hints: DetectionHints) -> DetectionResult:
    """Manual boxes bypass localisation. Every detected box requires explicit review."""
    validate_rgb(image)
    height, width = image.shape[:2]
    selected: dict[Literal["heatmap", "colourbar"], SelectedRegion] = {}
    kinds: tuple[Literal["heatmap", "colourbar"], ...] = ("heatmap", "colourbar")
    for name in kinds:
        box = getattr(hints, f"{name}_bbox")
        if box is not None:
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError(f"{name} bounding box must lie inside the image")
            selected[name] = SelectedRegion(bbox=box, origin="manual")
    if hints.labels is not None:
        for axis in ("rows", "columns"):
            source = getattr(hints.labels, axis)
            if isinstance(source, OCRLabelSource):
                box = source.bbox
                if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                    raise ValueError(f"labels.{axis} bounding box must lie inside the image")
    if hints.heatmap_bbox is not None:
        validate_grid_geometry(
            GridConfig(
                heatmap_bbox=hints.heatmap_bbox,
                rows=hints.rows,
                columns=hints.columns,
                labels=hints.labels,
            ),
            image.shape,
        )
        evidence = score_box(image, hints.heatmap_bbox, hints)
        warnings: tuple[str, ...] = ()
        if evidence["variation"] < 0.2 or evidence["equal_grid"] < 0.8:
            warnings = (
                "Manual bounds preserved; supplied grid has weak colour/structure evidence.",
            )
        return DetectionResult(
            candidates=(),
            selected=selected,
            status="manual",
            unresolved_requirements=(),
            warnings=warnings,
        )
    if image.shape[2] == 4 and np.any(image[:, :, 3] != 255):
        raise ValueError("ROI localisation requires an opaque image; supply manual boxes")
    full_mask = foreground(image, hints.detector.background_distance)
    boxes: set[BoundingBox] = set()
    for maximum in (hints.detector.max_analysis_size, hints.detector.max_analysis_size // 2):
        small, transform = analysis_image(image, maximum)
        mask = foreground(small, hints.detector.background_distance)
        closed = closing(mask, footprint=np.ones((hints.detector.gridline_gap,) * 2))
        regions = sorted(
            regionprops(label(closed)),  # type: ignore[no-untyped-call]
            key=lambda r: (-r.area, r.bbox),
        )
        for region in regions[: hints.detector.max_components]:
            top, left, bottom, right = region.bbox
            if (right - left) < 6 or (bottom - top) < 6:
                continue
            box = transform.original_box((left, top, right, bottom))
            margin = math.ceil(max(width / small.shape[1], height / small.shape[0]))
            boxes.add(refine_box(image, box, margin, hints.detector.background_distance, full_mask))
    scored = []
    for box in sorted(boxes):
        components = score_box(image, box, hints, full_mask)
        score = (
            0.15 * components["occupancy"]
            + 0.45 * components["piecewise_constant"]
            + 0.25 * components["variation"]
            + 0.15 * components["boundary"]
        )
        if (
            score >= hints.detector.min_score
            and components["variation"] >= 0.2
            and components["equal_grid"] >= 0.8
        ):
            scored.append((score, box, components))
    scored.sort(key=lambda item: (-item[0], item[1]))
    candidates = tuple(
        ROICandidate(
            id=f"heatmap-{index + 1:03d}",
            bbox=box,
            score=score,
            score_components=components,
            reasons=("review_required", "multiple_plausible_regions")
            if len(scored) > 1
            else ("review_required",),
        )
        for index, (score, box, components) in enumerate(scored[: hints.detector.max_candidates])
    )
    status: Literal["unresolved", "review_required", "ambiguous"] = (
        "unresolved" if not candidates else "review_required"
    )
    if len(scored) > 1:
        status = "ambiguous"
    return DetectionResult(
        candidates=candidates,
        selected=selected,
        status=status,
        unresolved_requirements=("heatmap: select a candidate or supply bounds",),
    )
