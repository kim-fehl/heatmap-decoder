"""Deterministic region and conditional-count evidence; no export or file writes."""

import math
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import closing

from .detection_policy import ACCEPTANCE_THRESHOLDS as A
from .detection_policy import THRESHOLDS as T
from .geometry import validate_grid_geometry
from .models import (
    Axis,
    BoolArray,
    BoundingBox,
    CountProposal,
    CountSignal,
    DetectionHints,
    DetectionResult,
    GridConfig,
    LabelCandidate,
    LabelLocalisationEvidence,
    OCRLabelSource,
    RGBArray,
    ROIAssociation,
    ROICandidate,
    ROIKind,
    SelectedRegion,
)
from .sampling import validate_rgb

DETECTOR_VERSION = "grid-strokes.1"


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


def foreground(image: RGBArray, threshold: float) -> BoolArray:
    """Background distance uses RGB variation too, so neutral palettes remain eligible."""
    rgb = image[:, :, :3].astype(float)
    perimeter = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.median(perimeter, axis=0)
    return np.asarray(np.max(np.abs(rgb - background), axis=2) > threshold, dtype=np.bool_)


def _terminal_grid_stroke(band: RGBArray) -> bool:
    """A terminal thin dark line must continue at least three equal spacings."""
    dark = np.mean(np.max(band, axis=2) < 48, axis=1) >= 0.95
    indices = np.flatnonzero(dark)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    if len(groups) < 4 or not len(groups[0]) or groups[0][0] != 0:
        return False
    groups = groups[:4]
    if any(len(group) != len(groups[0]) or len(group) > 3 for group in groups):
        return False
    centres = np.array([float(np.mean(group)) for group in groups])
    gaps = np.diff(centres)
    return bool(np.min(gaps) >= 6 and np.ptp(gaps) <= 0.25)


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
    right, bottom = left + int(np.max(xs)) + 1, top + int(np.max(ys)) + 1
    left, top = left + int(np.min(xs)), top + int(np.min(ys))
    # Bounded trim: borders are thin; never erode arbitrarily into a uniform panel.
    for _ in range(T.border_trim_pixels):
        crop = image[top:bottom, left:right, :3]
        if min(crop.shape[:2]) <= 2:
            break
        trims = []
        for axis, reverse in ((1, False), (0, False), (1, True), (0, True)):
            band = crop.transpose(1, 0, 2) if axis == 1 else crop
            if reverse:
                band = band[::-1]
            uniform = [
                np.max(np.ptp(edge.astype(float), axis=0)) < T.uniform_rgb_spread
                for edge in band[: T.border_trim_pixels + 1]
            ]
            # Evidence of a thin strip requires a nonuniform interior within
            # the bounded scan. A whole dark cell column must not be eroded.
            trims.append(
                uniform[0] and not all(uniform) and not (reverse and _terminal_grid_stroke(band))
            )
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
    if hints.rows is None or hints.columns is None:
        crop = image[top:bottom, left:right, :3].astype(float)
        variation = min(1.0, float(np.mean(np.std(crop, axis=(0, 1)))) / T.variation_normaliser)
        occupancy = float(
            np.mean(foreground(image, hints.detector.background_distance)[top:bottom, left:right])
        )
        if (hints.rows is not None and (bottom - top) / hints.rows < T.minimum_cell_pitch) or (
            hints.columns is not None and (right - left) / hints.columns < T.minimum_cell_pitch
        ):
            occupancy, variation = 0.0, 0.0
        return {"occupancy": occupancy, "variation": variation, "grid_scoring_available": 0.0}
    pitch_x, pitch_y = (right - left) / hints.columns, (bottom - top) / hints.rows
    if min(pitch_x, pitch_y) < T.minimum_cell_pitch:
        return {
            "occupancy": 0.0,
            "piecewise_constant": 0.0,
            "variation": 0.0,
            "boundary": 0.0,
            "equal_grid": 0.0,
        }
    # At most 32 slots per axis and 3x3 samples per cell, including both ends.
    rows = np.unique(
        np.linspace(0, hints.rows - 1, min(T.grid_slots_per_axis, hints.rows)).astype(int)
    )
    columns = np.unique(
        np.linspace(0, hints.columns - 1, min(T.grid_slots_per_axis, hints.columns)).astype(int)
    )
    colours, errors, local_noise = [], [], []
    for row in rows:
        for column in columns:
            ys = np.minimum(
                bottom - 1, (top + (row + np.array(T.interior_positions)) * pitch_y).astype(int)
            )
            xs = np.minimum(
                right - 1, (left + (column + np.array(T.interior_positions)) * pitch_x).astype(int)
            )
            pixels = image[ys[:, None], xs[None, :], :3].astype(float).reshape(-1, 3)
            colour = np.median(pixels, axis=0)
            colours.append(colour)
            errors.append(float(np.mean(np.abs(pixels - colour))))
            # Independent one-pixel differences estimate raster noise, whereas
            # the spread across separated interior samples measures structure.
            neighbours = []
            for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                near_y = np.clip(ys + dy, top, bottom - 1)
                near_x = np.clip(xs + dx, left, right - 1)
                neighbours.append(
                    image[near_y[:, None], near_x[None, :], :3].astype(float).reshape(-1, 3)
                )
            local_noise.append(
                float(np.median(np.mean(np.abs(np.asarray(neighbours) - pixels), axis=2)))
            )
    variation = float(np.mean(np.std(colours, axis=0)))
    constant = max(0.0, 1 - float(np.mean(errors)) / max(variation, 1))
    grid_colours = np.asarray(colours).reshape(len(rows), len(columns), 3)
    jumps = [
        float(np.mean(np.abs(np.diff(grid_colours, axis=axis))))
        for axis in (0, 1)
        if grid_colours.shape[axis] > 1
    ]
    residual = float(np.mean(errors))
    noise = float(np.mean(local_noise))
    # Never explain away more than half the observed residual as local noise.
    structural_residual = residual - min(noise, residual / 2)
    grid = max(
        0.0, 1 - T.grid_residual_multiplier * structural_residual / max(max(jumps, default=0), 1)
    )
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
        "variation": min(1.0, variation / T.variation_normaliser),
        "boundary": float(np.mean(boundary)),
        "equal_grid": grid,
        "interior_residual_rgb": residual,
        "local_pixel_noise_rgb": noise,
        "structural_residual_rgb": structural_residual,
    }


def _detect_heatmap(image: RGBArray, hints: DetectionHints) -> DetectionResult:
    """Manual boxes bypass localisation. Every detected box requires explicit review."""
    validate_rgb(image)
    height, width = image.shape[:2]
    selected: dict[ROIKind, SelectedRegion] = {}
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
        if (
            hints.rows is not None and hints.rows > hints.heatmap_bbox[3] - hints.heatmap_bbox[1]
        ) or (
            hints.columns is not None
            and hints.columns > hints.heatmap_bbox[2] - hints.heatmap_bbox[0]
        ):
            raise ValueError("each heatmap cell must span at least one pixel per axis")
        if hints.rows is not None and hints.columns is not None:
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
        if (
            evidence["variation"] < T.minimum_variation_score
            or evidence.get("equal_grid", 1) < T.minimum_equal_grid_score
        ):
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
            if (right - left) < T.minimum_analysis_extent or (
                bottom - top
            ) < T.minimum_analysis_extent:
                continue
            box = transform.original_box((left, top, right, bottom))
            margin = math.ceil(max(width / small.shape[1], height / small.shape[0]))
            from .edge_profiles import refine_measured_box

            boxes.add(
                refine_measured_box(
                    image,
                    refine_box(image, box, margin, hints.detector.background_distance, full_mask),
                )
            )
    scored = []
    for box in sorted(boxes):
        if (hints.rows is None or hints.columns is None) and min(
            box[2] - box[0], box[3] - box[1]
        ) < 3 * A.minimum_pitch:
            continue  # The inferred-grid model needs >=3 usable cells per axis.

        components = score_box(image, box, hints, full_mask)
        grid_available = hints.rows is not None and hints.columns is not None
        score = (
            (
                T.heatmap_score_weights[0] * components["occupancy"]
                + T.heatmap_score_weights[1] * components["piecewise_constant"]
                + T.heatmap_score_weights[2] * components["variation"]
                + T.heatmap_score_weights[3] * components["boundary"]
            )
            if grid_available
            else (components["occupancy"] + components["variation"]) / 2
        )
        if (
            score >= hints.detector.min_score
            and components["variation"] >= T.minimum_variation_score
            and (not grid_available or components["equal_grid"] >= T.minimum_equal_grid_score)
        ):
            scored.append((score, box, components))
    scored.sort(key=lambda item: (-item[0], item[1]))
    grid_unresolved = (
        ("grid_scoring_unavailable_until_count_review",)
        if (hints.rows is None or hints.columns is None)
        else ()
    )
    candidates = tuple(
        ROICandidate(
            id=f"heatmap-{index + 1:03d}",
            bbox=box,
            score=score,
            score_components=components,
            reasons=(
                ("review_required", "multiple_plausible_regions")
                if len(scored) > 1
                else ("review_required",)
            )
            + grid_unresolved
            + (
                ("candidate_limit_hides_competition",)
                if len(scored) > hints.detector.max_candidates
                else ()
            ),
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


def _component_boxes(
    image: RGBArray,
    hints: DetectionHints,
    audit: list[dict[str, object]] | None = None,
) -> list[BoundingBox]:
    small, transform = analysis_image(image, hints.detector.max_analysis_size)
    mask = closing(
        foreground(small, hints.detector.background_distance),
        footprint=np.ones((hints.detector.gridline_gap,) * 2),
    )
    regions = sorted(
        regionprops(label(mask)),  # type: ignore[no-untyped-call]
        key=lambda region: (-region.area, region.bbox),
    )
    eligible = []
    for region in regions:
        box = transform.original_box(
            (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2])
        )
        if max(box[2] - box[0], box[3] - box[1]) >= T.bar_minimum_length:
            eligible.append(box)
    if audit is not None:
        audit.append(
            {
                "stage": "component_budget",
                "eligible_windows": len(eligible),
                "truncated": len(eligible) > hints.detector.max_components,
            }
        )
    boxes = eligible[: hints.detector.max_components]
    # Only join collinear pieces with near-identical short-axis extents and a
    # bounded neutral gap. Continuity scoring still rejects label/text assemblies.
    joined = set(boxes)
    for a in boxes:
        for b in boxes:
            if a >= b:
                continue
            for vertical in (True, False):
                cross, along = (0, 1) if vertical else (1, 0)
                if (
                    max(abs(a[cross] - b[cross]), abs(a[cross + 2] - b[cross + 2]))
                    > T.bar_join_cross_tolerance
                ):
                    continue
                span = max(a[along + 2], b[along + 2]) - min(a[along], b[along])
                gap = max(0, max(a[along], b[along]) - min(a[along + 2], b[along + 2]))
                if gap <= T.bar_join_gap_fraction * span:
                    joined.add((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))
    windows = sorted(joined)
    if len(windows) > 192 and audit is not None:
        audit.append(
            {"stage": "window_budget", "available_windows": len(windows), "truncated": True}
        )
    return windows[:192]


def colourbar_candidates(
    image: RGBArray,
    hints: DetectionHints,
    audit: list[dict[str, object]] | None = None,
) -> tuple[ROICandidate, ...]:
    """Components are search windows; retain every plausible colour-bearing core."""
    from .bar_cores import search_bar_cores

    scored = {}
    truncated = False
    windows = _component_boxes(image, hints, audit)
    truncated |= bool(audit and any(e.get("truncated") for e in audit))
    for window in windows:
        cores, evidence = search_bar_cores(image, window, hints.detector.background_distance)
        truncated |= bool(evidence["truncated"])
        if audit is not None:
            audit.append(evidence)
        for refined, orientation in cores:
            left, top, right, bottom = refined
            crop = image[top:bottom, left:right, :3].astype(float)
            if orientation == "horizontal":
                crop = crop.transpose(1, 0, 2)
            if min(crop.shape[:2]) < 2 or max(crop.shape[:2]) / min(crop.shape[:2]) < 4:
                continue
            measured = bar_measurements(crop)
            # Proposal checks identify a potentially continuous core. Extent,
            # numerical QC and competition remain separate acceptance decisions.
            # Border/noise rejection must not erase an alternative from uniqueness.
            if (
                measured["variation_rgb"] < 12
                or measured["changing_fraction"] < 0.2
                or measured["maximum_step_rgb"] > measured["step_limit_rgb"]
            ):
                if audit is not None:
                    audit.append(
                        {
                            "bbox": refined,
                            "rejection": "noncontinuous_or_uniform_core",
                            "measurements": measured,
                        }
                    )
                continue
            components = {
                "short_axis_constancy": max(0.0, 1 - measured["transverse_error"] / 8),
                "variation": min(1.0, measured["variation_rgb"] / 25),
                **measured,
            }
            score = float(
                np.mean(
                    [
                        components[k]
                        for k in (
                            "short_axis_constancy",
                            "continuity",
                            "changing_fraction",
                            "variation",
                        )
                    ]
                )
            )
            reasons = ["review_required", "extent_and_scale_completeness_unverified"]
            if left == 0 or top == 0 or right == image.shape[1] or bottom == image.shape[0]:
                reasons.append("touches_image_edge_possible_clipping")
            scored[refined] = (score, components, orientation, tuple(reasons))
    # Component windows can truncate one physical bar around a neutral centre.
    # Suppress a contained fragment only when its internal endpoint demonstrably
    # continues through a smooth, changing profile into a closed full core.
    # Distinct bars and endpoints with strokes/plateaus remain competitors.
    from .edge_profiles import measure_edges

    fragments = set()
    for box, (_, parts, orientation, _) in scored.items():
        along, cross = (1, 0) if orientation == "vertical" else (0, 1)
        for parent, (_, parent_parts, parent_orientation, _) in scored.items():
            if box == parent or orientation != parent_orientation:
                continue
            if not (parent[along] <= box[along] < box[along + 2] <= parent[along + 2]):
                continue
            if max(abs(parent[cross] - box[cross]), abs(parent[cross + 2] - box[cross + 2])) > 1:
                continue
            if parent_parts["longest_plateau_fraction"] > 0.12:
                continue
            if not all(edge.closed for edge in measure_edges(image, parent)):
                continue
            x, y, r, t = parent
            crop = image[y:t, x:r, :3].astype(float)
            if orientation == "horizontal":
                crop = crop.transpose(1, 0, 2)
            profile = np.median(crop, axis=1)
            internal = [
                end - parent[along]
                for end in (box[along], box[along + 2])
                if parent[along] < end < parent[along + 2]
            ]
            continued = bool(internal)
            for end in internal:
                section = profile[max(0, end - 3) : min(len(profile), end + 3)]
                steps = np.max(np.abs(np.diff(section, axis=0)), axis=1)
                if (
                    not len(steps)
                    or np.max(steps) > T.bar_maximum_step
                    or np.mean(steps > 0.25) < 0.5
                ):
                    continued = False
            if continued:
                fragments.add(box)
                if audit is not None:
                    audit.append(
                        {
                            "rejection": "continued_core_fragment",
                            "bbox": box,
                            "parent_bbox": parent,
                            "evidence": "closed full core; smooth changing internal endpoint",
                        }
                    )
                break
    scored = {box: value for box, value in scored.items() if box not in fragments}
    if len(scored) > hints.detector.max_candidates:
        truncated = True
    return tuple(
        ROICandidate(
            id=f"colourbar-{i + 1:03d}",
            kind="colourbar",
            bbox=box,
            score=score,
            score_components=parts,
            orientation=cast(Literal["vertical", "horizontal"], orientation),
            reasons=reasons + (("candidate_limit_hides_competition",) if truncated else ()),
        )
        for i, (box, (score, parts, orientation, reasons)) in enumerate(
            sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))[
                : hints.detector.max_candidates
            ]
        )
    )


def bar_measurements(crop: NDArray[np.float64]) -> dict[str, float]:
    """Original-resolution continuity, with bounded measured transverse noise support.

    The maximum step check is separate from average continuity: a categorical
    jump cannot hide among hundreds of unchanged pixels. Never smooth along-axis
    jumps away. Input is long-axis x short-axis x RGB.
    """
    profile = np.median(crop, axis=1)
    transverse = float(np.mean(np.abs(crop - profile[:, None])))
    inset = crop.shape[1] // 4
    core = crop[:, inset : crop.shape[1] - inset]
    noise = float(np.quantile(np.abs(np.diff(core, axis=1)), 0.75))
    steps = np.max(np.abs(np.diff(profile, axis=0)), axis=1)
    limit = A.maximum_bar_step + A.bar_noise_multiplier * min(noise, A.maximum_bar_noise)
    plateaus = _runs(np.flatnonzero(steps <= 0.25))
    return {
        "transverse_error": transverse,
        "noise_rgb": noise,
        "variation_rgb": float(np.mean(np.std(profile, axis=0))),
        "continuity": float(np.mean(steps <= limit)),
        "changing_fraction": float(np.mean(steps > 0.25)),
        "maximum_step_rgb": float(np.max(steps[2:-2] if len(steps) > 8 else steps)),
        "endpoint_maximum_step_rgb": float(np.max(steps)),
        "step_limit_rgb": limit,
        "longest_plateau_fraction": max((len(run) for run in plateaus), default=0) / len(profile),
    }


def _associate_bars(
    image: RGBArray,
    heatmaps: dict[str, BoundingBox],
    bars: tuple[ROICandidate, ...],
    hints: DetectionHints | None = None,
) -> tuple["ROIAssociation", ...]:
    from skimage.color import rgb2lab

    from .models import ROIAssociation

    result: list[ROIAssociation] = []
    attempts = 0
    for heatmap_id, box in heatmaps.items():
        left, top, right, bottom = box
        crop = image[top:bottom, left:right, :3]
        # Sample bin interiors rather than the outer origin: origin-aligned
        # strides can land on every grid stroke and miss all cell colours.
        stride_y = max(1, crop.shape[0] // T.association_heatmap_samples)
        stride_x = max(1, crop.shape[1] // T.association_heatmap_samples)
        colours: NDArray[np.float64] = (
            crop[stride_y // 2 :: stride_y, stride_x // 2 :: stride_x].reshape(-1, 3).astype(float)
        )
        sampling = "bounded image-bin centres"
        if hints is not None and hints.rows is not None and hints.columns is not None:
            from .geometry import axis_slots, interior_slice
            from .models import GridConfig

            grid = GridConfig(heatmap_bbox=box, rows=hints.rows, columns=hints.columns)
            row_slots, column_slots = axis_slots(grid, "rows"), axis_slots(grid, "columns")
            row_indices = np.unique(
                np.linspace(0, grid.rows - 1, min(grid.rows, T.association_heatmap_samples)).astype(
                    int
                )
            )
            column_indices = np.unique(
                np.linspace(
                    0, grid.columns - 1, min(grid.columns, T.association_heatmap_samples)
                ).astype(int)
            )
            samples = []
            try:
                for row_index in row_indices:
                    row = row_slots[row_index]
                    ys = interior_slice(row.start, row.end, 0.25)
                    for column_index in column_indices:
                        column = column_slots[column_index]
                        xs = interior_slice(column.start, column.end, 0.25)
                        samples.append(np.median(image[ys, xs, :3], axis=(0, 1)))
            except ValueError:
                samples = []
            if samples:
                colours = np.asarray(samples)
                sampling = "median supplied-count cell interiors; inset 0.25"
        lab = rgb2lab(colours[None].astype(float) / 255)[0]
        for bar in bars:
            if attempts >= A.maximum_association_pairs:
                return tuple(result)
            attempts += 1
            x0, y0, x1, y1 = bar.bbox
            if max(left, x0) < min(right, x1) and max(top, y0) < min(bottom, y1):
                continue
            strip = image[y0:y1, x0:x1, :3]
            profile = np.median(strip, axis=1 if bar.orientation == "vertical" else 0)
            profile = profile[:: max(1, len(profile) // T.association_bar_samples)]
            bar_lab = rgb2lab(profile[None].astype(float) / 255)[0]
            distances = np.sqrt(np.sum((lab[:, None] - bar_lab[None]) ** 2, axis=2)).min(axis=1)
            coverage = float(np.mean(distances <= T.association_maximum_delta_e76))
            gap = math.hypot(max(0, x0 - right, left - x1), max(0, y0 - bottom, top - y1))
            proximity = max(0.0, 1 - gap / max(right - left, bottom - top))
            # Side and physical orientation are independent: vertical bars can
            # sit above/below; horizontal bars can sit left/right.
            overlap_y = max(0, min(bottom, y1) - max(top, y0)) / min(bottom - top, y1 - y0)
            overlap_x = max(0, min(right, x1) - max(left, x0)) / min(right - left, x1 - x0)
            alignment = max(overlap_x, overlap_y)
            if (
                coverage < T.association_minimum_palette_coverage
                or alignment < T.association_minimum_axis_overlap
                or proximity <= 0
            ):
                continue
            result.append(
                ROIAssociation(
                    id=f"association-{len(result) + 1:03d}",
                    heatmap_id=heatmap_id,
                    colourbar_id=bar.id,
                    score=(coverage + alignment + proximity) / 3,
                    score_components={
                        "palette_coverage": coverage,
                        "axis_alignment": alignment,
                        "proximity": proximity,
                    },
                    reasons=(
                        "review_required",
                        "palette_and_proximity_do_not_identify_target",
                        sampling,
                    ),
                )
            )
    return tuple(result)


def _label_searches(
    box: BoundingBox, shape: tuple[int, ...], maximum: int
) -> tuple[tuple[str, BoundingBox], ...]:
    left, top, right, bottom = box
    height, width = shape[:2]
    candidates = (
        ("left", (max(0, left - maximum), top, left, bottom)),
        ("right", (right, top, min(width, right + maximum), bottom)),
        ("above", (left, max(0, top - maximum), right, top)),
        ("below", (left, bottom, right, min(height, bottom + maximum))),
    )
    return tuple(
        (side, region)
        for side, region in candidates
        if region[2] > region[0] and region[3] > region[1]
    )


def _label_regions(
    image: RGBArray,
    hints: DetectionHints,
    heatmap_id: str = "heatmap-manual",
    pixel_budget: list[int] | None = None,
) -> tuple[tuple[ROICandidate, ...], tuple["LabelLocalisationEvidence", ...], tuple[str, ...]]:
    from typing import cast

    from .labels import CropTransform, group_unassigned_tokens, labels_from_tokens, transform_tokens
    from .models import Axis, LabelLocalisationEvidence, LabelsConfig, OCRConfig
    from .ocr import recognize

    assert hints.heatmap_bbox is not None
    settings = hints.labels.ocr if hints.labels is not None else OCRConfig()
    evidence, proposals, warnings = [], [], []
    for side, region in _label_searches(
        hints.heatmap_bbox, image.shape, hints.detector.label_search_pixels
    ):
        axis: Axis = "rows" if side in {"left", "right"} else "columns"
        kind = f"labels_{axis}"
        if kind not in hints.requested and not (
            hints.detector.count_label_geometry and getattr(hints, f"infer_{axis}")
        ):
            continue
        crop = Image.fromarray(image[:, :, :3]).crop(region).convert("L")
        factor = settings.upscale_factor
        pixels = crop.width * crop.height * factor**2
        if pixel_budget is None:
            pixel_budget = [A.maximum_total_ocr_pixels]
        if pixels > hints.detector.max_ocr_pixels or pixels > pixel_budget[0]:
            warnings.append(f"{kind}/{side}: OCR pixel cap exceeded; supply a manual label ROI")
            continue
        pixel_budget[0] -= pixels
        try:
            engine = recognize(
                crop.resize((crop.width * factor, crop.height * factor), Image.Resampling.BICUBIC),
                settings,
            )
        except (ValueError, OSError) as error:
            warnings.append(f"{kind}/{side}: {error}")
            continue
        tokens = transform_tokens(engine.tokens, CropTransform(region, factor))
        grouped = group_unassigned_tokens(tokens)
        assignment = None
        if getattr(hints, axis) is not None:
            # The other axis is unused by label assignment; its placeholder is
            # internal geometry only, never a proposed/extracted scientific count.
            config = GridConfig(
                heatmap_bbox=hints.heatmap_bbox,
                rows=hints.rows or 1,
                columns=hints.columns or 1,
                labels=LabelsConfig(ocr=settings),
            )
            assignment = labels_from_tokens(tokens, config, axis)
        evidence.append(
            LabelLocalisationEvidence(
                axis=axis,
                side=cast(Literal["left", "right", "above", "below"], side),
                search_bbox=region,
                preprocessing=settings,
                tokens=tokens,
                unassigned_candidates=grouped,
                assignment=assignment,
                engine=engine,
                heatmap_id=heatmap_id,
            )
        )
        if assignment is None:
            warnings.append(f"{kind}/{side}: review the count before label-to-slot localisation")
            continue
        slots = [slot for slot in assignment.slots if slot.status in {"accepted", "low_confidence"}]
        # Nearby isolated captions/ticks do not establish an axis region. Require
        # at least two distinct aligned slots (or the only supplied slot).
        count = getattr(hints, axis)
        assert count is not None
        if len(slots) < min(2, count):
            continue
        boxes = [slot.bbox for slot in slots if slot.bbox is not None]
        padding = hints.detector.label_padding_pixels
        box = (
            max(region[0], math.floor(min(b[0] for b in boxes)) - padding),
            max(region[1], math.floor(min(b[1] for b in boxes)) - padding),
            min(region[2], math.ceil(max(b[2] for b in boxes)) + padding),
            min(region[3], math.ceil(max(b[3] for b in boxes)) + padding),
        )
        reasons = ["review_required", "text_identity_unverified"]
        if len(slots) != count:
            reasons.append("incomplete_label_axis")
        if any(slot.status == "ambiguous" for slot in assignment.slots):
            reasons.append("ambiguous_label_assignment")
        if any(slot.status == "low_confidence" for slot in slots):
            reasons.append("low_confidence_text")
        proposals.append(
            ROICandidate(
                id=f"{kind}-{side}"
                if heatmap_id == "heatmap-manual"
                else f"{kind}-{heatmap_id}-{side}",
                kind=cast(Literal["labels_rows", "labels_columns"], kind),
                bbox=box,
                score=len(slots) / count,
                score_components={"aligned_slot_fraction": len(slots) / count},
                reasons=tuple(reasons),
                associations=(heatmap_id,),
            )
        )
    return tuple(proposals), tuple(evidence), tuple(warnings)


def _runs(indices: NDArray[np.int64]) -> list[NDArray[np.int64]]:
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)) if len(indices) else []


def _count_signals(image: RGBArray, box: BoundingBox, axis: Axis) -> dict[str, tuple[float, ...]]:
    from .lattice import observe_axis

    observed = observe_axis(image, box, axis)
    return {"colour_transitions": observed.transitions, "gridline_strokes": observed.strokes}


def count_proposals(
    image: RGBArray,
    box: BoundingBox,
    axis: "Axis",
    hints: DetectionHints,
    heatmap_id: str,
    label_evidence: tuple["LabelLocalisationEvidence", ...] = (),
) -> tuple["CountProposal", ...]:
    """Retain lattice alternatives, including harmonics; never accept a count by rank."""
    from .models import CountProposal

    signals = _count_signals(image, box, axis)
    coordinate = 1 if axis == "rows" else 0
    for record in label_evidence:
        if record.axis == axis:
            # Unassigned geometric groups, not labels fitted to a proposed count.
            # Identify the nearest aligned text band before any count fitting.
            # Keep all raw candidates in label_evidence, including rejected bands.
            candidates = record.unassigned_candidates

            def proximity(candidate: LabelCandidate) -> float:
                b = candidate.bbox
                return (
                    box[0] - b[2]
                    if record.side == "left"
                    else b[0] - box[2]
                    if record.side == "right"
                    else box[1] - b[3]
                    if record.side == "above"
                    else b[1] - box[3]
                )

            ordered = sorted(candidates, key=proximity)
            clusters: list[list[LabelCandidate]] = []
            for c in ordered:
                if not clusters or proximity(c) - proximity(clusters[-1][0]) > 12:
                    clusters.append([c])
                else:
                    clusters[-1].append(c)
            complete_bands = [group for group in clusters if len(group) >= 2]
            grouped = complete_bands[0] if complete_bands else []
            positions = tuple((c.bbox[coordinate] + c.bbox[coordinate + 2]) / 2 for c in grouped)
            if len(positions) >= 2:
                raw_centres = np.sort(np.array(positions, dtype=float))
                gap = float(np.median(np.diff(raw_centres)))
                start, end = box[coordinate], box[coordinate + 2]
                if (
                    gap <= 0
                    or abs(raw_centres[0] - start - gap / 2) > 0.25 * gap
                    or abs(end - raw_centres[-1] - gap / 2) > 0.25 * gap
                ):
                    continue  # Inconclusive caption/tick band, preserved in raw label evidence.
            signals[f"ocr_{record.side}"] = positions
    start, end = box[coordinate], box[coordinate + 2]
    hypotheses = []
    for count in range(
        1, min(hints.detector.max_count, int((end - start) // T.minimum_cell_pitch)) + 1
    ):
        pitch = (end - start) / count
        components = {}
        for name, positions in signals.items():
            if len(positions) < T.count_minimum_observations:
                continue
            centres = name.startswith("ocr_")
            offsets = (np.asarray(positions) - start) / pitch - (0.5 if centres else 0)
            nearest = np.rint(offsets)
            fits = (
                (np.abs(offsets - nearest) <= T.count_position_tolerance_fraction)
                & (nearest >= (0 if centres else 1))
                & (nearest <= (count - 1))
            )
            alignment = float(np.mean(fits))
            coverage = min(
                1.0, len(np.unique(nearest[fits])) / max(1, count if centres else count - 1)
            )
            components[name] = CountSignal(
                positions=positions,
                alignment=alignment,
                coverage=coverage,
                kind="centres" if centres else "boundaries",
                observation=(
                    "unassigned_text_geometry"
                    if centres
                    else "stroke_geometry"
                    if name == "gridline_strokes"
                    else "cell_colours_excluding_strokes"
                ),
                excluded_gridline_pixels=name == "colour_transitions",
            )
        if components and any(
            signal.alignment >= T.count_minimum_alignment for signal in components.values()
        ):
            score = max(signal.alignment * signal.coverage for signal in components.values())
            hypotheses.append((score, count, pitch, components))
    hypotheses.sort(key=lambda h: (-h[0], h[1]))

    def supported(parts: dict[str, CountSignal]) -> bool:
        support = [
            s
            for s in parts.values()
            if s.alignment >= A.minimum_count_alignment and s.coverage >= 0.8
        ]
        return len({s.observation for s in support}) >= 2 and any(
            s.coverage >= A.minimum_count_coverage for s in support
        )

    hidden_supported = any(
        supported(parts) for _, _, _, parts in hypotheses[hints.detector.max_count_candidates :]
    )
    proposals = []
    supplied = getattr(hints, axis)
    for score, count, pitch, components in hypotheses[: hints.detector.max_count_candidates]:
        reasons = [
            "review_required",
            "regular_grid_hypothesis",
            "cell_interiors_and_extent_require_validation",
            "rank_is_not_count_acceptance",
        ]
        if any(
            other != count and (other % count == 0 or count % other == 0)
            for _, other, _, _ in hypotheses
        ):
            reasons.append("integer_multiple_ambiguity")
        if hidden_supported:
            reasons.append("count_limit_hides_supported_competition")
        conflicts: tuple[str, ...] = (
            (f"supplied_{axis}={supplied}; proposal={count}",)
            if (supplied is not None and supplied != count)
            else ()
        )
        conflicts += tuple(
            f"{name}: observation alignment {signal.alignment:.3f} disagrees"
            for name, signal in components.items()
            if signal.alignment < T.count_minimum_alignment
        )
        if any(signal.alignment < T.count_minimum_alignment for signal in components.values()):
            reasons.append("conflicting_observations")
        proposals.append(
            CountProposal(
                id=f"count-{heatmap_id}-{axis}-{count}",
                axis=axis,
                count=count,
                heatmap_id=heatmap_id,
                bbox=box,
                pitch=pitch,
                phase=float(start),
                score=score,
                signals=components,
                reasons=tuple(reasons),
                conflicts=conflicts,
            )
        )
    return tuple(proposals)


def detect(image: RGBArray, hints: DetectionHints) -> DetectionResult:
    """Bounded staged proposals and M10 decisions; released automation stays disabled."""
    from .models import SelectedCount

    validate_rgb(image)
    if image.shape[0] * image.shape[1] > A.max_image_pixels:
        raise ValueError("detection pixel cap exceeded; crop the figure or supply manual inputs")

    # Reuse M7 geometry validation and localisation, without forcing a heatmap
    # detector rerun when fixed coordinates are already supplied.
    heatmap = _detect_heatmap(image, hints)
    if (
        image.shape[2] == 4
        and np.any(image[:, :, 3] != 255)
        and (hints.requested or hints.infer_rows or hints.infer_columns)
    ):
        raise ValueError("ROI/count proposals require an opaque image; supply manual inputs")
    proposals = list(heatmap.candidates) if "heatmap" in hints.requested else []
    selected = dict(heatmap.selected)
    if hints.labels is not None:
        for axis in ("rows", "columns"):
            source = getattr(hints.labels, axis)
            if isinstance(source, OCRLabelSource):
                selected[cast(ROIKind, f"labels_{axis}")] = SelectedRegion(
                    bbox=source.bbox, origin="manual"
                )
    heatmaps = {candidate.id: candidate.bbox for candidate in proposals}
    if hints.heatmap_bbox is not None:
        heatmaps["heatmap-manual"] = hints.heatmap_bbox
    bar_audit: list[dict[str, object]] = []
    bars = colourbar_candidates(image, hints, bar_audit) if "colourbar" in hints.requested else ()
    if hints.colourbar_bbox is not None:
        bars = (
            ROICandidate(
                id="colourbar-manual",
                kind="colourbar",
                bbox=hints.colourbar_bbox,
                orientation=hints.colourbar_orientation
                or (
                    "vertical"
                    if hints.colourbar_bbox[3] - hints.colourbar_bbox[1]
                    >= hints.colourbar_bbox[2] - hints.colourbar_bbox[0]
                    else "horizontal"
                ),
                score=1,
                score_components={},
            ),
        )
    else:
        proposals.extend(bars)
    associations = _associate_bars(image, heatmaps, bars, hints)
    warnings = list(heatmap.warnings)
    label_evidence: tuple[LabelLocalisationEvidence, ...] = ()
    if (
        any(kind.startswith("labels_") for kind in hints.requested)
        or hints.detector.count_label_geometry
        and (hints.infer_rows or hints.infer_columns)
    ):
        budget = [A.maximum_total_ocr_pixels]
        stages = list(heatmaps.items())[: A.max_shortlisted_heatmaps]
        if not stages:
            warnings.append("Label localisation unresolved: no shortlisted heatmap")
        for heatmap_id, box in stages:
            staged_hints = hints.model_copy(update={"heatmap_bbox": box})
            labels, records, label_warnings = _label_regions(
                image, staged_hints, heatmap_id, budget
            )
            label_evidence += records
            proposals.extend(labels)
            warnings.extend(label_warnings)
    counts: list[CountProposal] = []
    axes: tuple[Axis, ...] = ("rows", "columns")
    for axis in axes:
        if getattr(hints, f"infer_{axis}"):
            for heatmap_id, box in heatmaps.items():
                counts.extend(
                    count_proposals(
                        image,
                        box,
                        axis,
                        hints,
                        heatmap_id,
                        tuple(e for e in label_evidence if e.heatmap_id == heatmap_id),
                    )
                )
    unresolved = []
    for kind in hints.requested:
        matches = [c for c in proposals if c.kind == kind]
        if not matches:
            unresolved.append(f"{kind}: no supported candidate; supply manual coordinates")
        else:
            unresolved.append(f"{kind}: explicitly review/select a candidate")
    for axis in ("rows", "columns"):
        if getattr(hints, f"infer_{axis}"):
            unresolved.append(
                f"{axis}: review count proposals; supplied counts remain authoritative"
            )
    status: Literal["manual", "review_required", "ambiguous", "unresolved"] = "manual"
    if unresolved:
        status = "ambiguous" if heatmap.status == "ambiguous" else "review_required"
        if any(not any(c.kind == kind for c in proposals) for kind in hints.requested):
            status = "unresolved"
        elif any(sum(c.kind == kind for c in proposals) > 1 for kind in hints.requested):
            status = "ambiguous"
        if any(
            getattr(hints, f"infer_{axis}")
            and getattr(hints, axis) is None
            and not any(c.axis == axis for c in counts)
            for axis in ("rows", "columns")
        ):
            status = "unresolved"
        elif counts and status == "review_required":
            status = "ambiguous"
    result = DetectionResult(
        candidates=tuple(proposals),
        selected=selected,
        status=status,
        unresolved_requirements=tuple(unresolved),
        warnings=tuple(warnings),
        associations=associations,
        count_proposals=tuple(counts),
        label_evidence=label_evidence,
        bar_search_evidence=tuple(bar_audit),
        selected_counts={
            axis: SelectedCount(count=getattr(hints, axis))
            for axis in ("rows", "columns")
            if getattr(hints, axis) is not None
        },
    )
    from .acceptance import evaluate_acceptance

    acceptance = evaluate_acceptance(image, hints, result)
    resolved_counts = dict(result.selected_counts)
    for d in acceptance.decisions:
        if d.requirement in ("rows", "columns") and d.status == "accepted":
            axis = cast(Axis, d.requirement)
            if getattr(hints, axis) is None:
                resolved_counts[axis] = SelectedCount(
                    count=d.evidence["count"],
                    origin="measured_inference",
                    proposal_id=d.candidate_ids[0],
                )
    return result.model_copy(update={"acceptance": acceptance, "selected_counts": resolved_counts})
