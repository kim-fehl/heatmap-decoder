"""Linear positional calibration and nearest observed colour in CIELAB."""

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray
from skimage.color import rgb2lab

from .models import BoolArray, FloatArray

if TYPE_CHECKING:
    from .models import (
        BoundingBox,
        ExtractionConfig,
        ExtractionResult,
        LinearScaleFit,
        ScaleAnchor,
        ScaleConfig,
    )


def calibrate_values(sample_count: int, start: float, end: float) -> FloatArray:
    """Interpolate by bar position, allowing ascending or descending scales."""
    if sample_count < 2:
        raise ValueError("calibration requires at least two samples")
    if not np.isfinite([start, end]).all() or start == end:
        raise ValueError("calibration endpoints must be finite and distinct")
    if not np.isfinite(end - start):
        raise ValueError("calibration numeric range must be finite")
    return np.linspace(start, end, sample_count, dtype=np.float64)


def to_lab(colours: FloatArray) -> FloatArray:
    if colours.ndim != 2 or colours.shape[1] != 3 or len(colours) == 0:
        raise ValueError("colours must be a nonempty N × 3 RGB array")
    if not np.isfinite(colours).all() or np.any((colours < 0) | (colours > 255)):
        raise ValueError("RGB colours must be finite and within [0, 255]")
    return np.asarray(rgb2lab(colours / 255.0), dtype=np.float64)


@dataclass(frozen=True)
class ColourMatch:
    indices: NDArray[np.intp]
    distance: FloatArray
    ambiguous: BoolArray
    candidate_intervals: tuple[tuple[tuple[int, int], ...], ...] = ()


def match_with_quality(
    cell_rgb: FloatArray,
    bar_rgb: FloatArray,
    *,
    ambiguity_colour_tolerance: float = 1.0,
    ambiguity_position_fraction: float = 0.05,
) -> ColourMatch:
    """Nearest ΔE76 match, with ambiguity across distant positional candidates.

    Candidates are within best ΔE76 + tolerance. Flag if their total positional
    span exceeds the configured fraction of the full bar. This catches repeated
    colours and extended saturated plateaus, including those at endpoints.
    Inclusive sample-index components retain disconnected near-matches separately.
    This evidence does not relax the total-span rule.
    Chunking bounds temporary cell × bar working memory.
    """
    if not np.isfinite([ambiguity_colour_tolerance, ambiguity_position_fraction]).all():
        raise ValueError("ambiguity thresholds must be finite")
    if ambiguity_colour_tolerance < 0 or not 0 < ambiguity_position_fraction <= 1:
        raise ValueError("invalid ambiguity thresholds")
    cells, bar = to_lab(cell_rgb), to_lab(bar_rgb)
    if len(bar) < 2:
        raise ValueError("colourbar requires at least two positions")
    indices = np.empty(len(cells), dtype=np.intp)
    distance = np.empty(len(cells), dtype=np.float64)
    ambiguous = np.empty(len(cells), dtype=np.bool_)
    intervals: list[tuple[tuple[int, int], ...]] = []
    chunk_size = max(1, 1_000_000 // len(bar))
    for offset in range(0, len(cells), chunk_size):
        chunk = cells[offset : offset + chunk_size]
        distances = np.sqrt(np.sum((chunk[:, None, :] - bar[None, :, :]) ** 2, axis=2))
        best_index = np.argmin(distances, axis=1)
        best_distance = distances[np.arange(len(chunk)), best_index]
        candidates = distances <= best_distance[:, None] + ambiguity_colour_tolerance
        for mask in candidates:
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            intervals.append(tuple((int(a), int(b)) for a, b in zip(starts, ends, strict=True)))
        first = np.argmax(candidates, axis=1)
        last = len(bar) - 1 - np.argmax(candidates[:, ::-1], axis=1)
        target = slice(offset, offset + len(chunk))
        indices[target] = best_index
        distance[target] = best_distance
        ambiguous[target] = (last - first) / (len(bar) - 1) > ambiguity_position_fraction
    return ColourMatch(indices, distance, ambiguous, tuple(intervals))


def match_colours(cell_rgb: FloatArray, bar_rgb: FloatArray) -> NDArray[np.intp]:
    """Compatibility primitive: nearest indices, without interpreting QC."""
    return match_with_quality(cell_rgb, bar_rgb).indices


def normalise_bar_positions(
    positions: FloatArray,
    bbox: "BoundingBox",
    orientation: str,
) -> FloatArray:
    """0 = first top/left sample, 1 = last bottom/right sample; no axial inset."""
    coordinate = 1 if orientation == "vertical" else 0
    start, last = bbox[coordinate], bbox[coordinate + 2] - 1
    if orientation not in {"vertical", "horizontal"} or last <= start:
        raise ValueError("colourbar requires a valid orientation and at least two axial samples")
    if not np.isfinite(positions).all():
        raise ValueError("anchor positions must be finite")
    return np.asarray((positions - start) / (last - start), dtype=np.float64)


def fit_linear_scale(
    anchors: tuple["ScaleAnchor", ...],
    bbox: "BoundingBox",
    orientation: str,
    *,
    max_value_residual_fraction: float = 0.01,
    max_position_residual_pixels: float = 2.0,
    endpoint_tolerance_pixels: float = 2.0,
    enforce_residuals: bool = True,
) -> "LinearScaleFit":
    """Fit all spatial anchors; never discard an outlier or assume equal spacing."""
    from .models import LinearScaleFit

    if len(anchors) < 2:
        raise ValueError("linear scale requires at least two anchors")
    ordered = sorted(anchors, key=lambda anchor: anchor.position)
    positions = np.array([anchor.position for anchor in ordered], dtype=np.float64)
    values = np.array([anchor.value for anchor in ordered], dtype=np.float64)
    t = normalise_bar_positions(positions, bbox, orientation)
    coordinate = 1 if orientation == "vertical" else 0
    length = bbox[coordinate + 2] - bbox[coordinate] - 1
    tolerance = endpoint_tolerance_pixels / length
    if np.any(t < -tolerance) or np.any(t > 1 + tolerance):
        raise ValueError("anchor position lies outside the colourbar")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("duplicate/conflicting anchor positions")
    if not np.isfinite(values).all() or not np.isfinite(np.ptp(values)) or np.ptp(values) == 0:
        raise ValueError("scale needs finite distinct values with nonzero range")
    differences = np.diff(values)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("scale anchors must be strictly monotonic")
    design = np.column_stack((t, np.ones(len(t))))
    condition = float(np.linalg.cond(design))
    if condition > 1000 or float(np.ptp(t)) * length < 1:
        raise ValueError("ill-conditioned scale fit; anchor positions are too close")
    # Rescale values for fitting to avoid numerical overflow at large finite units.
    reference, value_range = values[0], float(np.ptp(values))
    coefficients = np.linalg.lstsq(design, (values - reference) / value_range, rcond=None)[0]
    slope = float(coefficients[0] * value_range)
    intercept = float(coefficients[1] * value_range + reference)
    residual = values - (slope * t + intercept)
    position_residual = residual / slope * length
    if not np.isfinite([slope, intercept]).all() or slope == 0:
        raise ValueError("invalid linear scale coefficients")
    within_limits = not (
        np.max(np.abs(residual)) / value_range > max_value_residual_fraction + 1e-12
        or np.max(np.abs(position_residual)) > max_position_residual_pixels + 1e-8
    )
    if enforce_residuals and not within_limits:
        raise ValueError("anchors contradict a linear scale; inspect values and positions")
    return LinearScaleFit(
        slope=slope,
        intercept=intercept,
        normalised_positions=tuple(t.tolist()),
        value_residuals=tuple(residual.tolist()),
        position_residuals_pixels=tuple(position_residual.tolist()),
        recognised_span=(float(t[0]), float(t[-1])),
        condition_number=condition,
        tests_linearity=len(anchors) >= 3,
        endpoint_coverage=abs(t[0]) * length <= endpoint_tolerance_pixels + 1e-8
        and abs(t[-1] - 1) * length <= endpoint_tolerance_pixels + 1e-8,
        within_residual_limits=within_limits,
    )


def piecewise_values(
    anchors: tuple["ScaleAnchor", ...],
    bbox: "BoundingBox",
    orientation: str,
    sample_count: int,
) -> FloatArray:
    """Original-coordinate monotonic knots; uncovered samples stay NaN, never extrapolate."""
    if sample_count < 2 or len(anchors) < 3:
        raise ValueError("piecewise calibration requires at least three anchors and two samples")
    if any(a.status != "reviewed" or a.origin == "ocr" for a in anchors):
        raise ValueError("piecewise anchors must be explicitly supplied/reviewed")
    positions = np.array([a.position for a in anchors], dtype=np.float64)
    values = np.array([a.value for a in anchors], dtype=np.float64)
    t = normalise_bar_positions(positions, bbox, orientation)
    if np.any((t < 0) | (t > 1)):
        raise ValueError("anchor position lies outside the colourbar")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("anchors require distinct spatially ordered positions")
    differences = np.diff(values)
    if not np.isfinite(values).all() or not np.isfinite(differences).all():
        raise ValueError("piecewise numeric range must be finite")
    if not np.isfinite(np.ptp(values)) or not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("scale anchors must be strictly monotonic with finite range")
    if not np.isfinite(differences / np.diff(positions)).all():
        raise ValueError("piecewise segment slopes must be finite")
    return np.asarray(
        np.interp(np.linspace(0, 1, sample_count), t, values, left=np.nan, right=np.nan),
        dtype=np.float64,
    )


def calibration_evidence(
    scale: "ScaleConfig", bbox: "BoundingBox", orientation: str
) -> dict[str, Any]:
    """Retain failed linear comparison without using it to choose a model."""
    if scale.model == "piecewise_linear":
        coordinate = 1 if orientation == "vertical" else 0
        piecewise_values(scale.anchors, bbox, orientation, bbox[coordinate + 2] - bbox[coordinate])
    fit = None
    comparison_error = None
    try:
        fit = fit_linear_scale(
            scale.anchors,
            bbox,
            orientation,
            max_value_residual_fraction=scale.max_value_residual_fraction,
            max_position_residual_pixels=scale.max_position_residual_pixels,
            enforce_residuals=scale.model == "linear",
        )
    except ValueError as error:
        if scale.model == "linear":
            raise
        comparison_error = str(error)
    evidence: dict[str, Any] = {
        "model": scale.model,
        "source": scale.source,
        "anchors": [a.model_dump(mode="json") for a in scale.anchors],
        "linear_comparison": fit.model_dump(mode="json") if fit else None,
        "linear_comparison_error": comparison_error,
        "fit": fit.model_dump(mode="json") if fit else None,
    }
    if scale.model == "piecewise_linear":
        coordinate = 1 if orientation == "vertical" else 0
        length = bbox[coordinate + 2] - bbox[coordinate]
        calibrated = piecewise_values(scale.anchors, bbox, orientation, length)
        positions = [a.position for a in scale.anchors]
        values = [a.value for a in scale.anchors]
        evidence.update(
            {
                "knots": list(zip(positions, values, strict=True)),
                "segment_slopes": (np.diff(values) / np.diff(positions)).tolist(),
                "covered_position_range": [positions[0], positions[-1]],
                "covered_value_range": [min(values), max(values)],
                "full_span": bool(np.isfinite(calibrated).all()),
                "unsupported_samples": int(np.sum(~np.isfinite(calibrated))),
                "extrapolation": "disabled; supply/review physical edge anchors for full span",
                "approximation": "zero knot residual does not validate mapping between ticks",
            }
        )
    return evidence


def scale_values(
    config: "ExtractionConfig",
    sample_count: int,
    *,
    ocr_anchors: tuple["ScaleAnchor", ...] | None = None,
) -> FloatArray:
    """Legacy endpoints stay bit-for-bit identical; tick scales use their fitted line."""
    if sample_count < 2:
        raise ValueError("calibration requires at least two samples")
    if config.scale is None or config.scale.source == "manual_endpoints":
        if config.colourbar_start_value is None or config.colourbar_end_value is None:
            raise ValueError("manual scale endpoints are unresolved")
        return calibrate_values(
            sample_count, config.colourbar_start_value, config.colourbar_end_value
        )
    if config.scale.source == "ocr_ticks":
        if ocr_anchors is None:
            raise ValueError("OCR scale requires legend inspection and explicit --legend-review")
        if config.scale.model != "linear" or len(ocr_anchors) < 3:
            raise ValueError("automatic OCR calibration requires three linear physical anchors")
        if any(a.origin != "ocr" or a.status != "suggested" for a in ocr_anchors):
            raise ValueError("runtime OCR anchors must retain their observed origin/status")
        fit = fit_linear_scale(
            ocr_anchors,
            config.colourbar_bbox,
            config.colourbar_orientation,
            max_value_residual_fraction=min(0.01, config.scale.max_value_residual_fraction),
            max_position_residual_pixels=min(2.0, config.scale.max_position_residual_pixels),
        )
        positions = np.linspace(0, 1, sample_count)
        values = np.asarray(fit.slope * positions + fit.intercept, dtype=np.float64)
        # Every near-colour candidate must be supported, not just the best match.
        lo, hi = fit.recognised_span
        values[(positions < lo) | (positions > hi)] = np.nan
        return values
    if ocr_anchors is not None:
        raise ValueError("runtime OCR anchors require an OCR scale")
    if any(a.status != "reviewed" or a.origin == "ocr" for a in config.scale.anchors):
        raise ValueError("tick anchors must be explicitly supplied/reviewed")
    if config.scale.model == "piecewise_linear":
        return piecewise_values(
            config.scale.anchors,
            config.colourbar_bbox,
            config.colourbar_orientation,
            sample_count,
        )
    fit = fit_linear_scale(
        config.scale.anchors,
        config.colourbar_bbox,
        config.colourbar_orientation,
        max_value_residual_fraction=config.scale.max_value_residual_fraction,
        max_position_residual_pixels=config.scale.max_position_residual_pixels,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(
            fit.slope * np.linspace(0, 1, sample_count) + fit.intercept, dtype=np.float64
        )
    if not np.isfinite(values).all():
        raise ValueError("fitted scale exceeds finite numerical range")
    return values


def colour_quality_summary(result: "ExtractionResult") -> dict[str, Any]:
    """Summarise engineering ambiguity evidence without changing acceptance."""
    evidence = [e for e in result.colour_evidence or () if e is not None]
    return {
        "colour_qc_version": "observed-bar-components.1",
        "ambiguity_kind_counts": dict(Counter(e.kind for e in evidence)),
        "maximum_candidate_value_span": max(
            (e.value_span for e in evidence if e.value_span is not None), default=None
        ),
        "components_are_statistical_intervals": False,
    }
