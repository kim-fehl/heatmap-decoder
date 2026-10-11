"""Linear positional calibration and nearest observed colour in CIELAB."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.color import rgb2lab

from .models import BoolArray, FloatArray


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
    chunk_size = max(1, 1_000_000 // len(bar))
    for offset in range(0, len(cells), chunk_size):
        chunk = cells[offset : offset + chunk_size]
        distances = np.sqrt(np.sum((chunk[:, None, :] - bar[None, :, :]) ** 2, axis=2))
        best_index = np.argmin(distances, axis=1)
        best_distance = distances[np.arange(len(chunk)), best_index]
        candidates = distances <= best_distance[:, None] + ambiguity_colour_tolerance
        first = np.argmax(candidates, axis=1)
        last = len(bar) - 1 - np.argmax(candidates[:, ::-1], axis=1)
        target = slice(offset, offset + len(chunk))
        indices[target] = best_index
        distance[target] = best_distance
        ambiguous[target] = (last - first) / (len(bar) - 1) > ambiguity_position_fraction
    return ColourMatch(indices, distance, ambiguous)


def match_colours(cell_rgb: FloatArray, bar_rgb: FloatArray) -> NDArray[np.intp]:
    """Compatibility primitive: nearest indices, without interpreting QC."""
    return match_with_quality(cell_rgb, bar_rgb).indices
