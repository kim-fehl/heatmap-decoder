"""Linear positional calibration and nearest observed colour in CIELAB."""

import numpy as np
from skimage.color import rgb2lab

from .models import FloatArray


def calibrate_values(sample_count: int, start: float, end: float) -> FloatArray:
    """Interpolate by bar position, allowing ascending or descending scales."""
    if sample_count < 2:
        raise ValueError("calibration requires at least two samples")
    if not np.isfinite([start, end]).all() or start == end:
        raise ValueError("calibration endpoints must be finite and distinct")
    return np.linspace(start, end, sample_count, dtype=np.float64)


def _to_lab(colours: FloatArray) -> FloatArray:
    if colours.ndim != 2 or colours.shape[1] != 3 or len(colours) == 0:
        raise ValueError("colours must be a nonempty N × 3 RGB array")
    if not np.isfinite(colours).all() or np.any((colours < 0) | (colours > 255)):
        raise ValueError("RGB colours must be finite and within [0, 255]")
    return np.asarray(rgb2lab(colours / 255.0), dtype=np.float64)


def match_colours(cell_rgb: FloatArray, bar_rgb: FloatArray) -> np.ndarray:
    """Indices minimizing ΔE76, deterministic first-match tie breaking.

    This M1 primitive assumes an identifiable scale. Ambiguity detection and
    per-cell reliability reporting belong to M2. Chunking bounds working memory.
    """
    cells, bar = _to_lab(cell_rgb), _to_lab(bar_rgb)
    indices = np.empty(len(cells), dtype=np.intp)
    chunk_size = max(1, 1_000_000 // len(bar))
    for offset in range(0, len(cells), chunk_size):
        chunk = cells[offset : offset + chunk_size]
        distances = np.sum((chunk[:, None, :] - bar[None, :, :]) ** 2, axis=2)
        indices[offset : offset + len(chunk)] = np.argmin(distances, axis=1)
    return indices
