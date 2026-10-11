"""Robust median RGB sampling; colourbar edges across its width are excluded."""

import numpy as np

from .geometry import box_slices, cell_slices, interior_slice
from .models import ExtractionConfig, FloatArray, RGBArray


def validate_rgb(image: RGBArray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a height × width × 3 uint8 RGB array")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be nonempty")


def sample_cells(image: RGBArray, config: ExtractionConfig) -> FloatArray:
    """One median colour per cell, flattened in row-major order, in [0, 255]."""
    return np.asarray(
        [np.median(image[ys, xs].reshape(-1, 3), axis=0) for ys, xs in cell_slices(config)],
        dtype=np.float64,
    )


def sample_colourbar(image: RGBArray, config: ExtractionConfig) -> FloatArray:
    """Sample every axial pixel, from top/left to bottom/right.

    The configured ROI must exclude axial borders/ticks and span the intended
    limits. The first and last axial pixel samples represent the endpoint values.
    Across the bar width, only the central half is used to avoid edge decoration.
    """
    bar = image[box_slices(config.colourbar_bbox)]
    if config.colourbar_orientation == "vertical":
        strip = bar[:, interior_slice(0, bar.shape[1], 0.25)]
        return np.asarray(np.median(strip, axis=1), dtype=np.float64)
    strip = bar[interior_slice(0, bar.shape[0], 0.25), :]
    return np.asarray(np.median(strip, axis=0), dtype=np.float64)
