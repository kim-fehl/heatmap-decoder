"""Robust median RGB sampling; colourbar edges across its width are excluded."""

import numpy as np

from .calibration import to_lab
from .geometry import box_slices, cell_slices, interior_slice
from .models import BoolArray, ExtractionConfig, FloatArray, RGBArray


def validate_rgb(image: RGBArray) -> None:
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.dtype != np.uint8:
        raise ValueError("image must be a height × width × 3/4 uint8 RGB/RGBA array")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be nonempty")


def sample_cells(image: RGBArray, config: ExtractionConfig) -> FloatArray:
    """One median valid colour per cell, row-major, in [0, 255] or NaN."""
    return sample_cell_data(image, config, missing_pixel_mask(image, config))[0]


def sample_colourbar(image: RGBArray, config: ExtractionConfig) -> FloatArray:
    """Sample every axial pixel, from top/left to bottom/right.

    The configured ROI must exclude axial borders/ticks and span the intended
    limits. The first and last axial pixel samples represent the endpoint values.
    Across the bar width, only the central half is used to avoid edge decoration.
    """
    bar = image[box_slices(config.colourbar_bbox)][:, :, :3]
    if config.colourbar_orientation == "vertical":
        strip = bar[:, interior_slice(0, bar.shape[1], 0.25)]
        return np.asarray(np.median(strip, axis=1), dtype=np.float64)
    strip = bar[interior_slice(0, bar.shape[0], 0.25), :]
    return np.asarray(np.median(strip, axis=0), dtype=np.float64)


def missing_pixel_mask(
    image: RGBArray, config: ExtractionConfig, mask: BoolArray | None = None
) -> BoolArray:
    """True means no data: explicit mask, nonopaque alpha or configured RGB sentinel.

    No RGB colour is assumed missing by default, including black and white.
    Sentinel matching uses ΔE76 and must be explicitly configured.
    """
    missing = np.zeros(image.shape[:2], dtype=np.bool_)
    if mask is not None:
        if mask.shape != missing.shape or mask.dtype != np.bool_:
            raise ValueError("mask must be a boolean array matching image height and width")
        missing |= mask
    if image.shape[2] == 4:
        missing |= image[:, :, 3] != 255
    if config.missing_colours:
        pixels = to_lab(image[:, :, :3].reshape(-1, 3).astype(np.float64))
        colours = to_lab(np.asarray(config.missing_colours, dtype=np.float64))
        for colour in colours:
            distances = np.linalg.norm(pixels - colour, axis=1).reshape(missing.shape)
            missing |= distances <= config.missing_colour_tolerance
    return missing


def sample_cell_data(
    image: RGBArray, config: ExtractionConfig, missing: BoolArray
) -> tuple[FloatArray, FloatArray]:
    """Median of valid interior pixels; NaN when coverage is below the threshold."""
    slices = cell_slices(config)
    colours = np.full((len(slices), 3), np.nan, dtype=np.float64)
    fractions = np.zeros(len(slices), dtype=np.float64)
    for index, (ys, xs) in enumerate(slices):
        valid = ~missing[ys, xs]
        fractions[index] = np.mean(valid)
        if fractions[index] >= config.minimum_valid_fraction:
            colours[index] = np.median(image[ys, xs, :3][valid], axis=0)
    return colours, fractions


def validate_bar_data(missing: BoolArray, config: ExtractionConfig) -> None:
    """Reject an incomplete calibration strip instead of inventing its colours."""
    bar = missing[box_slices(config.colourbar_bbox)]
    if config.colourbar_orientation == "vertical":
        strip = bar[:, interior_slice(0, bar.shape[1], 0.25)]
    else:
        strip = bar[interior_slice(0, bar.shape[0], 0.25), :]
    if np.any(strip):
        raise ValueError("colourbar sampling strip contains missing or transparent pixels")
