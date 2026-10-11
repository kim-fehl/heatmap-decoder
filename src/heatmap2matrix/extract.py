"""Deterministic extraction with no filesystem or CLI dependencies."""

import numpy as np

from .calibration import calibrate_values, match_with_quality
from .geometry import validate_geometry
from .models import BoolArray, CellFlag, ExtractionConfig, ExtractionResult, RGBArray
from .sampling import (
    missing_pixel_mask,
    sample_cell_data,
    sample_colourbar,
    validate_bar_data,
    validate_rgb,
)


def extract(
    image: RGBArray, config: ExtractionConfig, *, mask: BoolArray | None = None
) -> ExtractionResult:
    """Reconstruct approximate values, retaining colour diagnostics and QC flags.

    Explicit mask True and nonopaque alpha denote missing pixels. Ambiguous or
    insufficiently covered cells become NaN; poor matches keep flagged estimates.
    """
    validate_rgb(image)
    validate_geometry(config, image.shape)
    missing = missing_pixel_mask(image, config, mask)
    validate_bar_data(missing, config)
    cells, fractions = sample_cell_data(image, config, missing)
    bar = sample_colourbar(image[:, :, :3], config)
    scale = calibrate_values(len(bar), config.colourbar_start_value, config.colourbar_end_value)
    valid = np.isfinite(cells).all(axis=1)
    count = config.rows * config.columns
    inferred = np.full(count, np.nan, dtype=np.float64)
    distances = np.full(count, np.nan, dtype=np.float64)
    reconstructed = np.full((count, 3), np.nan, dtype=np.float64)
    flags = np.zeros(count, dtype=np.uint8)
    flags[~valid] = CellFlag.MISSING
    if np.any(valid):
        matches = match_with_quality(
            cells[valid],
            bar,
            ambiguity_colour_tolerance=config.ambiguity_colour_tolerance,
            ambiguity_position_fraction=config.ambiguity_position_fraction,
        )
        inferred[valid] = scale[matches.indices]
        distances[valid] = matches.distance
        reconstructed[valid] = bar[matches.indices]
        quality_flags = np.zeros(int(np.sum(valid)), dtype=np.uint8)
        quality_flags[matches.distance > config.max_colour_distance] |= np.uint8(
            CellFlag.POOR_MATCH
        )
        quality_flags[matches.ambiguous] |= np.uint8(CellFlag.AMBIGUOUS)
        flags[valid] = quality_flags
    values = inferred.copy()
    values[(flags & CellFlag.AMBIGUOUS) != 0] = np.nan
    shape = (config.rows, config.columns)
    return ExtractionResult(
        values=values.reshape(shape),
        inferred_values=inferred.reshape(shape),
        colour_distance=distances.reshape(shape),
        flags=flags.reshape(shape),
        valid_fraction=fractions.reshape(shape),
        sampled_rgb=cells.reshape(*shape, 3),
        reconstructed_rgb=reconstructed.reshape(*shape, 3),
        config=config,
        image_shape=image.shape,
    )
