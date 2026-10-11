"""Deterministic extraction with no filesystem or CLI dependencies."""

from .calibration import calibrate_values, match_colours
from .geometry import validate_geometry
from .models import ExtractionConfig, ExtractionResult, RGBArray
from .sampling import sample_cells, sample_colourbar, validate_rgb


def extract(image: RGBArray, config: ExtractionConfig) -> ExtractionResult:
    """Reconstruct approximate values from an opaque RGB raster and supplied ROIs."""
    validate_rgb(image)
    validate_geometry(config, image.shape)
    cells = sample_cells(image, config)
    bar = sample_colourbar(image, config)
    scale = calibrate_values(len(bar), config.colourbar_start_value, config.colourbar_end_value)
    indices = match_colours(cells, bar)
    return ExtractionResult(values=scale[indices].reshape(config.rows, config.columns))
