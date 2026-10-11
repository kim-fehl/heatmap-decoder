"""Raster and CSV adapters, independent of extraction calculations."""

from pathlib import Path

import numpy as np
from PIL import Image

from .models import ExtractionResult, RGBArray


def load_raster(path: str | Path) -> RGBArray:
    """Read PNG/JPEG as RGB; reject transparency instead of inventing a background."""
    with Image.open(path) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("only PNG and JPEG raster images are supported")
        rgba = np.asarray(image.convert("RGBA"))
        if np.any(rgba[:, :, 3] != 255):
            raise ValueError("transparent pixels are unsupported in M1; use an opaque raster")
        return np.array(image.convert("RGB"), dtype=np.uint8)


def write_csv(result: ExtractionResult, path: str | Path) -> None:
    """Write a headerless/indexless numerical matrix in source row/column order."""
    if result.values.ndim != 2 or 0 in result.values.shape:
        raise ValueError("CSV values must be a nonempty two-dimensional matrix")
    np.savetxt(path, result.values, delimiter=",", fmt="%.17g")
