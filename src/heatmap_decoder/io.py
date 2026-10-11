"""Raster, CSV, provenance and diagnostic adapters, separate from calculations."""

import csv
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .diagnostics import diagnostic_image
from .extract import extract
from .models import BoolArray, CellFlag, ExtractionConfig, ExtractionResult, RGBArray
from .version import __version__


def _decode_raster(data: bytes) -> RGBArray:
    with Image.open(BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG"}:
            raise ValueError("only PNG and JPEG raster images are supported")
        rgba = np.asarray(image.convert("RGBA"))
        mode = "RGBA" if np.any(rgba[:, :, 3] != 255) else "RGB"
        return np.array(image.convert(mode), dtype=np.uint8)


def load_raster(path: str | Path) -> RGBArray:
    """Read PNG/JPEG; preserve nonopaque alpha so extraction can mark no data."""
    return _decode_raster(Path(path).read_bytes())


def write_csv(result: ExtractionResult, path: str | Path) -> None:
    """Headerless/indexless numerical matrix; NaN means missing or ambiguous."""
    if result.values.ndim != 2 or 0 in result.values.shape:
        raise ValueError("CSV values must be a nonempty two-dimensional matrix")
    np.savetxt(path, result.values, delimiter=",", fmt="%.17g")


def flag_names(flag: int) -> str:
    return "|".join((item.name or str(item)) for item in CellFlag if flag & item) or "OK"


def write_quality_csv(result: ExtractionResult, path: str | Path) -> None:
    """Long-form per-cell diagnostics, with zero-based row and column indices."""
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["row", "column", "delta_e76", "flags", "valid_fraction", "candidate_value"]
        )
        for row, column in np.ndindex(result.values.shape):
            writer.writerow(
                [
                    row,
                    column,
                    result.colour_distance[row, column],
                    flag_names(int(result.flags[row, column])),
                    result.valid_fraction[row, column],
                    result.inferred_values[row, column],
                ]
            )


def provenance(
    result: ExtractionResult,
    *,
    input_filename: str,
    input_sha256: str,
    mask_sha256: str | None = None,
) -> dict[str, Any]:
    """Deterministic, JSON-safe input identity, configuration, methods and QC summary."""
    thresholds = {
        name: getattr(result.config, name)
        for name in (
            "max_colour_distance",
            "ambiguity_colour_tolerance",
            "ambiguity_position_fraction",
            "minimum_valid_fraction",
            "missing_colour_tolerance",
        )
    }
    return {
        "schema_version": 1,
        "software": {"name": "heatmap-decoder", "version": __version__},
        "input": {
            "filename": input_filename,
            "sha256": input_sha256,
            "image_shape": list(result.image_shape),
            "mask_sha256": mask_sha256,
        },
        "config": result.config.model_dump(mode="json"),
        "qc_thresholds": thresholds,
        "methods": {
            "sampling": "median RGB of valid inset pixel centres",
            "calibration": "linear position; first/last bar pixel = start/end",
            "colour_distance": "CIELAB Delta E 1976 (D65); not numeric uncertainty",
            "ambiguity": "near-best candidates spanning more than positional threshold",
            "missing": "explicit mask, nonopaque alpha, configured missing colours",
            "values": "NaN for missing/ambiguous; poor unambiguous estimates retained",
        },
        "quality": {
            "cells": int(result.flags.size),
            "flagged_fraction": float(np.mean(result.flags != 0)),
            "flag_fractions": {
                item.name: float(np.mean((result.flags & item) != 0)) for item in CellFlag
            },
        },
        "outputs": {
            "matrix.csv": "headerless row-major matrix",
            "quality.csv": "long-form QC, zero-based row/column",
            "diagnostic.png": "source sampling / nearest candidates / QC",
        },
        "limitations": "Approximate displayed-scale values, not exact or pre-transformation data.",
    }


def write_outputs(
    result: ExtractionResult,
    directory: str | Path,
    *,
    input_filename: str,
    input_sha256: str,
    mask_sha256: str | None = None,
) -> None:
    """Write all four M2 outputs. Use extract_file to compute input hashes automatically."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(result, directory / "matrix.csv")
    write_quality_csv(result, directory / "quality.csv")
    metadata = provenance(
        result, input_filename=input_filename, input_sha256=input_sha256, mask_sha256=mask_sha256
    )
    (directory / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    diagnostic_image(result).save(directory / "diagnostic.png")


def extract_file(
    path: str | Path,
    config: ExtractionConfig,
    directory: str | Path,
    *,
    mask: BoolArray | None = None,
) -> ExtractionResult:
    """Read exact source bytes once, extract, hash input/mask, and write M2 outputs."""
    path = Path(path)
    output_paths = [
        Path(directory) / name
        for name in ("matrix.csv", "quality.csv", "provenance.json", "diagnostic.png")
    ]
    if any(output.resolve() == path.resolve() for output in output_paths):
        raise ValueError("output directory would overwrite the input figure")
    data = path.read_bytes()
    result = extract(_decode_raster(data), config, mask=mask)
    mask_hash = hashlib.sha256(mask.tobytes(order="C")).hexdigest() if mask is not None else None
    write_outputs(
        result,
        directory,
        input_filename=path.name,
        input_sha256=hashlib.sha256(data).hexdigest(),
        mask_sha256=mask_hash,
    )
    return result
