"""Raster, CSV, provenance and diagnostic adapters, separate from calculations."""

import csv
import hashlib
import json
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .diagnostics import diagnostic_image, labels_diagnostic_image
from .extract import extract, extract_labels
from .models import (
    Axis,
    BoolArray,
    CellFlag,
    ExtractionConfig,
    ExtractionResult,
    FileLabelSource,
    GridConfig,
    LabelExtractionResult,
    RGBArray,
)
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


LABEL_OUTPUT_NAMES = ("labels.tsv", "labels_review.tsv", "labels_diagnostic.png", "provenance.json")


def label_input_paths(config: GridConfig, base_directory: Path) -> tuple[Path, ...]:
    """File source paths resolve relative to the configuration file's directory."""
    return (
        tuple(
            base_directory / source.path
            for source in (config.labels.rows, config.labels.columns)
            if isinstance(source, FileLabelSource)
        )
        if config.labels is not None
        else ()
    )


def write_labels_tsv(result: LabelExtractionResult, path: str | Path) -> None:
    """One properly quoted record per slot, including generated omitted axes."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "axis",
                "index",
                "raw_text",
                "final_text",
                "confidence",
                "bbox",
                "source",
                "status",
                "duplicate",
                "issues",
            ]
        )
        for axis in result.axes:
            for slot in axis.slots:
                writer.writerow(
                    [
                        axis.axis,
                        slot.index,
                        slot.raw_text,
                        slot.final_text,
                        slot.confidence,
                        json.dumps(slot.bbox) if slot.bbox is not None else "",
                        slot.source,
                        slot.status,
                        str(slot.duplicate).lower(),
                        "|".join(slot.issues),
                    ]
                )


def write_labels_review(result: LabelExtractionResult, path: str | Path) -> None:
    """Editable template for requested axes only; unresolved entries stay empty."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["axis", "index", "text"])
        for axis in result.axes:
            if result.config.labels is None or getattr(result.config.labels, axis.axis) is None:
                continue
            writer.writerows((axis.axis, slot.index, slot.final_text) for slot in axis.slots)


def label_provenance(
    result: LabelExtractionResult,
    *,
    input_filename: str,
    input_sha256: str,
    image_shape: tuple[int, ...],
    file_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Raw engine evidence is distinct from original-coordinate assignment results."""
    return {
        "schema_version": 2,
        "software": {"name": "heatmap-decoder", "version": __version__},
        "operation": "labels",
        "input": {
            "filename": input_filename,
            "sha256": input_sha256,
            "image_shape": list(image_shape),
        },
        "config": result.config.model_dump(mode="json"),
        "ocr": {
            "availability": "available" if result.ocr_runs else "not_requested",
            "runs": {axis: asdict(run) for axis, run in result.ocr_runs},
            "raw_token_coordinate_frame": "upscaled crop pixels",
            "assigned_text_token_coordinate_frame": "original image pixels",
            "structural_records": "retained in raw coordinates; excluded from assignment",
        },
        "preprocessing": {
            "grayscale": "Pillow L",
            "resize": "Pillow bicubic",
            "upscale_factor": result.config.labels.ocr.upscale_factor
            if result.config.labels is not None
            else None,
        },
        "axes": [asdict(axis) for axis in result.axes],
        "file_label_sha256": file_hashes or {},
        "unresolved_slots": result.unresolved_count,
        "outputs": list(LABEL_OUTPUT_NAMES),
        "limitations": "OCR confidence is a heuristic, not a probability or biological identity. "
        "Verify text and displayed order; only upright horizontal labels are supported.",
    }


def extract_labels_file(
    path: str | Path,
    config: GridConfig,
    directory: str | Path,
    *,
    base_directory: str | Path = ".",
    overwrite: bool = False,
    config_path: str | Path | None = None,
) -> LabelExtractionResult:
    """Extract label evidence and write review files; never publish numeric outputs."""
    path, directory, base = Path(path), Path(directory), Path(base_directory)
    inputs = (path,) + label_input_paths(config, base)
    if config_path is not None:
        inputs += (Path(config_path),)
    outputs = [directory / name for name in LABEL_OUTPUT_NAMES]
    if any(output.resolve() == source.resolve() for output in outputs for source in inputs):
        raise ValueError(
            "label output paths would overwrite an input figure, configuration or labels"
        )
    existing = [output.name for output in outputs if output.exists()]
    if existing and not overwrite:
        raise ValueError(f"outputs already exist ({', '.join(existing)}); use --overwrite")
    labels: dict[Axis, list[str]] = {}
    hashes: dict[str, str] = {}
    axes: tuple[Axis, ...] = ("rows", "columns")
    for axis in axes:
        source = getattr(config.labels, axis) if config.labels is not None else None
        if isinstance(source, FileLabelSource):
            data = (base / source.path).read_bytes()
            labels[axis] = data.decode("utf-8").splitlines()
            hashes[axis] = hashlib.sha256(data).hexdigest()
    data = path.read_bytes()
    image = _decode_raster(data)
    result = extract_labels(image, config, file_labels=labels)
    metadata = label_provenance(
        result,
        input_filename=path.name,
        input_sha256=hashlib.sha256(data).hexdigest(),
        image_shape=image.shape,
        file_hashes=hashes,
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_labels_tsv(result, directory / "labels.tsv")
    write_labels_review(result, directory / "labels_review.tsv")
    labels_diagnostic_image(image, result).save(directory / "labels_diagnostic.png")
    (directory / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result
