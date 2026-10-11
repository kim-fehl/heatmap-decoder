"""Raster, CSV, provenance and diagnostic adapters, separate from calculations."""

import csv
import hashlib
import json
import re
from dataclasses import asdict, replace
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from .diagnostics import diagnostic_image, labels_diagnostic_image
from .extract import UnresolvedLabelsError, extract, extract_labels
from .geometry import validate_geometry
from .labels import apply_review, requested_review_slots, validate_label_slots
from .models import (
    Axis,
    BoolArray,
    CellFlag,
    ExtractionConfig,
    ExtractionResult,
    FileLabelSource,
    GridConfig,
    LabelExtractionResult,
    ReviewLabels,
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
    """Numeric-only stays headerless; labelled matrices preserve all strings/duplicates."""
    if result.values.ndim != 2 or 0 in result.values.shape:
        raise ValueError("CSV values must be a nonempty two-dimensional matrix")
    validate_export_labels(result)
    if result.labels is None:
        np.savetxt(path, result.values, delimiter=",", fmt="%.17g")
    else:
        rows, columns = result.labels.axes
        with Path(path).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["gene"] + [slot.final_text for slot in columns.slots])
            for slot, values in zip(rows.slots, result.values, strict=True):
                writer.writerow([slot.final_text] + [format(value, ".17g") for value in values])


def flag_names(flag: int) -> str:
    return "|".join((item.name or str(item)) for item in CellFlag if flag & item) or "OK"


def write_quality_csv(result: ExtractionResult, path: str | Path) -> None:
    """Long-form per-cell diagnostics, with zero-based row and column indices."""
    validate_export_labels(result)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["row", "column", "delta_e76", "flags", "valid_fraction", "candidate_value"]
        writer.writerow(
            header + (["row_label", "column_label"] if result.labels is not None else [])
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
                + (
                    [
                        result.labels.axes[0].slots[row].final_text,
                        result.labels.axes[1].slots[column].final_text,
                    ]
                    if result.labels is not None
                    else []
                )
            )


def validate_export_labels(result: ExtractionResult) -> None:
    if result.labels is not None:
        validate_label_slots(result.labels)
        if result.labels.config != result.config:
            raise ValueError("label results must use the same extraction configuration")
        if result.labels.unresolved_count:
            raise UnresolvedLabelsError(result.labels)


def provenance(
    result: ExtractionResult,
    *,
    input_filename: str,
    input_sha256: str,
    mask_sha256: str | None = None,
    file_hashes: dict[str, str] | None = None,
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
    metadata: dict[str, Any] = {
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
    if result.labels is not None:
        metadata["schema_version"] = 2
        metadata["operation"] = "extract"
        metadata["publication"] = {"status": "published", "numeric_outputs_written": True}
        metadata["labels"] = label_provenance(
            result.labels,
            input_filename=input_filename,
            input_sha256=input_sha256,
            image_shape=result.image_shape,
            file_hashes=file_hashes,
        )
        metadata["outputs"]["matrix.csv"] = "gene column and sample header; duplicates preserved"
        metadata["outputs"]["quality.csv"] = (
            "long-form QC with indices and identical gene/sample strings"
        )
    return metadata


def write_outputs(
    result: ExtractionResult,
    directory: str | Path,
    *,
    input_filename: str,
    input_sha256: str,
    mask_sha256: str | None = None,
    file_hashes: dict[str, str] | None = None,
) -> None:
    """Write all four M2 outputs. Use extract_file to compute input hashes automatically."""
    directory = Path(directory)
    validate_export_labels(result)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(result, directory / "matrix.csv")
    write_quality_csv(result, directory / "quality.csv")
    metadata = provenance(
        result,
        input_filename=input_filename,
        input_sha256=input_sha256,
        mask_sha256=mask_sha256,
        file_hashes=file_hashes,
    )
    if result.labels is not None:
        metadata["outputs"].update(
            {
                name: "label review artifact"
                for name in LABEL_OUTPUT_NAMES
                if name != "provenance.json" and (directory / name).is_file()
            }
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
    base_directory: str | Path = ".",
    overwrite: bool = False,
    config_path: str | Path | None = None,
    labels_review: str | Path | None = None,
) -> ExtractionResult:
    """Write review evidence first; only resolved labels may publish numeric outputs."""
    path, directory, base = Path(path), Path(directory), Path(base_directory)
    requested = config.labels is not None and (
        config.labels.rows is not None or config.labels.columns is not None
    )
    if labels_review is not None and not requested:
        raise ValueError("labels review requires configured label sources")
    output_paths = [directory / name for name in NUMERIC_OUTPUT_NAMES]
    if requested:
        output_paths += [
            directory / name for name in LABEL_OUTPUT_NAMES if name != "provenance.json"
        ]
    if not requested and any(output.resolve() == path.resolve() for output in output_paths):
        raise ValueError("output directory would overwrite the input figure")
    inputs = (path,) + label_input_paths(config, base)
    if config_path is not None:
        inputs += (Path(config_path),)
    if labels_review is not None:
        inputs += (Path(labels_review),)
    check_output_paths(output_paths, inputs, overwrite)
    reviewed, review_hash = (
        load_labels_review(labels_review, config) if labels_review is not None else (None, None)
    )
    data = path.read_bytes()
    image = _decode_raster(data)
    validate_geometry(config, image.shape)
    mask_hash = hashlib.sha256(mask.tobytes(order="C")).hexdigest() if mask is not None else None
    label_result = None
    hashes: dict[str, str] = {}
    if requested:
        file_labels, hashes = load_file_labels(config, base)
        label_result = extract_labels(image, config, file_labels=file_labels)
        if reviewed is not None:
            label_result = apply_review(label_result, reviewed)
            label_result = replace(
                label_result,
                review_sha256=review_hash,
                review_filename=Path(labels_review).name if labels_review is not None else None,
            )
        metadata = label_provenance(
            label_result,
            input_filename=path.name,
            input_sha256=hashlib.sha256(data).hexdigest(),
            image_shape=image.shape,
            file_hashes=hashes,
        )
        metadata["operation"] = "extract"
        metadata["input"]["mask_sha256"] = mask_hash
        metadata["publication"] = {
            "status": "blocked_labels" if label_result.unresolved_count else "labels_ready",
            "numeric_outputs_written": False,
            "existing_numeric_outputs": [
                name for name in ("matrix.csv", "quality.csv") if (directory / name).exists()
            ],
        }
        write_label_outputs(image, label_result, directory, metadata)
        if label_result.unresolved_count:
            raise UnresolvedLabelsError(label_result)
    result = extract(image, config, mask=mask, label_result=label_result)
    write_outputs(
        result,
        directory,
        input_filename=path.name,
        input_sha256=hashlib.sha256(data).hexdigest(),
        mask_sha256=mask_hash,
        file_hashes=hashes,
    )
    return result


LABEL_OUTPUT_NAMES = ("labels.tsv", "labels_review.tsv", "labels_diagnostic.png", "provenance.json")
NUMERIC_OUTPUT_NAMES = ("matrix.csv", "quality.csv", "provenance.json", "diagnostic.png")


def check_output_paths(outputs: list[Path], inputs: tuple[Path, ...], overwrite: bool) -> None:
    if any(output.resolve() == source.resolve() for output in outputs for source in inputs):
        raise ValueError(
            "output paths would overwrite an input figure, configuration, mask, labels or review"
        )
    existing = [output.name for output in outputs if output.exists()]
    if existing and not overwrite:
        raise ValueError(f"outputs already exist ({', '.join(existing)}); use --overwrite")


def load_file_labels(
    config: GridConfig, base: Path
) -> tuple[dict[Axis, list[str]], dict[str, str]]:
    labels: dict[Axis, list[str]] = {}
    hashes: dict[str, str] = {}
    axes: tuple[Axis, ...] = ("rows", "columns")
    for axis in axes:
        source = getattr(config.labels, axis) if config.labels is not None else None
        if isinstance(source, FileLabelSource):
            data = (base / source.path).read_bytes()
            labels[axis] = data.decode("utf-8").splitlines()
            hashes[axis] = hashlib.sha256(data).hexdigest()
    return labels, hashes


def load_labels_review(path: str | Path, config: GridConfig) -> tuple[ReviewLabels, str]:
    """Strict TSV import; duplicates of text are allowed, duplicates of indices are not."""
    expected = requested_review_slots(config)
    if not expected:
        raise ValueError("labels review requires configured label sources")
    data = Path(path).read_bytes()
    reader = csv.reader(StringIO(data.decode("utf-8"), newline=""), delimiter="\t", strict=True)
    reviewed: ReviewLabels = {}
    try:
        if next(reader, None) != ["axis", "index", "text"]:
            raise ValueError("labels review header must be exactly axis, index, text")
        for row in reader:
            if len(row) != 3:
                raise ValueError("labels review rows must have exactly three fields")
            axis, index_text, text = row
            if axis not in {"rows", "columns"}:
                raise ValueError(f"unexpected review axis: {axis}")
            if re.fullmatch(r"0|[1-9][0-9]*", index_text) is None:
                raise ValueError(f"invalid zero-based review index: {index_text}")
            key: tuple[Axis, int] = (cast(Axis, axis), int(index_text))
            if key not in expected:
                raise ValueError(f"unexpected or out-of-range review slot: {axis} {index_text}")
            if key in reviewed:
                raise ValueError(f"duplicate review index: {axis} {index_text}")
            if not text.strip():
                raise ValueError("labels review text must be non-empty for every requested slot")
            reviewed[key] = text.strip()
    except csv.Error as error:
        raise ValueError(f"invalid labels review TSV: {error}") from error
    if set(reviewed) != expected:
        raise ValueError("labels review is missing requested slots or has incorrect counts")
    return reviewed, hashlib.sha256(data).hexdigest()


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
        "unreviewed_axes": [asdict(axis) for axis in result.unreviewed_axes]
        if result.unreviewed_axes is not None
        else None,
        "review": {"filename": result.review_filename, "sha256": result.review_sha256}
        if result.review_sha256 is not None
        else None,
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
    check_output_paths([directory / name for name in LABEL_OUTPUT_NAMES], inputs, overwrite)
    labels, hashes = load_file_labels(config, base)
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
    write_label_outputs(image, result, directory, metadata)
    return result


def write_label_outputs(
    image: RGBArray,
    result: LabelExtractionResult,
    directory: Path,
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_labels_tsv(result, directory / "labels.tsv")
    write_labels_review(result, directory / "labels_review.tsv")
    labels_diagnostic_image(image, result).save(directory / "labels_diagnostic.png")
    (directory / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
