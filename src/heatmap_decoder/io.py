"""Raster, CSV, provenance and diagnostic adapters, separate from calculations."""

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, replace
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from .calibration import colour_quality_summary
from .diagnostics import diagnostic_image, grid_diagnostic_image, labels_diagnostic_image
from .extract import UnresolvedLabelsError, extract, extract_labels
from .geometry import resolve_grid, validate_geometry
from .labels import apply_review, requested_review_slots, validate_label_slots
from .models import (
    Axis,
    BoolArray,
    CellFlag,
    DetectionHints,
    ExtractionConfig,
    ExtractionResult,
    FileLabelSource,
    GridConfig,
    LabelExtractionResult,
    LegendConfig,
    LegendResult,
    LegendTick,
    OCRLabelSource,
    OCRRun,
    ReviewLabels,
    RGBArray,
    ROIReview,
    SavedDetection,
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
        header = [
            "row",
            "column",
            "delta_e76",
            "flags",
            "valid_fraction",
            "candidate_value",
            "value_kind",
            "colour_candidate_evidence",
        ]
        writer.writerow(
            header + (["row_label", "column_label"] if result.labels is not None else [])
        )
        for row, column in np.ndindex(result.values.shape):
            colour_evidence = (
                result.colour_evidence[row * result.config.columns + column]
                if result.colour_evidence is not None
                else None
            )
            writer.writerow(
                [
                    row,
                    column,
                    result.colour_distance[row, column],
                    flag_names(int(result.flags[row, column])),
                    result.valid_fraction[row, column],
                    result.inferred_values[row, column],
                    "unsupported_calibration_span"
                    if result.flags[row, column] & CellFlag.UNCALIBRATED
                    else "missing_or_ambiguous"
                    if not np.isfinite(result.values[row, column])
                    else "possible_upper_bound"
                    if result.range_limit is not None and result.range_limit[row, column] == -1
                    else "possible_lower_bound"
                    if result.range_limit is not None and result.range_limit[row, column] == 1
                    else "estimate",
                    json.dumps(asdict(colour_evidence)) if colour_evidence is not None else "",
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
        "schema_version": 3,
        "software": {"name": "heatmap-decoder", "version": __version__},
        "input": {
            "filename": input_filename,
            "sha256": input_sha256,
            "image_shape": list(result.image_shape),
            "mask_sha256": mask_sha256,
        },
        "config": result.config.model_dump(mode="json"),
        "config_sha256": hashlib.sha256(result.config.model_dump_json().encode()).hexdigest(),
        "resolved_grid": resolve_grid(result.config).metadata(),
        "grid_sha256": resolve_grid(result.config).metadata()["sha256"],
        "qc_thresholds": thresholds,
        "diagnostic_policy": "m16-roi-confidence.1",
        "methods": {
            "sampling": "median RGB of valid inset pixel centres",
            "calibration": "linear position; first/last bar pixel = start/end",
            "colour_distance": "CIELAB Delta E 1976 (D65); not numeric uncertainty",
            "ambiguity": "near-best candidates spanning more than positional threshold",
            "colour_qc_version": "observed-bar-components.1",
            "candidate_components": "inclusive sample endpoints mapped through calibrated scale",
            "candidate_intervals": "engineering tolerance sets; not statistical intervals",
            "missing": "explicit mask, nonopaque alpha, configured missing colours",
            "values": "NaN for missing/ambiguous; poor unambiguous estimates retained",
        },
        "quality": {
            "cells": int(result.flags.size),
            **colour_quality_summary(result),
            "flagged_fraction": float(np.mean(result.flags != 0)),
            "flag_fractions": {
                item.name: float(np.mean((result.flags & item) != 0)) for item in CellFlag
            },
        },
        "outputs": {
            "matrix.csv": "headerless row-major matrix",
            "quality.csv": "long-form QC, zero-based row/column",
            "diagnostic.png": "source sampling / nearest candidates / QC",
            "colourbar_enlarged.png": "observed gradient checkpoints and swatches",
            "gradient.json": "original-coordinate sampling footprints, RGB/Lab, dispersion",
        },
        "limitations": "Approximate displayed-scale values, not exact or pre-transformation data.",
        "recovery": {
            "quantity": result.config.scale.quantity
            if result.config.scale is not None
            else "displayed colourbar values; transformation unspecified",
            "original_expression_counts_recovered": False,
            "endpoint_semantics": "Endpoint colours may be clipped: potential bounds, not exact "
            "values beyond the displayed range; see quality.csv value_kind.",
            "precision": "CSV digits preserve computation, not measurement precision. "
            "Colour distances and fit residuals are QC, not uncertainty intervals.",
        },
    }
    if result.config.scale is not None and result.config.scale.source == "manual_ticks":
        from .calibration import calibration_evidence

        scale = result.config.scale
        metadata["schema_version"] = 4
        metadata["methods"]["calibration"] = f"{scale.model} on original-position anchors"
        metadata["scale"] = {
            "quantity": scale.quantity,
            **calibration_evidence(
                scale, result.config.colourbar_bbox, result.config.colourbar_orientation
            ),
        }
    if result.labels is not None:
        metadata["schema_version"] = max(2, metadata["schema_version"])
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
    detection_review: dict[str, Any] | None = None,
    legend_metadata: dict[str, Any] | None = None,
    automation_metadata: dict[str, Any] | None = None,
    source_image: RGBArray | None = None,
    matrix_filename: str = "matrix.csv",
) -> None:
    """Write all four M2 outputs. Use extract_file to compute input hashes automatically."""
    directory = Path(directory)
    validate_export_labels(result)
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(result, directory / matrix_filename)
    write_quality_csv(result, directory / "quality.csv")
    metadata = provenance(
        result,
        input_filename=input_filename,
        input_sha256=input_sha256,
        mask_sha256=mask_sha256,
        file_hashes=file_hashes,
    )
    if source_image is not None:
        grid_diagnostic_image(source_image, result.config).save(directory / "grid_diagnostic.png")
        metadata["outputs"]["grid_diagnostic.png"] = (
            "resolved source intervals, gaps and group indices"
        )
    if automation_metadata is not None:
        metadata["schema_version"] = max(3, metadata["schema_version"])
        metadata["automation"] = automation_metadata
        metadata["publication"] = {"status": "experimental", "numeric_outputs_written": True}
    if detection_review is not None:
        metadata["roi_review"] = detection_review
    if legend_metadata is not None:
        metadata["schema_version"] = 3
        metadata["legend"] = legend_metadata
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
    roi_review: str | Path | None = None,
    heatmap_candidate: str | None = None,
    roi_selections: str | Path | None = None,
    legend_review: str | Path | None = None,
) -> ExtractionResult:
    """Write review evidence first; only resolved labels may publish numeric outputs."""
    path, directory, base = Path(path), Path(directory), Path(base_directory)
    requested_legend = config.scale is not None and config.scale.source == "ocr_ticks"
    if legend_review is not None and not requested_legend:
        raise ValueError("legend review requires scale.source=ocr_ticks")
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
    if requested_legend:
        output_paths += [directory / name for name in LEGEND_OUTPUT_NAMES]
    if not requested and any(output.resolve() == path.resolve() for output in output_paths):
        raise ValueError("output directory would overwrite the input figure")
    inputs = (path,) + label_input_paths(config, base)
    if config_path is not None:
        inputs += (Path(config_path),)
    if labels_review is not None:
        inputs += (Path(labels_review),)
    if legend_review is not None:
        inputs += (Path(legend_review),)
    if (heatmap_candidate is not None or roi_selections is not None) and roi_review is None:
        raise ValueError("heatmap candidate selection requires --roi-review")
    if roi_review is not None:
        inputs += (Path(roi_review),)
    if roi_selections is not None:
        inputs += (Path(roi_selections),)
    check_output_paths(output_paths, inputs, overwrite)
    reviewed, review_hash = (
        load_labels_review(labels_review, config) if labels_review is not None else (None, None)
    )
    data = path.read_bytes()
    image = _decode_raster(data)
    validate_geometry(config, image.shape)
    detection_review = None
    if roi_review is not None:
        _, detection_review = review_detection(
            Path(roi_review).read_bytes(),
            image,
            hashlib.sha256(data).hexdigest(),
            config.model_dump(mode="json"),
            heatmap_candidate,
            ROIReview.model_validate_json(Path(roi_selections).read_bytes())
            if roi_selections is not None
            else None,
        )
        if roi_selections is not None:
            detection_review["selections_sha256"] = hashlib.sha256(
                Path(roi_selections).read_bytes()
            ).hexdigest()
        if config_path is not None:
            detection_review["config_sha256"] = hashlib.sha256(
                Path(config_path).read_bytes()
            ).hexdigest()
    mask_hash = hashlib.sha256(mask.tobytes(order="C")).hexdigest() if mask is not None else None
    legend_metadata = None
    if requested_legend:
        from .legend import inspect_legend

        assert config.scale is not None
        legend_config = LegendConfig(
            colourbar_bbox=config.colourbar_bbox,
            colourbar_orientation=config.colourbar_orientation,
            legend=config.scale.legend,
            scale_model=config.scale.model,
        )
        legend_result = inspect_legend(image, legend_config)
        digest = hashlib.sha256(data).hexdigest()
        write_legend_outputs(image, legend_result, directory, digest, path.name)
        if legend_review is None:
            raise UnresolvedLegendError(
                "OCR scale requires explicit --legend-review; " + "; ".join(legend_result.reasons)
            )
        raw_evidence = legend_provenance(legend_result, digest, path.name)
        legend_result = load_legend_review(legend_review, legend_result, digest)
        write_legend_outputs(image, legend_result, directory, digest, path.name)
        legend_metadata = legend_provenance(legend_result, digest, path.name)
        legend_metadata["unreviewed_evidence"] = raw_evidence
        legend_metadata["requested_config"] = config.model_dump(mode="json")
        scale = config.scale.model_copy(
            update={"source": "manual_ticks", "anchors": legend_result.reviewed_anchors}
        )
        config = ExtractionConfig.model_validate({**config.model_dump(), "scale": scale})
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
        if detection_review is not None:
            metadata["roi_review"] = detection_review
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
    from .gradient import gradient_diagnostic, inspect_gradient

    directory.mkdir(parents=True, exist_ok=True)
    gradient = inspect_gradient(
        image, config, tuple(a.position for a in config.scale.anchors) if config.scale else ()
    )
    gradient_diagnostic(image, gradient).save(directory / "colourbar_enlarged.png")
    (directory / "gradient.json").write_text(json.dumps(gradient, indent=2) + "\n")
    write_outputs(
        result,
        directory,
        input_filename=path.name,
        input_sha256=hashlib.sha256(data).hexdigest(),
        mask_sha256=mask_hash,
        file_hashes=hashes,
        detection_review=detection_review,
        legend_metadata=legend_metadata,
        source_image=image,
    )
    return result


LABEL_OUTPUT_NAMES = (
    "labels.tsv",
    "labels_review.tsv",
    "labels_diagnostic.png",
    "provenance.json",
    "labels_rows_enlarged.png",
    "labels_columns_enlarged.png",
)
NUMERIC_OUTPUT_NAMES = (
    "grid_diagnostic.png",
    "matrix.csv",
    "quality.csv",
    "provenance.json",
    "diagnostic.png",
    "colourbar_enlarged.png",
    "gradient.json",
)


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


def _label_transform(
    config: GridConfig, axis: Axis, box: tuple[int, int, int, int]
) -> dict[str, Any]:
    from .labels import CropTransform

    source = getattr(config.labels, axis) if config.labels else None
    angle = source.text_rotation_degrees if isinstance(source, OCRLabelSource) else 0
    factor = config.labels.ocr.upscale_factor if config.labels else 1
    white = source.white_padding_pixels if isinstance(source, OCRLabelSource) else 0
    transform = CropTransform(box, factor, angle, white)
    return {
        "crop_bbox": box,
        "text_rotation_degrees": angle,
        "normalisation_rotation_degrees": -angle,
        "upscale_factor": factor,
        "ocr_size": transform.crop_size,
        "white_padding_pixels": white,
        "rotation": transform.metadata()["interpolation"],
        "affine": transform.metadata(),
        "box_mapping": "all four half-open corners; inverse rotation then crop origin",
        "grouping_frame": "upright OCR pixels",
        "assignment_frame": "original image pixels",
    }


def _label_run_geometry(
    config: GridConfig, axis: Axis, box: tuple[int, int, int, int], run: OCRRun
) -> dict[str, Any]:
    """Keep candidate boxes in both frames, linked to the local engine token indices."""
    from .labels import CropTransform, group_tokens, transform_tokens

    details = _label_transform(config, axis, box)
    transform = CropTransform(
        box, details["upscale_factor"], details["text_rotation_degrees"], run.white_padding_pixels
    )
    return {
        "geometry_kind": run.geometry_kind,
        "score_kind": run.score_kind,
        "original_tokens": [asdict(token) for token in transform_tokens(run.tokens, transform)],
        "candidates": [
            {
                **asdict(candidate),
                "ocr_bbox": transform.to_crop(candidate.bbox),
                "original_bbox": candidate.bbox,
            }
            for candidate in group_tokens(run.tokens, config, axis, transform=transform)
        ],
    }


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
        "config_sha256": hashlib.sha256(result.config.model_dump_json().encode()).hexdigest(),
        "resolved_grid": resolve_grid(result.config).metadata(),
        "grid_sha256": resolve_grid(result.config).metadata()["sha256"],
        "ocr": {
            "availability": "available"
            if result.ocr_runs or result.slot_ocr_runs or result.oriented_ocr_records
            else "not_requested",
            "runs": {axis: asdict(run) for axis, run in result.ocr_runs},
            "transforms": {
                axis: _label_transform(result.config, axis, source.bbox)
                for axis, _ in result.ocr_runs
                if result.config.labels is not None
                and isinstance(source := getattr(result.config.labels, axis), OCRLabelSource)
            },
            "run_geometry": {
                axis: _label_run_geometry(result.config, axis, source.bbox, run)
                for axis, run in result.ocr_runs
                if result.config.labels is not None
                and isinstance(source := getattr(result.config.labels, axis), OCRLabelSource)
            },
            "slot_runs": [
                {
                    "axis": axis,
                    "index": index,
                    "crop_bbox": box,
                    "run": asdict(run),
                    "transform": _label_transform(result.config, axis, box),
                    "geometry": _label_run_geometry(result.config, axis, box, run),
                }
                for axis, index, box, run in result.slot_ocr_runs
            ],
            "oriented_lines": result.oriented_ocr_records,
            "skipped_attempts": result.skipped_ocr_attempts,
            "raw_token_coordinate_frame": "upright upscaled crop pixels",
            "assigned_text_token_coordinate_frame": "original image pixels",
            "structural_records": "retained in raw coordinates; excluded from assignment",
        },
        "preprocessing": {
            "grayscale": "Pillow L",
            "retry_policy": "opt-in slots only; original + contrast + threshold; shared budgets",
            "contrast": "Pillow autocontrast cutoff=0",
            "threshold": "fixed luminance >=160 -> 255; otherwise 0",
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
        "Verify text and displayed order; rows must be upright and columns allow "
        "finite explicit angles; arbitrary-angle line identifiers require review.",
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
    for axis in ("rows", "columns"):
        roi = getattr(result.config.labels, axis) if result.config.labels else None
        if isinstance(roi, OCRLabelSource):
            crop = Image.fromarray(image).crop(roi.bbox)
            crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST).save(
                directory / f"labels_{axis}_enlarged.png"
            )
    (directory / "provenance.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


ROI_OUTPUT_NAMES = ("roi_candidates.json", "roi_diagnostic.png", "config_proposed.json")


def detect_file(
    path: str | Path,
    hints: "DetectionHints",
    directory: str | Path,
    *,
    config_path: str | Path | None = None,
    overwrite: bool = False,
) -> "SavedDetection":
    """Export proposals and M10 decisions; never insert unreviewed boxes or counts."""
    from PIL import Image

    from .detection import DETECTOR_VERSION, analysis_image, detect
    from .detection_policy import policy_snapshot, threshold_snapshot

    path, directory = Path(path), Path(directory)
    check_output_paths(
        [directory / name for name in ROI_OUTPUT_NAMES],
        (path,) + ((Path(config_path),) if config_path is not None else ()),
        overwrite,
    )
    base = Path(config_path).parent if config_path is not None else Path(".")
    label_files = (
        tuple(
            base / source.path
            for source in (hints.labels.rows, hints.labels.columns)
            if isinstance(source, FileLabelSource)
        )
        if hints.labels is not None
        else ()
    )
    check_output_paths([directory / name for name in ROI_OUTPUT_NAMES], label_files, overwrite)
    data = path.read_bytes()
    image = _decode_raster(data)
    result = detect(image, hints)
    proposed = hints.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"requested", "detector", "infer_rows", "infer_columns"},
    )
    if "labels" in proposed:
        for axis in ("rows", "columns"):
            source = proposed["labels"].get(axis)
            if source is not None and source["source"] == "file":
                source["path"] = os.path.relpath(
                    (base / source["path"]).resolve(), directory.resolve()
                )
    required = set(ExtractionConfig.model_fields) - {
        key for key, field in ExtractionConfig.model_fields.items() if not field.is_required()
    }
    missing = tuple(sorted(required - proposed.keys()))
    if "scale" not in proposed or proposed["scale"]["source"] == "manual_endpoints":
        missing += tuple(
            k for k in ("colourbar_start_value", "colourbar_end_value") if k not in proposed
        )
    missing += tuple(
        f"labels.{kind.removeprefix('labels_')}"
        for kind in hints.requested
        if kind.startswith("labels_")
    )
    complete = False
    if not missing:
        validate_geometry(ExtractionConfig.model_validate(proposed), image.shape)
        complete = not result.unresolved_requirements
    saved = SavedDetection(
        schema_version=3,
        image_sha256=hashlib.sha256(data).hexdigest(),
        image_dimensions=(image.shape[1], image.shape[0]),
        detector_version=DETECTOR_VERSION,
        detector_thresholds=threshold_snapshot(),
        automation_policy=policy_snapshot(),
        analysis_transforms=tuple(
            asdict(analysis_image(image, maximum)[1])
            for maximum in (hints.detector.max_analysis_size, hints.detector.max_analysis_size // 2)
        )
        if "heatmap" in hints.requested or "colourbar" in hints.requested
        else (),
        hints=hints,
        result=result,
        proposed_config_complete=complete,
        missing_extraction_fields=missing,
    )
    from .diagnostic_style import DiagnosticCanvas

    canvas = DiagnosticCanvas(
        Image.fromarray(image),
        lines=len(result.candidates)
        + sum(len(e.tokens) + len(e.unassigned_candidates) for e in result.label_evidence)
        + 40,
    )
    for candidate in result.candidates:
        canvas.roi(candidate.bbox, candidate.kind, "proposed", candidate.id)
    for kind, region in result.selected.items():
        canvas.roi(
            region.bbox,
            kind,
            "manual" if region.origin == "manual" else "selected",
            region.candidate_id or "",
        )
    for i, evidence in enumerate(result.label_evidence):
        for j, token in enumerate(evidence.tokens):
            if token.text.strip():
                canvas.ocr(
                    token.bbox,
                    token.confidence,
                    f"L{i}-T{j}",
                    token.text,
                    "raw",
                    evidence.preprocessing.min_confidence,
                    evidence.preprocessing.high_confidence,
                )
        for j, text_candidate in enumerate(evidence.unassigned_candidates):
            canvas.ocr(
                text_candidate.bbox,
                text_candidate.confidence,
                f"L{i}-C{j}",
                text_candidate.text,
                evidence.orientation_status,
                evidence.preprocessing.min_confidence,
                evidence.preprocessing.high_confidence,
            )
    for proposal in result.block_proposals:
        canvas.note(f"{proposal.id}: {proposal.status}; {', '.join(proposal.reasons)}")
        x, y, r, b = proposal.bbox
        for block_axis in proposal.axes:
            for start, end in block_axis.gaps:
                box = (x, start, r, end) if block_axis.axis == "rows" else (start, y, end, b)
                canvas.roi(box, "heatmap", "proposed", f"{proposal.id} {block_axis.axis} gap")
            for segment in block_axis.segments:
                pitch = (segment.end - segment.start) / segment.count
                for local in range(segment.count):
                    start, end = segment.start + local * pitch, segment.start + (local + 1) * pitch
                    box = (x, start, r, end) if block_axis.axis == "rows" else (start, y, end, b)
                    canvas.roi(box, "heatmap", "proposed", f"{block_axis.axis} cell {local}")
    canvas.note(result.status)
    if hints.rows is not None and hints.columns is not None:
        canvas.note(f"Supplied dimensions {hints.rows} x {hints.columns}")
    overlay = canvas.image
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "roi_candidates.json").write_text(
        saved.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (directory / "config_proposed.json").write_text(
        json.dumps(proposed, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    overlay.save(directory / "roi_diagnostic.png")
    return saved


def review_detection(
    data: bytes,
    image: RGBArray,
    image_sha256: str,
    config_payload: dict[str, Any],
    candidate_id: str | None,
    selections: ROIReview | None = None,
) -> tuple[ExtractionConfig, dict[str, Any]]:
    """Validate saved M7/M8 evidence and explicit review without rerunning detection."""
    from .detection_policy import policy_snapshot

    saved = SavedDetection.model_validate_json(data)
    if saved.image_sha256 != image_sha256 or saved.image_dimensions != (
        image.shape[1],
        image.shape[0],
    ):
        raise ValueError("saved detection image hash/dimensions do not match the input figure")
    ids = [candidate.id for candidate in saved.result.candidates]
    count_ids = [proposal.id for proposal in saved.result.count_proposals]
    if len(ids + count_ids) != len(set(ids + count_ids)):
        raise ValueError("saved detection has duplicate candidate IDs")
    candidates = {candidate.id: candidate for candidate in saved.result.candidates}
    heatmaps = {
        candidate.id: candidate.bbox
        for candidate in saved.result.candidates
        if candidate.kind == "heatmap"
    }
    if saved.hints.heatmap_bbox is not None:
        heatmaps["heatmap-manual"] = saved.hints.heatmap_bbox
    bars = {candidate.id for candidate in saved.result.candidates if candidate.kind == "colourbar"}
    if saved.hints.colourbar_bbox is not None:
        bars.add("colourbar-manual")
    for box in (saved.hints.heatmap_bbox, saved.hints.colourbar_bbox):
        if box is not None:
            _validate_roi_box(box, image.shape)
    if saved.hints.labels is not None:
        for source in (saved.hints.labels.rows, saved.hints.labels.columns):
            if isinstance(source, OCRLabelSource):
                _validate_roi_box(source.bbox, image.shape)
    for proposal in saved.result.candidates:
        _validate_roi_box(proposal.bbox, image.shape)
        if saved.schema_version >= 3 and proposal.kind.startswith("labels_"):
            if not proposal.associations or any(i not in heatmaps for i in proposal.associations):
                raise ValueError("label proposal references an unknown heatmap")
    if saved.schema_version >= 3:
        for record in saved.result.label_evidence:
            _validate_roi_box(record.search_bbox, image.shape)
            if record.heatmap_id not in heatmaps:
                raise ValueError("label evidence references an unknown heatmap")
    for kind, region in saved.result.selected.items():
        _validate_roi_box(region.bbox, image.shape)
        if region.candidate_id is not None:
            candidate = candidates.get(region.candidate_id)
            if candidate is None or candidate.kind != kind or candidate.bbox != region.bbox:
                raise ValueError(
                    "saved selection references an unknown or conflicting candidate ID"
                )
    association_ids = [a.id for a in saved.result.associations]
    if len(association_ids) != len(set(association_ids)):
        raise ValueError("saved detection has duplicate association IDs")
    for association in saved.result.associations:
        if association.heatmap_id not in heatmaps or association.colourbar_id not in bars:
            raise ValueError("saved association references an unknown region ID")
    for count_proposal in saved.result.count_proposals:
        _validate_roi_box(count_proposal.bbox, image.shape)
        coordinate = 1 if count_proposal.axis == "rows" else 0
        if (
            count_proposal.heatmap_id not in heatmaps
            or heatmaps[count_proposal.heatmap_id] != count_proposal.bbox
        ):
            raise ValueError("count proposal references an unknown/conflicting heatmap")
        for signal in count_proposal.signals.values():
            if any(
                not count_proposal.bbox[coordinate]
                <= position
                <= count_proposal.bbox[coordinate + 2]
                for position in signal.positions
            ):
                raise ValueError("count signal position lies outside its heatmap")
        length = count_proposal.bbox[coordinate + 2] - count_proposal.bbox[coordinate]
        if abs(count_proposal.pitch - length / count_proposal.count) > 1e-8 or (
            count_proposal.phase != count_proposal.bbox[coordinate]
        ):
            raise ValueError("count proposal has contradictory pitch/phase")
    for axis, selected_count in saved.result.selected_counts.items():
        supplied = getattr(saved.hints, axis)
        if selected_count.origin == "manual" and selected_count.count != supplied:
            raise ValueError("saved selected count contradicts supplied count")
        if selected_count.proposal_id is not None:
            choice = next(
                (p for p in saved.result.count_proposals if p.id == selected_count.proposal_id),
                None,
            )
            if choice is None or choice.axis != axis or choice.count != selected_count.count:
                raise ValueError("saved selected count references a conflicting proposal")
    if not isinstance(config_payload, dict):
        raise ValueError("reviewed extraction config must be a JSON object")
    payload = json.loads(json.dumps(config_payload))
    review = selections or ROIReview()
    regions = dict(review.regions)
    if candidate_id is not None:
        if "heatmap" in regions and regions["heatmap"] != candidate_id:
            raise ValueError("conflicting heatmap candidate selections")
        regions["heatmap"] = candidate_id
    for kind, identifier in regions.items():
        candidate = candidates.get(identifier)
        if candidate is None or candidate.kind != kind:
            raise ValueError(f"unknown {kind} candidate ID or wrong region kind")
        if kind.startswith("labels_"):
            label_axis = kind.removeprefix("labels_")
            configured_source = (payload.get("labels") or {}).get(label_axis)
            if (
                configured_source is not None
                and configured_source.get("source") == "ocr"
                and configured_source.get("text_rotation_degrees", 0)
                != candidate.text_rotation_degrees
            ):
                raise ValueError(f"review conflicts with configured labels.{label_axis} angle")
        _apply_roi_region(payload, kind, candidate.bbox)
        if kind.startswith("labels_"):
            label_axis = kind.removeprefix("labels_")
            # A selected region carries its reviewed text orientation into OCR.
            payload["labels"][label_axis]["text_rotation_degrees"] = candidate.text_rotation_degrees
        if kind == "colourbar" and candidate.orientation is not None:
            if payload.get("colourbar_orientation", candidate.orientation) != candidate.orientation:
                raise ValueError(
                    "colourbar orientation conflicts with reviewed physical orientation"
                )
            payload["colourbar_orientation"] = candidate.orientation
    selected_counts = {}
    for axis in ("rows", "columns"):
        supplied = getattr(saved.hints, axis)
        count_identifier = review.counts.get(axis)
        if count_identifier is not None:
            count_choice = next(
                (p for p in saved.result.count_proposals if p.id == count_identifier), None
            )
            if count_choice is None or count_choice.axis != axis:
                raise ValueError(f"unknown {axis} count proposal ID")
            if tuple(payload.get("heatmap_bbox", ())) != count_choice.bbox:
                raise ValueError("reviewed count conflicts with selected heatmap bounds")
            if axis in payload and payload[axis] != count_choice.count:
                raise ValueError(f"reviewed {axis} count conflicts with configured count")
            payload[axis] = count_choice.count
        if supplied is not None and payload.get(axis) != supplied:
            raise ValueError(f"reviewed {axis} conflicts with supplied detection count")
        selected_counts[axis] = {
            "count": payload.get(axis),
            "origin": "reviewed_detection" if count_identifier else "manual",
            "proposal_id": count_identifier,
        }
    config = ExtractionConfig.model_validate(payload)
    validate_geometry(config, image.shape)
    for kind, identifier in regions.items():
        proposal = candidates[identifier]
        if (
            kind.startswith("labels_")
            and proposal.associations
            and not any(heatmaps.get(i) == config.heatmap_bbox for i in proposal.associations)
        ):
            raise ValueError("reviewed label region conflicts with selected heatmap")
    for kind in saved.hints.requested:
        if kind.startswith("labels_"):
            requested_axis = kind.removeprefix("labels_")
            if config.labels is None or getattr(config.labels, requested_axis) is None:
                raise ValueError(f"requested {kind} is unresolved; supply/review its label source")
    final_payload = config.model_dump(mode="json")
    for kind, region in saved.result.selected.items():
        if _roi_config_box(final_payload, kind) != region.bbox:
            raise ValueError(f"review conflicts with fixed manual {kind} bounds")
    if saved.hints.labels is not None:
        for axis in ("rows", "columns"):
            source = getattr(saved.hints.labels, axis)
            if isinstance(source, OCRLabelSource):
                final_source = getattr(config.labels, axis) if config.labels is not None else None
                if (
                    not isinstance(final_source, OCRLabelSource)
                    or final_source.text_rotation_degrees != source.text_rotation_degrees
                ):
                    raise ValueError(f"review conflicts with fixed labels.{axis} angle")
            if isinstance(source, FileLabelSource):
                # Paths may be rebased by export; explicit source type must stay fixed.
                final_source = getattr(config.labels, axis) if config.labels is not None else None
                if not isinstance(final_source, FileLabelSource):
                    raise ValueError(f"review conflicts with fixed labels.{axis} file source")
    selected_regions = {}
    for kind in ("heatmap", "colourbar", "labels_rows", "labels_columns"):
        box = _roi_config_box(final_payload, kind)
        if box is not None:
            selected_regions[kind] = {
                "bbox": list(box),
                "origin": "reviewed_detection" if kind in regions else "manual",
                "candidate_id": regions.get(kind),
            }
    return config, {
        "evidence": saved.model_dump(mode="json"),
        "enforced_automation_policy": policy_snapshot(),
        "review_sha256": hashlib.sha256(data).hexdigest(),
        "origin": "reviewed_detection" if regions or review.counts else "manual",
        "candidate_id": regions.get("heatmap"),
        "reviewed_config": final_payload,
        "selected_regions": selected_regions,
        "selected_counts": selected_counts,
    }


def _roi_config_box(payload: dict[str, Any], kind: str) -> tuple[int, int, int, int] | None:
    if kind.startswith("labels_"):
        labels = payload.get("labels") or {}
        if not isinstance(labels, dict):
            raise ValueError("labels must be an object")
        source = labels.get(kind.removeprefix("labels_")) or {}
        if not isinstance(source, dict):
            raise ValueError("label sources must be objects")
        box = source.get("bbox")
    else:
        box = payload.get(f"{kind}_bbox")
    return tuple(box) if box is not None else None


def _apply_roi_region(payload: dict[str, Any], kind: str, box: tuple[int, int, int, int]) -> None:
    existing = _roi_config_box(payload, kind)
    if existing is not None and existing != box:
        raise ValueError(f"configured {kind}_bbox conflicts with selected candidate")
    if kind.startswith("labels_"):
        axis = kind.removeprefix("labels_")
        labels = payload.get("labels") or {}
        payload["labels"] = labels
        if labels.get(axis) is not None and labels[axis].get("source") != "ocr":
            raise ValueError(f"configured labels.{axis} source conflicts with selected candidate")
        labels[axis] = {"source": "ocr", "bbox": list(box)}
    else:
        payload[f"{kind}_bbox"] = list(box)


def _validate_roi_box(box: tuple[int, int, int, int], shape: tuple[int, ...]) -> None:
    left, top, right, bottom = box
    if not (0 <= left < right <= shape[1] and 0 <= top < bottom <= shape[0]):
        raise ValueError("saved detection box must lie inside the image")


LEGEND_OUTPUT_NAMES = (
    "legend_ticks.tsv",
    "legend_review.tsv",
    "legend_diagnostic.png",
    "legend_fields_enlarged.png",
    "legend_provenance.json",
    "colourbar_enlarged.png",
    "gradient.json",
)


class UnresolvedLegendError(ValueError):
    """Requested OCR scale is unresolved/unreviewed; no new numeric outputs."""


def load_legend_review(
    path: str | Path,
    result: "LegendResult",
    image_sha256: str,
) -> "LegendResult":
    from .calibration import calibration_evidence, fit_linear_scale
    from .models import ScaleAnchor, ScaleConfig

    if any(
        "cap_exceeded" in reason
        or "OCR unavailable/failed" in reason
        or "OCR failed" in reason
        or "cap exceeded" in reason
        for reason in result.reasons
    ):
        raise ValueError(
            "incomplete/unavailable legend evidence; use manual_ticks or fix OCR limits"
        )
    data = Path(path).read_bytes()
    rows = csv.DictReader(StringIO(data.decode("utf-8-sig")), delimiter="\t", strict=True)
    expected_fields = ["index", "image_sha256", "include", "position", "value"]
    if rows.fieldnames != expected_fields:
        raise ValueError(
            "legend review columns must be index, image_sha256, include, position, value"
        )
    expected = {t.index for t in result.ticks}
    seen, anchors = set(), []
    coordinate = 1 if result.config.colourbar_orientation == "vertical" else 0
    start, last = (
        result.config.colourbar_bbox[coordinate],
        result.config.colourbar_bbox[coordinate + 2] - 1,
    )
    try:
        for row in rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed legend review row")
            if not re.fullmatch(r"0|[1-9][0-9]*", row["index"]):
                raise ValueError("invalid legend review index")
            index = int(row["index"])
            if index in seen or index not in expected:
                raise ValueError("duplicate/out-of-range legend review index")
            seen.add(index)
            if row["image_sha256"] != image_sha256:
                raise ValueError("legend review image hash does not match the input image")
            if row["include"] not in {"yes", "no"}:
                raise ValueError("legend review include must be yes or no")
            if row["include"] == "no":
                continue
            if not row["position"].strip() or not row["value"].strip():
                raise ValueError("included legend anchors require non-empty position and value")
            position, value = float(row["position"]), float(row["value"])
            if not np.isfinite([position, value]).all() or not start <= position <= last:
                raise ValueError("legend review requires finite values/positions inside the bar")
            anchors.append(
                ScaleAnchor(position=position, value=value, origin="reviewed", status="reviewed")
            )
    except csv.Error as error:
        raise ValueError(f"invalid legend review TSV: {error}") from error
    if seen != expected:
        raise ValueError("legend review is missing indices")
    scale = ScaleConfig(
        source="manual_ticks", model=result.config.scale_model, anchors=tuple(anchors)
    )
    evidence = calibration_evidence(
        scale, result.config.colourbar_bbox, result.config.colourbar_orientation
    )
    fit = None
    if evidence["linear_comparison"] is not None:
        fit = fit_linear_scale(
            tuple(anchors),
            result.config.colourbar_bbox,
            result.config.colourbar_orientation,
            enforce_residuals=result.config.scale_model == "linear",
        )
    return result.model_copy(
        update={
            "reviewed_anchors": tuple(anchors),
            "calibration_evidence": evidence,
            "fit": fit,
            "status": "reviewed",
            "development_eligible": False,
            "review_sha256": hashlib.sha256(data).hexdigest(),
            "unreviewed_fit": result.fit,
            "unreviewed_reasons": result.reasons,
            "reasons": ("explicit_human_scale_review",),
        }
    )


def legend_provenance(result: "LegendResult", input_sha256: str, filename: str) -> dict[str, Any]:
    from .legend import legend_policy_snapshot

    return {
        "schema_version": 1,
        "operation": "legend",
        "software_version": __version__,
        "input": {"filename": filename, "sha256": input_sha256},
        "policy": legend_policy_snapshot(),
        "result": result.model_dump(mode="json"),
        "sample_position_convention": "first top/left pixel = 0; last bottom/right pixel = 1; "
        "no axial inset; transverse central-half sampling",
        "ocr_coordinate_frame": "raw tokens in upscaled crop; tick/text boxes in original image",
        "review": {"sha256": result.review_sha256},
        "limitations": "Line residuals and OCR confidence do not certify tick values, "
        "biological meaning, original counts or measurement uncertainty.",
    }


def write_legend_outputs(
    image: RGBArray, result: "LegendResult", directory: Path, image_sha256: str, filename: str
) -> None:

    from .legend import search_regions

    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "legend_ticks.tsv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(LegendTick.model_fields)
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for tick in result.ticks:
            row = tick.model_dump(mode="json")
            writer.writerow(
                {
                    k: json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (dict, list))
                    else ""
                    if v is None
                    else v
                    for k, v in row.items()
                }
            )
    with (directory / "legend_review.tsv").open("w", newline="", encoding="utf-8") as stream:
        review_writer = csv.writer(stream, delimiter="\t")
        review_writer.writerow(["index", "image_sha256", "include", "position", "value"])
        selected = {a.position: a.value for a in result.reviewed_anchors}
        for tick in result.ticks:
            review_writer.writerow(
                [
                    tick.index,
                    image_sha256,
                    "yes" if result.status != "reviewed" or tick.position in selected else "no",
                    tick.position,
                    selected.get(tick.position, tick.parsed_value),
                ]
            )
    from .diagnostic_style import DiagnosticCanvas
    from .gradient import gradient_diagnostic, inspect_gradient
    from .labels import CropTransform, transform_tokens

    canvas = DiagnosticCanvas(
        Image.fromarray(image[:, :, :3]),
        lines=len(result.ticks) + sum(len(r.run.tokens) for r in result.ocr_records) + 20,
    )
    canvas.roi(result.config.colourbar_bbox, "colourbar", "manual")
    for side, box in search_regions(image, result.config):
        canvas.roi(box, "tick_search", "proposed", side)
    settings = result.config.legend.ocr
    for i, record in enumerate(result.ocr_records):
        for j, token in enumerate(
            transform_tokens(
                record.run.tokens,
                CropTransform(
                    record.crop_bbox,
                    settings.upscale_factor,
                    white_padding_pixels=record.run.white_padding_pixels,
                ),
            )
        ):
            if token.text.strip():
                canvas.ocr(
                    token.bbox,
                    token.confidence,
                    f"R{i}-T{j}",
                    token.text,
                    "raw",
                    settings.min_confidence,
                    settings.high_confidence,
                )
    for tick in result.ticks:
        if tick.ocr_bbox is not None:
            canvas.roi(tick.ocr_bbox, "tick_search", "proposed", f"tick-{tick.index}-field")
        canvas.ocr(
            tick.text_bbox,
            tick.confidence,
            f"tick-{tick.index}",
            (
                f"{tick.normalised_text} (raw: {tick.raw_text})"
                if tick.normalised_text != tick.raw_text
                else tick.raw_text
            ),
            f"{tick.status}: {','.join(tick.reasons)}",
            settings.min_confidence,
            settings.high_confidence,
        )
    if result.fit is not None:
        x, y, r, b = result.config.colourbar_bbox
        lo, hi = result.fit.recognised_span
        span_box: tuple[float, float, float, float]
        if result.config.colourbar_orientation == "vertical":
            span_box = (x, y + lo * (b - y - 1), r, y + hi * (b - y - 1) + 1)
        else:
            span_box = (x + lo * (r - x - 1), y, x + hi * (r - x - 1) + 1, b)
        canvas.roi(span_box, "calibration_span", "proposed", "no implicit extrapolation")
        canvas.note(f"Observed tick span: {lo:.5g}..{hi:.5g}; outside span unresolved")
    canvas.note(f"{result.status}; review scale; model={result.config.scale_model}")
    overlay = canvas.image
    gradient = inspect_gradient(image, result.config, tuple(t.position for t in result.ticks))
    gradient_diagnostic(image, gradient).save(directory / "colourbar_enlarged.png")
    (directory / "gradient.json").write_text(json.dumps(gradient, indent=2) + "\n")
    overlay.save(directory / "legend_diagnostic.png")
    from .diagnostics import legend_fields_diagnostic_image

    legend_fields_diagnostic_image(image, result).save(directory / "legend_fields_enlarged.png")
    (directory / "legend_provenance.json").write_text(
        json.dumps(legend_provenance(result, image_sha256, filename), indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def inspect_legend_file(
    path: str | Path,
    config: "LegendConfig",
    directory: str | Path,
    *,
    config_path: str | Path | None = None,
    review_path: str | Path | None = None,
    overwrite: bool = False,
) -> "LegendResult":
    from .legend import inspect_legend

    path, directory = Path(path), Path(directory)
    inputs = (
        (path,)
        + ((Path(config_path),) if config_path else ())
        + ((Path(review_path),) if review_path else ())
    )
    check_output_paths([directory / name for name in LEGEND_OUTPUT_NAMES], inputs, overwrite)
    data = path.read_bytes()
    image, digest = _decode_raster(data), hashlib.sha256(data).hexdigest()
    result = inspect_legend(image, config)
    # Preserve all raw evidence before attempting a possibly invalid review.
    write_legend_outputs(image, result, directory, digest, path.name)
    if review_path is not None:
        result = load_legend_review(review_path, result, digest)
        write_legend_outputs(image, result, directory, digest, path.name)
    return result


def write_positional_outputs(
    result: ExtractionResult,
    directory: Path,
    *,
    input_filename: str,
    input_sha256: str,
    source_image: RGBArray,
    automation_metadata: dict[str, Any],
) -> None:
    """Retain numeric results without fulfilling a request for reviewed names."""
    name = (
        "numeric_partial.csv"
        if automation_metadata["numeric_state"] != "complete"
        else "numeric_positional.csv"
    )
    write_outputs(
        result,
        directory,
        input_filename=input_filename,
        input_sha256=input_sha256,
        source_image=source_image,
        automation_metadata=automation_metadata,
        matrix_filename=name,
    )
    with (directory / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["positional_row_id"] + [f"column_{i}" for i in range(result.config.columns)]
        )
        for i, values in enumerate(result.values):
            writer.writerow([f"row_{i}"] + [format(value, ".17g") for value in values])
    path = directory / "provenance.json"
    metadata = json.loads(path.read_text())
    metadata["outputs"].pop("matrix.csv", None)
    metadata["outputs"][name] = "positional IDs, zero-based; these are not recognized names"
    metadata["publication"]["numeric_state"] = automation_metadata["numeric_state"]
    metadata["publication"]["labelled_request_fulfilled"] = False
    path.write_text(json.dumps(metadata, indent=2) + "\n")
