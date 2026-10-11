"""Deterministic extraction with no filesystem or CLI dependencies."""

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from PIL import Image

from .calibration import calibrate_values, match_with_quality
from .geometry import axis_slots, validate_geometry, validate_grid_geometry
from .labels import crop_label_region, generated_labels, labels_from_tokens, transform_tokens
from .models import (
    Axis,
    AxisLabelResults,
    BoolArray,
    CellFlag,
    ExtractionConfig,
    ExtractionResult,
    FileLabelSource,
    GridConfig,
    LabelExtractionResult,
    LabelResult,
    OCRRun,
    RGBArray,
)
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
    if config.labels is not None and (
        config.labels.rows is not None or config.labels.columns is not None
    ):
        raise ValueError(
            "labelled reconstruction is reserved for M6; "
            "use the labels command for review, or omit labels for numeric-only extraction"
        )
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


def extract_labels(
    image: RGBArray,
    config: GridConfig,
    *,
    file_labels: Mapping[Axis, Sequence[str]] | None = None,
) -> LabelExtractionResult:
    """Label-only orchestration shared with future reconstruction integration.

    Text-file contents are supplied by I/O, keeping this function filesystem-free.
    The optional wrapper and engine are used only for explicitly requested OCR.
    """
    validate_rgb(image)
    validate_grid_geometry(config, image.shape)
    if config.labels is None or (config.labels.rows is None and config.labels.columns is None):
        raise ValueError("label extraction requires labels.rows or labels.columns with a source")
    axes = []
    runs: list[tuple[Axis, OCRRun]] = []
    for axis in ("rows", "columns"):
        source = getattr(config.labels, axis)
        if source is None:
            axes.append(generated_labels(config, axis))
        elif isinstance(source, FileLabelSource):
            supplied = file_labels.get(axis) if file_labels is not None else None
            if supplied is None:
                raise ValueError(f"labels.{axis} requires supplied UTF-8 text-file labels")
            texts = [text.strip() for text in supplied]
            slots = axis_slots(config, axis)
            if len(texts) != len(slots) or any(not text for text in texts):
                raise ValueError(f"labels.{axis} needs exactly {len(slots)} non-empty strings")
            counts = Counter(texts)
            axes.append(
                AxisLabelResults(
                    axis,
                    tuple(
                        LabelResult(
                            s.index,
                            text,
                            text,
                            None,
                            None,
                            "file",
                            "manual",
                            duplicate=counts[text] > 1,
                            issues=("duplicate_text",) if counts[text] > 1 else (),
                        )
                        for s, text in zip(slots, texts, strict=True)
                    ),
                )
            )
        else:
            from .ocr import recognize

            crop, transform = crop_label_region(Image.fromarray(image), config, axis)
            run = recognize(crop, config.labels.ocr)
            runs.append((axis, run))
            axes.append(labels_from_tokens(transform_tokens(run.tokens, transform), config, axis))
    return LabelExtractionResult(config, tuple(axes), tuple(runs))
