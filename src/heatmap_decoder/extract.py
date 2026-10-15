"""Deterministic extraction with no filesystem or CLI dependencies."""

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Literal

import numpy as np
from PIL import Image, ImageOps

from .calibration import match_with_quality, scale_values
from .geometry import axis_slots, validate_geometry, validate_grid_geometry
from .labels import (
    CropTransform,
    apply_character_policy,
    apply_review,
    assign_candidates,
    crop_label_region,
    generated_labels,
    group_tokens,
    transform_tokens,
    validate_label_slots,
)
from .models import (
    Axis,
    AxisLabelResults,
    BoolArray,
    CellColourEvidence,
    CellFlag,
    ColourCandidateComponent,
    ExtractionConfig,
    ExtractionResult,
    FileLabelSource,
    GridConfig,
    LabelCandidate,
    LabelExtractionResult,
    LabelResult,
    OCRLabelSource,
    OCRRun,
    OCRToken,
    RGBArray,
    ScaleAnchor,
)
from .sampling import (
    missing_pixel_mask,
    sample_cell_data,
    sample_colourbar,
    validate_bar_data,
    validate_rgb,
)


class UnresolvedLabelsError(ValueError):
    """Label evidence exists, but numeric outputs must not be published."""

    def __init__(self, labels: LabelExtractionResult) -> None:
        self.labels = labels
        super().__init__(
            f"{labels.unresolved_count} unresolved label slots; "
            "complete labels_review.tsv and rerun with --labels-review"
        )


def extract(
    image: RGBArray,
    config: ExtractionConfig,
    *,
    mask: BoolArray | None = None,
    ocr_anchors: tuple[ScaleAnchor, ...] | None = None,
    label_result: LabelExtractionResult | None = None,
    file_labels: Mapping[Axis, Sequence[str]] | None = None,
    reviewed_labels: Mapping[tuple[Axis, int], str] | None = None,
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
        if label_result is None:
            label_result = extract_labels(image, config, file_labels=file_labels)
        if label_result.config != config:
            raise ValueError("precomputed labels must use the same extraction configuration")
        validate_label_slots(label_result)
        if reviewed_labels is not None:
            label_result = apply_review(label_result, reviewed_labels)
        if label_result.unresolved_count:
            raise UnresolvedLabelsError(label_result)
    elif label_result is not None or reviewed_labels is not None:
        raise ValueError("labels or review provided without configured label sources")
    missing = missing_pixel_mask(image, config, mask)
    validate_bar_data(missing, config)
    cells, fractions = sample_cell_data(image, config, missing)
    bar = sample_colourbar(image[:, :, :3], config)
    scale = scale_values(config, len(bar), ocr_anchors=ocr_anchors)
    valid = np.isfinite(cells).all(axis=1)
    count = config.rows * config.columns
    inferred = np.full(count, np.nan, dtype=np.float64)
    distances = np.full(count, np.nan, dtype=np.float64)
    reconstructed = np.full((count, 3), np.nan, dtype=np.float64)
    flags = np.zeros(count, dtype=np.uint8)
    limits = np.zeros(count, dtype=np.int8)
    evidence: list[CellColourEvidence | None] = [None] * count
    flags[~valid] = CellFlag.MISSING
    if np.any(valid):
        matches = match_with_quality(
            cells[valid],
            bar,
            ambiguity_colour_tolerance=config.ambiguity_colour_tolerance,
            ambiguity_position_fraction=config.ambiguity_position_fraction,
        )
        for cell, intervals in zip(np.flatnonzero(valid), matches.candidate_intervals, strict=True):
            components = tuple(
                ColourCandidateComponent(
                    a,
                    b,
                    a / (len(bar) - 1),
                    b / (len(bar) - 1),
                    float(np.min(scale[a : b + 1]))
                    if np.isfinite(scale[a : b + 1]).all()
                    else None,
                    float(np.max(scale[a : b + 1]))
                    if np.isfinite(scale[a : b + 1]).all()
                    else None,
                    bool(np.isfinite(scale[a : b + 1]).all()),
                )
                for a, b in intervals
            )
            span = (intervals[-1][1] - intervals[0][0]) / (len(bar) - 1)
            kind: Literal["local", "broad_contiguous", "disconnected_near_matches"] = (
                "local"
                if span <= config.ambiguity_position_fraction
                else "broad_contiguous"
                if len(components) == 1
                else "disconnected_near_matches"
            )
            evidence[int(cell)] = CellColourEvidence(
                kind,
                components,
                span,
                max(c.maximum_value for c in components if c.maximum_value is not None)
                - min(c.minimum_value for c in components if c.minimum_value is not None)
                if all(c.calibrated for c in components)
                else None,
            )
        inferred[valid] = scale[matches.indices]
        distances[valid] = matches.distance
        reconstructed[valid] = bar[matches.indices]
        at_start = np.all(bar[matches.indices] == bar[0], axis=1)
        at_end = np.all(bar[matches.indices] == bar[-1], axis=1)
        covered = scale[np.isfinite(scale)]
        direction = (1 if covered[-1] > covered[0] else -1) if len(covered) else 1
        limits[valid] = np.where(at_start, -direction, np.where(at_end, direction, 0))
        quality_flags = np.zeros(int(np.sum(valid)), dtype=np.uint8)
        quality_flags[matches.distance > config.max_colour_distance] |= np.uint8(
            CellFlag.POOR_MATCH
        )
        quality_flags[matches.ambiguous] |= np.uint8(CellFlag.AMBIGUOUS)
        unsupported = np.array(
            [
                any(not np.isfinite(scale[a : b + 1]).all() for a, b in intervals)
                for intervals in matches.candidate_intervals
            ]
        )
        quality_flags[unsupported] |= np.uint8(CellFlag.UNCALIBRATED)
        inferred[np.flatnonzero(valid)[unsupported]] = np.nan
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
        labels=label_result,
        range_limit=limits.reshape(shape),
        colour_evidence=tuple(evidence),
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
    slot_runs: list[tuple[Axis, int, tuple[int, int, int, int], OCRRun]] = []
    skipped: list[dict[str, object]] = []
    oriented_records: list[dict[str, object]] = []
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

            if axis == "columns" and (
                source.recognition == "lines" or source.text_rotation_degrees not in (0, 90, -90)
            ):
                from .oriented_labels import recognise_lines

                assigned, records, attempts = recognise_lines(image, config, source)
                axes.append(assigned)
                oriented_records.extend(records)
                skipped.extend(attempts)
                continue
            if source.recognition == "slots":
                # Counts/regions are already explicit. One fixed single-line
                # pass per slot, no dictionaries or confidence-based retries.
                slots = axis_slots(config, axis)
                if len(slots) > 128:
                    raise ValueError("slot OCR call cap exceeded (128 per axis)")
                coordinate = 1 if axis == "rows" else 0
                tokens: list[OCRToken] = []
                candidates: list[LabelCandidate] = []
                pixel_budget = 16000000
                deadline = time.monotonic() + 60
                calls_remaining = 128
                for slot in slots:
                    box_list = list(source.bbox)
                    box_list[coordinate] = max(box_list[coordinate], math.floor(slot.start))
                    box_list[coordinate + 2] = min(box_list[coordinate + 2], math.ceil(slot.end))
                    box = (box_list[0], box_list[1], box_list[2], box_list[3])
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    if source.trim_whitespace:
                        # A configured light-background field can contain a short
                        # label beside much longer neighbors. Crop observed ink,
                        # preserving original-coordinate geometry before rotation.
                        gray = np.asarray(Image.fromarray(image).crop(box).convert("L"))
                        ink_y, ink_x = np.where(gray < 160)
                        if ink_x.size:
                            left, top, right, bottom = box
                            box = (
                                max(left, left + int(np.min(ink_x)) - 1),
                                max(top, top + int(np.min(ink_y)) - 1),
                                min(right, left + int(np.max(ink_x)) + 2),
                                min(bottom, top + int(np.max(ink_y)) + 2),
                            )
                    pad = source.crop_padding_pixels
                    box = (
                        max(0, box[0] - pad),
                        max(0, box[1] - pad),
                        min(image.shape[1], box[2] + pad + source.crop_right_padding_pixels),
                        min(image.shape[0], box[3] + pad),
                    )
                    transform = CropTransform(
                        box,
                        config.labels.ocr.upscale_factor,
                        source.text_rotation_degrees,
                        source.white_padding_pixels,
                    )
                    original_crop: Image.Image | None = None
                    slot_candidates: list[LabelCandidate] = []
                    # Fixed transforms share the original call/pixel/deadline budgets.
                    variants = ("original", *config.labels.ocr.retry_variants)
                    for variant in variants:
                        pixels = math.prod(transform.crop_size)
                        remaining_seconds = deadline - time.monotonic()
                        if pixels > pixel_budget or remaining_seconds <= 0 or calls_remaining <= 0:
                            skipped.append(
                                {
                                    "axis": axis,
                                    "slot": slot.index,
                                    "variant": variant,
                                    "reason": "shared_budget_exhausted",
                                }
                            )
                            continue
                        if original_crop is None:
                            original_crop = transform.prepare(Image.fromarray(image))
                        pixel_budget -= pixels
                        calls_remaining -= 1
                        crop = (
                            ImageOps.autocontrast(original_crop)
                            if variant == "contrast"
                            else original_crop.point(lambda p: 255 if p >= 160 else 0)
                            if variant == "threshold"
                            else original_crop
                        )
                        try:
                            run = recognize(
                                crop,
                                config.labels.ocr,
                                mode="text_line",
                                timeout_seconds=min(30, remaining_seconds),
                            )
                        except ValueError as error:
                            if variant == "original" and not config.labels.ocr.retry_variants:
                                raise
                            skipped.append(
                                {
                                    "axis": axis,
                                    "slot": slot.index,
                                    "variant": variant,
                                    "reason": str(error),
                                }
                            )
                            continue
                        run = replace(
                            run,
                            preprocessing=variant,
                            white_padding_pixels=source.white_padding_pixels,
                            tokens=tuple(
                                replace(token, bbox=transform.to_crop(box)) for token in run.tokens
                            )
                            if run.geometry_kind == "source_field"
                            else run.tokens,
                        )
                        slot_runs.append((axis, slot.index, box, run))
                        offset = len(tokens)
                        observed = [
                            replace(c, token_indices=tuple(i + offset for i in c.token_indices))
                            for c in group_tokens(
                                run.tokens, config, axis, transform=transform, slot_index=slot.index
                            )
                        ]
                        tokens.extend(transform_tokens(run.tokens, transform))
                        if not config.labels.ocr.retry_variants:
                            slot_candidates.extend(observed)
                            continue
                        # Merge only exact same-slot readings; retain all raw runs/tokens.
                        # A disagreement remains a separate candidate requiring review.
                        for candidate in observed:
                            same = next(
                                (
                                    i
                                    for i, c in enumerate(slot_candidates)
                                    if c.text == candidate.text
                                ),
                                None,
                            )
                            if same is None:
                                slot_candidates.append(candidate)
                            else:
                                previous = slot_candidates[same]
                                a, b = previous.bbox, candidate.bbox
                                slot_candidates[same] = replace(
                                    previous,
                                    bbox=(
                                        min(a[0], b[0]),
                                        min(a[1], b[1]),
                                        max(a[2], b[2]),
                                        max(a[3], b[3]),
                                    ),
                                    confidence=min(previous.confidence, candidate.confidence),
                                    token_indices=previous.token_indices + candidate.token_indices,
                                )
                    candidates.extend(slot_candidates)
                axes.append(assign_candidates(candidates, config, axis, tokens=tokens))
                continue
            crop, transform = crop_label_region(Image.fromarray(image), config, axis)
            run = recognize(crop, config.labels.ocr)
            runs.append((axis, run))
            axes.append(
                assign_candidates(
                    group_tokens(run.tokens, config, axis, transform=transform),
                    config,
                    axis,
                    tokens=transform_tokens(run.tokens, transform),
                )
            )
    reviewed_axes = []
    for result_axis in axes:
        result_axis = apply_character_policy(result_axis, config)
        source = getattr(config.labels, result_axis.axis) if config.labels is not None else None
        if isinstance(source, OCRLabelSource) and source.require_review:
            result_axis = replace(
                result_axis,
                slots=tuple(
                    replace(
                        slot,
                        final_text=None,
                        status="unreviewed",
                        issues=(*slot.issues, "configured_identifier_review_required"),
                    )
                    if slot.status == "accepted"
                    else slot
                    for slot in result_axis.slots
                ),
            )
        reviewed_axes.append(result_axis)
    return LabelExtractionResult(
        config,
        tuple(reviewed_axes),
        tuple(runs),
        slot_ocr_runs=tuple(slot_runs),
        skipped_ocr_attempts=tuple(skipped),
        oriented_ocr_records=tuple(oriented_records),
    )
