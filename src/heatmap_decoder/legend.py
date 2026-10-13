"""Bounded upright numeric tick recognition, linear fitting and explicit review.

No tick OCR verifies the biological quantity or enables automatic export. Raw
recognition, stroke geometry, parsing and scale fit remain separate evidence.
"""

import math
import re
import time
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Literal

import numpy as np
from PIL import Image

from .calibration import fit_linear_scale
from .labels import CropTransform, group_unassigned_tokens, transform_tokens
from .models import (
    BoundingBox,
    LegendConfig,
    LegendOCRRecord,
    LegendResult,
    LegendTick,
    OCRRun,
    RGBArray,
    ScaleAnchor,
)
from .sampling import validate_rgb

LEGEND_POLICY_VERSION = "m16-reviewed-model.1"
Side = Literal["left", "right", "above", "below"]


def legend_policy_snapshot() -> dict[str, object]:
    return {
        "version": LEGEND_POLICY_VERSION,
        "automatic_scale_enabled": False,
        "required_ticks_for_development_eligibility": 3,
        "stroke_dark_rgb_threshold": 110,
        "stroke_zone_pixels": 14,
        "maximum_stroke_width_pixels": 4,
        "maximum_numeric_text_length": 256,
        "maximum_localised_text_tokens": 256,
        "ocr_modes": ["--oem 1 --psm 11", "--oem 1 --psm 7"],
        "maximum_fit_value_residual_fraction": 0.01,
        "maximum_fit_position_residual_pixels": 2.0,
        "maximum_fit_condition_number": 1000,
        "reason": "OCR linear scales require review until the independent M12 gate",
    }


@dataclass(frozen=True)
class ParsedNumber:
    raw_text: str
    normalised_text: str
    value: float | None
    actions: tuple[str, ...]
    reason: str | None = None


@dataclass
class _OCRBudget:
    pixels_remaining: int
    targeted_calls_remaining: int
    deadline: float


def parse_numeric_tick(raw: str, decimal_separator: str = ".") -> ParsedNumber:
    """Explicit grammar: signed decimal/e notation or 10^signed_integer.

    No identifier substitutions, minus guessing, thousands separators or
    superscript reconstruction. Spaces after an observed sign are grammatical.
    """
    if decimal_separator not in {".", ","}:
        raise ValueError("decimal separator must be '.' or ','")
    if len(raw) > 256:
        return ParsedNumber(raw, raw, None, (), "numeric_tick_length_exceeded")
    text = raw.strip()
    actions = []
    if text != raw:
        actions.append("trim_outer_whitespace")
    if "−" in text:
        text = text.replace("−", "-")
        actions.append("unicode_minus_to_ascii")
    if re.match(r"^[+-]\s+\d", text):
        text = re.sub(r"^([+-])\s+", r"\1", text)
        actions.append("join_observed_sign_whitespace")
    if "," in text:
        if decimal_separator != "," or "." in text or text.count(",") != 1:
            return ParsedNumber(raw, text, None, tuple(actions), "ambiguous_locale_separator")
        text = text.replace(",", ".")
        actions.append("explicit_decimal_comma")
    if re.fullmatch(r"10\^[+-]?\d+", text):
        try:
            value = float(10.0 ** int(text[3:]))
        except (OverflowError, ValueError):
            return ParsedNumber(raw, text, None, tuple(actions), "nonfinite_numeric_tick")
        if value == 0:
            return ParsedNumber(raw, text, None, tuple(actions), "numeric_tick_underflow")
        actions.append("explicit_power_of_ten_grammar")
    elif re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
        value = float(text)
        if value == 0 and Decimal(text) != 0:
            return ParsedNumber(raw, text, None, tuple(actions), "numeric_tick_underflow")
    else:
        return ParsedNumber(raw, text, None, tuple(actions), "unparsed_numeric_tick")
    if not math.isfinite(value):
        return ParsedNumber(raw, text, None, tuple(actions), "nonfinite_numeric_tick")
    return ParsedNumber(raw, text, value, tuple(actions))


def validate_legend_geometry(image: RGBArray, config: LegendConfig) -> None:
    validate_rgb(image)
    x, y, r, s = config.colourbar_bbox
    if not (0 <= x < r <= image.shape[1] and 0 <= y < s <= image.shape[0]):
        raise ValueError("legend colourbar bbox must lie inside the image")
    length = s - y if config.colourbar_orientation == "vertical" else r - x
    if length < 2:
        raise ValueError("legend colourbar requires at least two axial samples")
    if image.shape[0] * image.shape[1] > 16000000:
        raise ValueError("legend image pixel cap exceeded; crop the figure")
    if image.shape[2] == 4 and np.any(image[:, :, 3] != 255):
        raise ValueError("legend recognition requires an opaque image")


def search_regions(image: RGBArray, config: LegendConfig) -> tuple[tuple[Side, BoundingBox], ...]:
    x, y, r, s = config.colourbar_bbox
    h, w = image.shape[:2]
    distance = (
        config.legend.search_pixels
        if config.colourbar_orientation == "vertical"
        else config.legend.horizontal_search_pixels
    )
    pad = min(64, config.legend.search_pixels)
    choices: tuple[tuple[Side, BoundingBox], ...] = (
        (
            ("left", (max(0, x - distance), max(0, y - pad), x, min(h, s + pad))),
            ("right", (r, max(0, y - pad), min(w, r + distance), min(h, s + pad))),
        )
        if config.colourbar_orientation == "vertical"
        else (
            ("above", (max(0, x - pad), max(0, y - distance), min(w, r + pad), y)),
            ("below", (max(0, x - pad), s, min(w, r + pad), min(h, s + distance))),
        )
    )
    return tuple((side, box) for side, box in choices if box[2] > box[0] and box[3] > box[1])


def stroke_ticks(
    image: RGBArray, config: LegendConfig, side: Side, region: BoundingBox
) -> tuple[LegendTick, ...]:
    """Measure short dark strokes close to the bar, independent of text centres."""
    x, y, r, s = region
    pixels = image[y:s, x:r, :3]
    vertical = config.colourbar_orientation == "vertical"
    if not vertical:
        pixels = pixels.transpose(1, 0, 2)
    width = min(14, pixels.shape[1])
    zone = pixels[:, -width:] if side in {"left", "above"} else pixels[:, :width]
    dark = np.max(zone, axis=2) < 110
    # A connected axial border can merge several genuine strokes into one run.
    # Ignore only near-edge transverse rows dark over >=80% of the physical bar;
    # keep the original image and every independently measured stroke box.
    coordinate = 1 if vertical else 0
    bar_start = config.colourbar_bbox[coordinate]
    bar_last = config.colourbar_bbox[coordinate + 2]
    origin = y if vertical else x
    axial = slice(max(0, bar_start - origin), min(len(dark), bar_last - origin))
    near_edge = (
        range(max(0, width - 3), width) if side in {"left", "above"} else range(min(3, width))
    )
    for transverse in near_edge:
        if np.mean(dark[axial, transverse]) >= 0.8:
            dark[:, transverse] = False
    occupied = np.flatnonzero(np.sum(dark, axis=1) >= 3)
    groups = np.split(occupied, np.flatnonzero(np.diff(occupied) > 1) + 1) if len(occupied) else []
    results: list[LegendTick] = []
    for group in groups:
        if len(group) > 4:
            continue
        xs = np.flatnonzero(np.any(dark[group], axis=0))
        # Strokes must begin near the bar. Nearby text is not itself a tick.
        gap = width - 1 - int(xs[-1]) if side in {"left", "above"} else int(xs[0])
        if gap > 6:
            continue
        cross_origin = (
            (r - width if vertical else s - width)
            if side in {"left", "above"}
            else (x if vertical else y)
        )
        along_origin = y if vertical else x
        a, b = along_origin + int(group[0]), along_origin + int(group[-1]) + 1
        c, d = cross_origin + int(xs[0]), cross_origin + int(xs[-1]) + 1
        bbox = (c, a, d, b) if vertical else (a, c, b, d)
        results.append(
            LegendTick(
                index=len(results),
                side=side,
                position=(a + b - 1) / 2,
                position_source="stroke",
                stroke_bbox=bbox,
            )
        )
        if len(results) > config.legend.max_ticks:
            break  # One extra observation records that the supported cap was exceeded.
    return tuple(results)


def _ocr_crop(
    image: RGBArray,
    config: LegendConfig,
    side: Side,
    box: BoundingBox,
    purpose: Literal["localisation", "numeric_line"],
    budget: _OCRBudget,
    strokes: tuple[LegendTick, ...] = (),
    *,
    white_padding_pixels: int | None = None,
) -> LegendOCRRecord:
    from .ocr import recognize

    factor = config.legend.ocr.upscale_factor
    white = (
        (
            config.legend.white_padding_pixels
            if white_padding_pixels is None
            else white_padding_pixels
        )
        if purpose == "numeric_line"
        else 0
    )
    transform = CropTransform(box, factor, white_padding_pixels=white)
    pixels = math.prod(transform.crop_size)
    if time.monotonic() > budget.deadline:
        raise ValueError("legend OCR runtime cap exceeded")
    if purpose == "numeric_line":
        if budget.targeted_calls_remaining <= 0:
            raise ValueError("legend targeted OCR call cap exceeded")
        budget.targeted_calls_remaining -= 1
    if pixels > budget.pixels_remaining:
        raise ValueError("legend OCR pixel cap exceeded; reduce search strip/upscale factor")
    budget.pixels_remaining -= pixels
    crop = Image.fromarray(image).crop(box).convert("L")
    # A real tick stroke can look like a minus sign. Remove only independently
    # measured stroke pixels from the OCR copy; never alter observed glyphs.
    from PIL import ImageDraw

    masks = tuple(t.stroke_bbox for t in strokes if t.stroke_bbox is not None)
    painter = ImageDraw.Draw(crop)
    for x, y, r, s in masks:
        painter.rectangle((x - box[0], y - box[1], r - box[0] - 1, s - box[1] - 1), fill=255)
    if white:
        from PIL import ImageOps

        crop = ImageOps.expand(crop, border=white, fill=255)
    run = recognize(
        crop.resize(transform.crop_size, Image.Resampling.BICUBIC),
        config.legend.ocr,
        mode=purpose,
        timeout_seconds=min(30, max(0.001, budget.deadline - time.monotonic())),
    )
    run = replace(run, white_padding_pixels=white)
    return LegendOCRRecord(
        side=side, crop_bbox=box, purpose=purpose, run=run, masked_stroke_boxes=masks
    )


def associate_tokens(
    config: LegendConfig,
    side: Side,
    strokes: tuple[LegendTick, ...],
    run: OCRRun,
    region: BoundingBox,
) -> tuple[LegendTick, ...]:
    """One-to-one centre geometry; ambiguous/unparsed candidates remain in evidence."""
    tokens = transform_tokens(run.tokens, CropTransform(region, config.legend.ocr.upscale_factor))
    coordinate = 1 if config.colourbar_orientation == "vertical" else 0
    # Stroke midpoints define spatial bands before grouping. Nearby horizontal
    # numbers must not become one line merely because their word gap is small.
    # A token crossing a band remains intact and ambiguous; never split glyphs.
    positions = sorted(t.position for t in strokes)
    bands = [-math.inf] + [(a + b) / 2 for a, b in zip(positions, positions[1:])] + [math.inf]
    buckets: dict[int, list[int]] = {}
    for i, token in enumerate(tokens):
        if not token.text.strip() or token.confidence < 0:
            continue
        matches = [
            j
            for j in range(len(bands) - 1)
            if bands[j] <= token.bbox[coordinate] and token.bbox[coordinate + 2] <= bands[j + 1]
        ]
        bucket = matches[0] if len(matches) == 1 else -i - 1
        buckets.setdefault(bucket, []).append(i)
    candidates = tuple(
        replace(c, token_indices=tuple(indices[j] for j in c.token_indices))
        for indices in buckets.values()
        for c in group_unassigned_tokens([tokens[j] for j in indices])
    )
    start, last = config.colourbar_bbox[coordinate], config.colourbar_bbox[coordinate + 2] - 1
    candidate_matches = []
    for c in candidates:
        centre = (c.bbox[coordinate] + c.bbox[coordinate + 2]) / 2
        candidate_matches.append(
            [
                i
                for i, t in enumerate(strokes)
                if abs(centre - t.position) <= config.legend.max_axis_offset_pixels
            ]
        )
    results = []
    used = set()
    for index, tick in enumerate(strokes):
        matches = [i for i, choices in enumerate(candidate_matches) if index in choices]
        if len(matches) != 1 or len(candidate_matches[matches[0]]) != 1:
            reason = "missing_tick_text" if not matches else "ambiguous_tick_text_association"
            results.append(
                tick.model_copy(
                    update={
                        "status": "unresolved" if not matches else "ambiguous",
                        "reasons": (reason,),
                    }
                )
            )
            used.update(matches)
            continue
        i = matches[0]
        used.add(i)
        candidate = candidates[i]
        parsed = parse_numeric_tick(candidate.text, config.legend.decimal_separator)
        reasons = [parsed.reason] if parsed.reason else []
        if (
            candidate.bbox[0] <= region[0] + 1
            or candidate.bbox[1] <= region[1] + 1
            or candidate.bbox[2] >= region[2] - 1
            or candidate.bbox[3] >= region[3] - 1
        ):
            reasons.append("numeric_text_touches_search_edge")
        if candidate.confidence < config.legend.ocr.min_confidence:
            reasons.append("low_numeric_ocr_confidence")
        results.append(
            tick.model_copy(
                update={
                    "text_bbox": candidate.bbox,
                    "raw_text": parsed.raw_text,
                    "normalised_text": parsed.normalised_text,
                    "normalisation_actions": parsed.actions,
                    "parsed_value": parsed.value,
                    "confidence": candidate.confidence,
                    "token_indices": candidate.token_indices,
                    "status": "unresolved" if reasons else "suggested",
                    "reasons": tuple(reasons),
                }
            )
        )
    for i, c in enumerate(candidates):
        if i in used:
            continue
        centre = (c.bbox[coordinate] + c.bbox[coordinate + 2]) / 2
        parsed = parse_numeric_tick(c.text, config.legend.decimal_separator)
        reasons = ["unassociated_text_requires_review"]
        if not start <= centre <= last:
            reasons.append("outside_bar_span_title_or_caption")
        if parsed.reason:
            reasons.append(parsed.reason)
        results.append(
            LegendTick(
                index=len(results),
                side=side,
                position=centre,
                position_source="text_centre",
                text_bbox=c.bbox,
                raw_text=c.text,
                normalised_text=parsed.normalised_text,
                normalisation_actions=parsed.actions,
                parsed_value=parsed.value,
                confidence=c.confidence,
                token_indices=c.token_indices,
                status="excluded"
                if centre < start - config.legend.max_axis_offset_pixels
                or centre > last + config.legend.max_axis_offset_pixels
                else "unresolved",
                reasons=tuple(reasons),
            )
        )
    return tuple(results)


def evaluate_legend(
    config: LegendConfig,
    ticks: tuple[LegendTick, ...],
    records: tuple[LegendOCRRecord, ...] = (),
    warnings: tuple[str, ...] = (),
) -> LegendResult:
    """All plausible ticks participate; no confidence-based pruning or outlier removal."""
    reasons = list(warnings)
    fit = None
    plausible = tuple(t for t in ticks if t.status != "excluded")
    if any(t.status != "suggested" or t.parsed_value is None for t in plausible):
        reasons.append("unresolved_or_conflicting_plausible_ticks")
    anchors = tuple(
        ScaleAnchor(position=t.position, value=t.parsed_value, origin="ocr", status="suggested")
        for t in plausible
        if t.parsed_value is not None
    )
    if len(anchors) < 2:
        reasons.append("fewer_than_two_numeric_ticks")
    else:
        try:
            fit = fit_linear_scale(
                anchors,
                config.colourbar_bbox,
                config.colourbar_orientation,
                endpoint_tolerance_pixels=config.legend.endpoint_tolerance_pixels,
                enforce_residuals=False,
            )
        except ValueError as error:
            reasons.append(str(error))
    if fit is not None and not fit.within_residual_limits:
        reasons.append("anchors_contradict_linear_scale")
    eligible = not reasons and fit is not None and fit.tests_linearity and fit.endpoint_coverage
    if fit is not None:
        if not fit.tests_linearity:
            reasons.append("two_ticks_cannot_test_linearity")
        if not fit.endpoint_coverage:
            reasons.append("interior_ticks_require_extrapolation_review")
    if any(t.position_source != "stroke" for t in plausible):
        reasons.append("text_centres_are_weaker_position_evidence")
        eligible = False
    reasons.append("M12_scale_gate_closed")
    unresolved = (
        fit is None
        or not fit.within_residual_limits
        or any(t.status != "suggested" for t in plausible)
        or bool(warnings)
    )
    return LegendResult(
        config=config,
        ticks=ticks,
        ocr_records=records,
        fit=fit,
        status="unresolved" if unresolved else "review_required",
        reasons=tuple(dict.fromkeys(reasons)),
        development_eligible=eligible,
    )


def inspect_legend(image: RGBArray, config: LegendConfig) -> LegendResult:
    validate_legend_geometry(image, config)
    ticks: list[LegendTick] = []
    records: list[LegendOCRRecord] = []
    warnings: list[str] = []
    budget = _OCRBudget(
        config.legend.max_ocr_pixels,
        config.legend.max_ticks,
        time.monotonic() + config.legend.max_ocr_seconds,
    )
    for side, region in search_regions(image, config):
        strokes = stroke_ticks(image, config, side, region)
        fields: dict[float, BoundingBox] = {}
        field_evidence = None
        if config.legend.align_tick_fields:
            from .ocr_fields import aligned_tick_fields

            fields, field_evidence = aligned_tick_fields(image, config, side, region, strokes)
        try:
            record = _ocr_crop(image, config, side, region, "localisation", budget, strokes)
            records.append(record)
            if sum(bool(t.text.strip()) and t.confidence >= 0 for t in record.run.tokens) > 256:
                warnings.append("legend_token_cap_exceeded")
                ticks.extend(strokes)
                continue
            proposals = tuple(
                t.model_copy(update={"localisation_record_index": len(records) - 1})
                for t in associate_tokens(config, side, strokes, record.run, region)
            )
            # Exactly one deterministic targeted pass per unambiguous text line.
            revised = []
            for tick in proposals:
                if tick.position_source == "stroke" and tick.status != "ambiguous":
                    b = tick.text_bbox
                    if tick.position in fields:
                        box = fields[tick.position]
                    elif b is not None:
                        box = (
                            max(region[0], math.floor(b[0]) - 3),
                            max(region[1], math.floor(b[1]) - 3),
                            min(region[2], math.ceil(b[2]) + 3),
                            min(region[3], math.ceil(b[3]) + 3),
                        )
                    elif config.colourbar_orientation == "vertical":
                        box = (
                            region[0] if side == "left" else region[0] + 14,
                            max(region[1], math.floor(tick.position) - 14),
                            region[2] - 14 if side == "left" else region[2],
                            min(region[3], math.ceil(tick.position) + 15),
                        )
                    else:
                        others = [
                            abs(tick.position - t.position)
                            for t in strokes
                            if t.position != tick.position
                        ]
                        pad = min(48, math.floor(min(others, default=96) / 2))
                        box = (
                            max(region[0], math.floor(tick.position) - pad),
                            region[1] if side == "above" else region[1] + 14,
                            min(region[2], math.ceil(tick.position) + pad + 1),
                            region[3] - 14 if side == "above" else region[3],
                        )
                    tick = tick.model_copy(
                        update={"ocr_bbox": box, "field_evidence": field_evidence}
                    )
                    try:
                        numeric = _ocr_crop(image, config, side, box, "numeric_line", budget)
                    except (ValueError, OSError) as error:
                        warnings.append(f"{side}: numeric OCR failed: {error}")
                        revised.append(
                            tick.model_copy(
                                update={
                                    "status": "unresolved",
                                    "reasons": tick.reasons + ("numeric_ocr_failed",),
                                }
                            )
                        )
                        continue
                    records.append(numeric)
                    member_indices = tuple(
                        i
                        for i, t in enumerate(numeric.run.tokens)
                        if t.text.strip() and t.confidence >= 0
                    )
                    words = [numeric.run.tokens[i] for i in member_indices]
                    original_words = transform_tokens(
                        words,
                        CropTransform(
                            box,
                            config.legend.ocr.upscale_factor,
                            white_padding_pixels=numeric.run.white_padding_pixels,
                        ),
                    )
                    text_box = (
                        (
                            min(t.bbox[0] for t in original_words),
                            min(t.bbox[1] for t in original_words),
                            max(t.bbox[2] for t in original_words),
                            max(t.bbox[3] for t in original_words),
                        )
                        if original_words
                        else tick.text_bbox
                    )
                    text = " ".join(t.text.strip() for t in words)
                    parsed = parse_numeric_tick(text, config.legend.decimal_separator)
                    confidence = min((t.confidence for t in words), default=0)
                    issues = [parsed.reason] if parsed.reason else []
                    issues += [
                        reason
                        for reason in tick.reasons
                        if reason == "numeric_text_touches_search_edge"
                    ]
                    coordinate = 1 if config.colourbar_orientation == "vertical" else 0
                    if text_box is not None and (
                        not (
                            text_box[coordinate] - config.legend.max_axis_offset_pixels
                            <= tick.position
                            <= text_box[coordinate + 2] + config.legend.max_axis_offset_pixels
                        )
                        if tick.position in fields
                        else abs(
                            (text_box[coordinate] + text_box[coordinate + 2]) / 2 - tick.position
                        )
                        > config.legend.max_axis_offset_pixels
                    ):
                        issues.append("numeric_line_position_conflict")
                    if config.legend.align_tick_fields and tick.position not in fields:
                        issues.append("shared_tick_field_unresolved")
                    if confidence < config.legend.ocr.min_confidence:
                        issues.append("low_numeric_ocr_confidence")
                    tick = tick.model_copy(
                        update={
                            "numeric_record_index": len(records) - 1,
                            "numeric_token_indices": member_indices,
                            "text_bbox": text_box,
                            "raw_text": text,
                            "normalised_text": parsed.normalised_text,
                            "normalisation_actions": parsed.actions,
                            "parsed_value": parsed.value,
                            "confidence": confidence,
                            "status": "unresolved" if issues else "suggested",
                            "reasons": tuple(issues),
                        }
                    )
                if tick.ocr_bbox is not None and tick.numeric_record_index is not None:
                    alternatives: list[dict[str, Any]] = [
                        {
                            "record_index": tick.numeric_record_index,
                            "raw_text": tick.raw_text,
                            "parsed_value": tick.parsed_value,
                            "confidence": tick.confidence,
                            "text_bbox": tick.text_bbox,
                            "white_padding_pixels": records[
                                tick.numeric_record_index
                            ].run.white_padding_pixels,
                            "normalisation_actions": tick.normalisation_actions,
                        }
                    ]
                    for white in config.legend.retry_white_padding_pixels:
                        if white == config.legend.white_padding_pixels:
                            continue
                        try:
                            retry = _ocr_crop(
                                image,
                                config,
                                side,
                                tick.ocr_bbox,
                                "numeric_line",
                                budget,
                                white_padding_pixels=white,
                            )
                        except (ValueError, OSError) as error:
                            warnings.append(f"{side}: numeric padding retry failed: {error}")
                            alternatives.append(
                                {"white_padding_pixels": white, "skipped": str(error)}
                            )
                            continue
                        records.append(retry)
                        words = [
                            t for t in retry.run.tokens if t.text.strip() and t.confidence >= 0
                        ]
                        original = transform_tokens(
                            words,
                            CropTransform(
                                tick.ocr_bbox,
                                config.legend.ocr.upscale_factor,
                                white_padding_pixels=retry.run.white_padding_pixels,
                            ),
                        )
                        text = " ".join(w.text.strip() for w in words)
                        parsed_retry = parse_numeric_tick(text, config.legend.decimal_separator)
                        alternatives.append(
                            {
                                "record_index": len(records) - 1,
                                "token_indices": tuple(
                                    i
                                    for i, token in enumerate(retry.run.tokens)
                                    if token.text.strip() and token.confidence >= 0
                                ),
                                "raw_text": text,
                                "parsed_value": parsed_retry.value,
                                "confidence": min((w.confidence for w in words), default=0),
                                "text_bbox": (
                                    min(w.bbox[0] for w in original),
                                    min(w.bbox[1] for w in original),
                                    max(w.bbox[2] for w in original),
                                    max(w.bbox[3] for w in original),
                                )
                                if original
                                else None,
                                "white_padding_pixels": white,
                                "normalisation_actions": parsed_retry.actions,
                            }
                        )
                    reasons = list(tick.reasons)
                    distinct = {a["raw_text"] for a in alternatives if a.get("raw_text")}
                    numeric_values = {
                        a["parsed_value"] for a in alternatives if a.get("parsed_value") is not None
                    }
                    updates: dict[str, Any] = {"numeric_alternatives": tuple(alternatives)}
                    if len(distinct) > 1:
                        reasons.append("conflicting_numeric_readings")
                        updates["status"] = "ambiguous"
                    if len(numeric_values) > 1:
                        updates["parsed_value"] = None
                        reasons.append("contradictory_numeric_values")
                    elif tick.parsed_value is None and len(numeric_values) == 1:
                        recovered = next(
                            a for a in alternatives if a.get("parsed_value") is not None
                        )
                        updates.update(
                            raw_text=recovered["raw_text"],
                            parsed_value=recovered["parsed_value"],
                            confidence=recovered["confidence"],
                            text_bbox=recovered["text_bbox"],
                            numeric_record_index=recovered["record_index"],
                            numeric_token_indices=recovered["token_indices"],
                            normalised_text=parse_numeric_tick(
                                recovered["raw_text"], config.legend.decimal_separator
                            ).normalised_text,
                            normalisation_actions=recovered["normalisation_actions"],
                        )
                        if len(distinct) <= 1:
                            updates["status"] = "unresolved"
                        reasons.append("padding_retry_proposal_requires_review")
                    updates["reasons"] = tuple(dict.fromkeys(reasons))
                    tick = tick.model_copy(update=updates)
                revised.append(tick)
            ticks.extend(revised)
        except (ValueError, OSError) as error:
            warnings.append(f"{side}: OCR unavailable/failed: {error}")
            ticks.extend(strokes)
    if time.monotonic() > budget.deadline:
        warnings.append("legend_ocr_runtime_cap_exceeded")
    if len(ticks) > config.legend.max_ticks:
        warnings.append("legend_tick_cap_exceeded")
        ticks = ticks[: config.legend.max_ticks]
    ticks = [
        t.model_copy(update={"index": i})
        for i, t in enumerate(sorted(ticks, key=lambda t: (t.position, t.side, t.position_source)))
    ]
    x, y, r, s = config.colourbar_bbox
    if x == 0 or y == 0 or r == image.shape[1] or s == image.shape[0]:
        warnings.append("colourbar_touches_image_edge_possible_clipping")
    return evaluate_legend(config, tuple(ticks), tuple(records), tuple(warnings))
