"""Composable selective decisions. Explicit runtime opt-in gates scoped experimental export."""

import math
import time
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .acceptance import decision
from .calibration import calibration_evidence, colour_quality_summary, fit_linear_scale
from .detection import detect
from .detection_policy import ACCEPTANCE_THRESHOLDS as A
from .extract import extract, extract_labels
from .legend import inspect_legend
from .models import (
    AcceptanceDecision,
    DetectionHints,
    DetectionResult,
    DetectorOptions,
    ExtractionConfig,
    LegendConfig,
    LegendResult,
    RGBArray,
    ScaleAnchor,
    ScaleConfig,
)

AUTOMATION_POLICY_VERSION = "capability-selective.3"

# Frozen M19 clean/negative gates qualify sharp measured grids and linear ticks.
# Broad degradation coverage failed and remains closed. Qualification never
# substitutes for the per-image checks; see benchmarks/m19/validation-final.json.
M19_QUALIFICATION = {
    "inferred_counts": False,
    "sharp_measured_counts": True,
    "linear_ocr_scale": True,
}


def sharp_measured_count_scope(decisions: list[AcceptanceDecision]) -> bool:
    """Registered sharp common-pitch support, not a claim about all degradations."""
    from .blocks import BLOCK_POLICY

    for d in decisions:
        if d.requirement not in {"rows", "columns"} or d.evidence.get("origin") == "manual":
            continue
        selected = d.evidence.get("observations", {}).get("selected")
        if d.status != "accepted" or not selected:
            return False
        if selected["maximum_transition_width"] > BLOCK_POLICY["maximum_block_transition_width"]:
            return False
    return True


class AutomationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hints: DetectionHints
    numeric_only: Annotated[bool, Field(strict=True)]
    scale: ScaleConfig | None = None

    @model_validator(mode="after")
    def consistent_request(self) -> "AutomationRequest":
        if "colourbar" not in self.hints.requested and self.hints.colourbar_bbox is None:
            raise ValueError("automation requires a colourbar request or supplied bar")
        identifiers = (
            self.hints.labels is not None
            and (self.hints.labels.rows is not None or self.hints.labels.columns is not None)
        ) or any(k.startswith("labels_") for k in self.hints.requested)
        if self.numeric_only and identifiers:
            raise ValueError("numeric_only conflicts with requested identifiers")
        if not self.numeric_only and not identifiers:
            raise ValueError("request identifiers or explicitly select numeric_only")
        if self.scale is not None and self.scale.source != "manual_endpoints":
            if (
                self.hints.colourbar_start_value is not None
                or self.hints.colourbar_end_value is not None
            ):
                raise ValueError("tick scales conflict with supplied endpoint values")
        return self


class AutomationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_version: str = AUTOMATION_POLICY_VERSION
    detection: DetectionResult
    legend: LegendResult | None = None
    decisions: tuple[AcceptanceDecision, ...]
    complete: AcceptanceDecision
    automatic_export_allowed: bool
    release_reasons: tuple[str, ...]
    diagnostic_config: ExtractionConfig | None = None
    legend_diagnostics: tuple[LegendResult, ...] = ()
    experimental_export_requested: bool = False
    export_maturity: str = "review_only"
    numeric_complete: AcceptanceDecision
    label_geometry_complete: AcceptanceDecision
    identifiers_complete: AcceptanceDecision
    numeric_retention_allowed: bool = False
    numeric_state: str = "unresolved"
    identifier_state: str = "not_requested"
    ocr_anchors: tuple[ScaleAnchor, ...] = ()
    label_evidence: tuple[dict[str, object], ...] = ()
    qualification: dict[str, bool] = Field(default_factory=lambda: dict(M19_QUALIFICATION))


def automate(
    image: RGBArray,
    request: AutomationRequest,
    *,
    experimental_export: bool = False,
) -> AutomationResult:
    """Deployed classifier used by CLI and benchmark; receives no evaluation truth.

    The concrete config records resolved geometry and counts. Default operation
    is diagnostic; explicit experimental export requires every per-image decision
    and qualified evidence scope. OCR identity requires review.
    """
    hints = request.hints
    detected = detect(image, hints)
    assert detected.acceptance is not None
    decisions = [
        d
        for d in detected.acceptance.decisions
        if d.requirement not in {"scale", "numeric_quality"}
    ]
    by_kind = {d.requirement: d for d in decisions}
    candidates = {c.id: c for c in detected.candidates}
    bar_check = by_kind.get("colourbar")
    bar = hints.colourbar_bbox
    orientation = hints.colourbar_orientation
    if bar is None and bar_check is not None and bar_check.status == "accepted":
        candidate = candidates[bar_check.candidate_ids[0]]
        bar, orientation = candidate.bbox, candidate.orientation
    if bar is not None and orientation is None:
        orientation = "vertical" if bar[3] - bar[1] >= bar[2] - bar[0] else "horizontal"
    start, end = hints.colourbar_start_value, hints.colourbar_end_value
    legend = None
    legend_diagnostics: list[LegendResult] = []
    scale = request.scale
    ocr_anchors: tuple[ScaleAnchor, ...] = ()
    if scale is not None and scale.source == "ocr_ticks" and bar is None:
        # Inspect plausible bars even if geometry/competition remains unresolved.
        # These runs never replace required per-image decisions or choose a target.
        pixel_budget = 12000000
        deadline = time.monotonic() + 60
        for candidate in (c for c in detected.candidates if c.kind == "colourbar"):
            if len(legend_diagnostics) >= 3 or time.monotonic() >= deadline or pixel_budget < 1000:
                break
            assert candidate.orientation is not None
            options = scale.legend.model_copy(
                update={
                    "max_ocr_pixels": min(scale.legend.max_ocr_pixels, pixel_budget),
                    "max_ocr_seconds": max(
                        1, min(scale.legend.max_ocr_seconds, int(deadline - time.monotonic()))
                    ),
                }
            )
            diagnostic = inspect_legend(
                image,
                LegendConfig(
                    colourbar_bbox=candidate.bbox,
                    colourbar_orientation=candidate.orientation,
                    legend=options,
                    scale_model=scale.model,
                ),
            )
            legend_diagnostics.append(diagnostic)
            pixel_budget -= sum(
                (r.crop_bbox[2] - r.crop_bbox[0])
                * (r.crop_bbox[3] - r.crop_bbox[1])
                * options.ocr.upscale_factor**2
                for r in diagnostic.ocr_records
            )

    if scale is None or scale.source == "manual_endpoints":
        reasons = (
            []
            if start is not None and end is not None and start != end
            else ["finite_distinct_endpoints_required"]
        )
        scale_check = decision("scale", reasons, {"origin": "manual_endpoints"})
    elif bar is None or orientation is None:
        scale_check = decision(
            "scale", ["accepted_bar_required_for_tick_scale"], {}, unresolved=True
        )
    elif scale.source == "manual_ticks":
        try:
            if any(a.origin == "ocr" or a.status != "reviewed" for a in scale.anchors):
                raise ValueError("supplied ticks must be manually reviewed")
            evidence = calibration_evidence(scale, bar, orientation)
            if scale.model == "linear":
                fit = fit_linear_scale(
                    scale.anchors,
                    bar,
                    orientation,
                    max_value_residual_fraction=scale.max_value_residual_fraction,
                    max_position_residual_pixels=scale.max_position_residual_pixels,
                )
                start, end = fit.intercept, fit.intercept + fit.slope
            else:
                start, end = scale.anchors[0].value, scale.anchors[-1].value
            scale_check = decision(
                "scale",
                ["unsupported_calibration_span"]
                if scale.model == "piecewise_linear" and not evidence["full_span"]
                else [],
                {"origin": "reviewed_ticks", **evidence},
            )
        except ValueError as error:
            scale_check = decision("scale", [str(error)], {}, unresolved=True)
    else:
        legend = inspect_legend(
            image,
            LegendConfig(
                colourbar_bbox=bar,
                colourbar_orientation=orientation,
                legend=scale.legend,
                scale_model=scale.model,
            ),
        )
        legend_diagnostics.append(legend)
        scale_check = decision(
            "scale",
            []
            if legend.development_eligible and scale.model == "linear"
            else list(
                r
                for r in legend.reasons
                if r not in {"M12_scale_gate_closed", "interior_ticks_limit_calibration_span"}
            )
            + (
                ["piecewise_OCR_anchors_require_review"]
                if scale.model == "piecewise_linear"
                else []
            ),
            {
                "origin": "ocr_ticks",
                "experimental_eligible": legend.development_eligible,
                "fit": legend.fit.model_dump(mode="json") if legend.fit else None,
            },
            unresolved=legend.status == "unresolved",
        )
        if legend.development_eligible and legend.fit is not None and scale.model == "linear":
            ocr_anchors = tuple(
                ScaleAnchor(
                    position=t.position, value=t.parsed_value, origin="ocr", status="suggested"
                )
                for t in legend.ticks
                if t.status != "excluded" and t.parsed_value is not None
            )
    if ocr_anchors and scale is not None and bar is not None and orientation is not None:
        try:
            fit_linear_scale(
                ocr_anchors,
                bar,
                orientation,
                max_value_residual_fraction=min(0.01, scale.max_value_residual_fraction),
                max_position_residual_pixels=min(2.0, scale.max_position_residual_pixels),
            )
        except ValueError as error:
            scale_check = decision("scale", [str(error)], scale_check.evidence, unresolved=True)
    decisions.append(scale_check)
    concrete = None
    required = {"heatmap", "colourbar", "association", "rows", "columns", "scale"}
    if all(d.status == "accepted" for d in decisions if d.requirement in required):
        rows = by_kind["rows"].evidence["count"]
        columns = by_kind["columns"].evidence["count"]
        assert bar is not None and orientation is not None
        heatmap = hints.heatmap_bbox
        if heatmap is None:
            heatmap = candidates[by_kind["heatmap"].candidate_ids[0]].bbox
        if rows * columns > A.maximum_numeric_cells:
            numeric = decision("numeric_quality", ["numeric_evaluation_cell_cap_exceeded"], {})
        else:
            try:
                concrete = ExtractionConfig(
                    heatmap_bbox=heatmap,
                    colourbar_bbox=bar,
                    colourbar_orientation=orientation,
                    rows=rows,
                    columns=columns,
                    rows_segments=by_kind["heatmap"].evidence.get(
                        "rows_segments", hints.rows_segments
                    ),
                    columns_segments=by_kind["heatmap"].evidence.get(
                        "columns_segments", hints.columns_segments
                    ),
                    colourbar_start_value=None
                    if scale and scale.source != "manual_endpoints"
                    else start,
                    colourbar_end_value=None
                    if scale and scale.source != "manual_endpoints"
                    else end,
                    scale=(
                        scale
                        if scale.source != "manual_endpoints"
                        else ScaleConfig(source="manual_endpoints", quantity=scale.quantity)
                    )
                    if scale
                    else None,
                )
                values = extract(image, concrete, ocr_anchors=ocr_anchors or None)
                flagged = float(np.mean(values.flags != 0))
                finite = float(np.mean(np.isfinite(values.values)))
                reasons = []
                if flagged > A.maximum_flagged_fraction:
                    reasons.append("excessive_colour_matching_flags")
                if not np.isfinite(values.values).all():
                    reasons.append("missing_or_ambiguous_numeric_cells")
                numeric = decision(
                    "numeric_quality",
                    reasons,
                    {
                        "flagged_fraction": flagged,
                        "finite_fraction": finite,
                        "unresolved_cells": int(np.sum(~np.isfinite(values.values))),
                        **colour_quality_summary(values),
                    },
                )
            except ValueError as error:
                numeric = decision("numeric_quality", [str(error)], {}, unresolved=True)
                concrete = None
    else:
        numeric = decision("numeric_quality", ["required_capability_unresolved"], {})
    decisions.append(numeric)
    label_evidence: list[dict[str, object]] = []
    if not request.numeric_only:
        for localisation in detected.label_evidence:
            if localisation.assignment is not None:
                for slot in localisation.assignment.slots:
                    label_evidence.append(
                        {
                            "axis": localisation.axis,
                            "index": slot.index,
                            "raw_text": slot.raw_text,
                            "identifier_status": "review_required",
                            "trial_id": localisation.trial_id,
                            "geometry": localisation.assignment.geometry_evidence,
                        }
                    )
        if concrete is not None and hints.labels is not None:
            # Explicit OCR fields need actual assignment evidence, not just a supplied ROI.
            sources = [getattr(hints.labels, axis) for axis in ("rows", "columns")]
            if all(source is None or source.source == "ocr" for source in sources):
                try:
                    labels = extract_labels(
                        image, concrete.model_copy(update={"labels": hints.labels})
                    )
                    for axis in labels.axes:
                        source = getattr(hints.labels, axis.axis)
                        if source is None:
                            continue
                        geometry = axis.geometry_evidence
                        problems = []
                        if geometry is not None:
                            if geometry.get("status") != "accepted" or not geometry.get("complete"):
                                problems.append("incomplete_or_ambiguous_label_attachments")
                        elif any(not slot.candidate_indices for slot in axis.slots):
                            problems.append("incomplete_label_axis")
                        replacement = decision(
                            f"labels_{axis.axis}",
                            problems,
                            {"geometry": geometry, "identifier_review_independent": True},
                        )
                        decisions = [
                            replacement if d.requirement == replacement.requirement else d
                            for d in decisions
                        ]
                        for slot in axis.slots:
                            label_evidence.append(
                                {
                                    "axis": axis.axis,
                                    "index": slot.index,
                                    "raw_text": slot.raw_text,
                                    "identifier_status": "review_required",
                                    "geometry": geometry,
                                }
                            )
                except (ValueError, OSError, RuntimeError) as error:
                    for axis_name in ("rows", "columns"):
                        if getattr(hints.labels, axis_name) is not None:
                            replacement = decision(
                                f"labels_{axis_name}",
                                [f"label_evidence_unavailable:{error}"],
                                {},
                                unresolved=True,
                            )
                            decisions = [
                                replacement if d.requirement == replacement.requirement else d
                                for d in decisions
                            ]
        if not any(d.requirement.startswith("identifiers") for d in decisions):
            decisions.append(
                decision("identifiers", ["supplied_identifiers_require_file_validation"], {})
            )
    numeric_decisions = [d for d in decisions if d.requirement in required | {"numeric_quality"}]
    numeric_failures = [
        f"{d.requirement}:{d.status}" for d in numeric_decisions if d.status != "accepted"
    ]
    numeric_complete = decision(
        "numeric_complete",
        numeric_failures,
        {
            "identifiers": "positional",
            "heuristic_scores_are_probabilities": False,
            "calibration_span": legend.fit.recognised_span if legend and legend.fit else None,
            "sharp_measured_count_scope": sharp_measured_count_scope(decisions),
        },
    )
    geometry_decisions = [d for d in decisions if d.requirement.startswith("labels_")]
    identifier_decisions = [d for d in decisions if d.requirement.startswith("identifiers")]
    label_geometry_complete = decision(
        "label_geometry_complete",
        [f"{d.requirement}:{d.status}" for d in geometry_decisions if d.status != "accepted"],
        {"requested": not request.numeric_only, "region_acceptance_is_identifier_review": False},
    )
    identifiers_complete = decision(
        "identifiers_complete",
        [f"{d.requirement}:{d.status}" for d in identifier_decisions if d.status != "accepted"],
        {"requested": not request.numeric_only, "OCR_confidence_authorizes_names": False},
    )
    failures = [f"{d.requirement}:{d.status}" for d in decisions if d.status != "accepted"]
    complete = decision(
        "complete",
        failures,
        {"experimental_only": True},
        unresolved=any(d.status == "unresolved" for d in decisions),
    )
    release = ["experimental_export_not_requested", "stable_release_not_enabled"]
    scope_reasons = []
    if (hints.rows is None or hints.columns is None) and not M19_QUALIFICATION["inferred_counts"]:
        if not (
            M19_QUALIFICATION["sharp_measured_counts"] and sharp_measured_count_scope(decisions)
        ):
            scope_reasons.append("experimental_inferred_count_export_not_validated")
    if (
        scale is not None
        and scale.source == "ocr_ticks"
        and not M19_QUALIFICATION["linear_ocr_scale"]
    ):
        scope_reasons.append("experimental_OCR_scale_export_not_validated")
    geometry_options = (
        "max_analysis_size",
        "max_candidates",
        "max_components",
        "background_distance",
        "gridline_gap",
        "min_score",
        "max_count",
        "max_count_candidates",
    )
    defaults = DetectorOptions()
    if any(getattr(hints.detector, key) != getattr(defaults, key) for key in geometry_options):
        scope_reasons.append("experimental_export_requires_default_detector")
    if scale is not None and scale.source == "ocr_ticks":
        ocr = scale.legend.ocr
        if (
            ocr.engine != "tesseract"
            or ocr.language != "eng"
            or ocr.upscale_factor != 3
            or ocr.min_confidence < 80
            or scale.legend.max_axis_offset_pixels > 6
        ):
            scope_reasons.append("experimental_OCR_settings_outside_validated_scope")
    retained = experimental_export and not scope_reasons and concrete is not None
    allowed = retained and complete.status == "accepted" and numeric_complete.status == "accepted"
    numeric_state = (
        "complete"
        if numeric_complete.status == "accepted"
        else "partial"
        if concrete is not None
        else "unresolved"
    )
    if experimental_export:
        release = scope_reasons or ([] if allowed else ["required_per_image_capability_failed"])

    if start is not None and end is not None and not all(map(math.isfinite, (start, end))):
        raise ValueError("nonfinite scale")
    return AutomationResult(
        detection=detected,
        legend=legend,
        decisions=tuple(
            d.model_copy(
                update={
                    "policy_version": AUTOMATION_POLICY_VERSION,
                    "evidence": {
                        **d.evidence,
                        "scores_are_probabilities": False,
                        "support_scope": "aligned common-pitch grids; linear ticks; experimental",
                        "assumptions": ["displayed scale only", "no OCR identifier authorization"],
                        "alternatives_reference": "detection candidates/blocks; legend ticks",
                        "evidence_class": "supported"
                        if d.status == "accepted"
                        else "contradiction"
                        if any(
                            any(
                                word in reason
                                for word in ("contradict", "conflict", "competing", "ambiguous")
                            )
                            for reason in d.reasons
                        )
                        else "incomplete_or_weak",
                    },
                }
            )
            for d in decisions
        ),
        complete=complete,
        automatic_export_allowed=allowed,
        release_reasons=tuple(release),
        diagnostic_config=concrete,
        legend_diagnostics=tuple(legend_diagnostics),
        experimental_export_requested=experimental_export,
        export_maturity="experimental" if retained else "review_only",
        numeric_complete=numeric_complete,
        label_geometry_complete=label_geometry_complete,
        identifiers_complete=identifiers_complete,
        numeric_retention_allowed=retained,
        numeric_state=numeric_state,
        identifier_state="not_requested"
        if request.numeric_only
        else "review_required"
        if identifier_decisions
        else "user_supplied",
        ocr_anchors=ocr_anchors,
        label_evidence=tuple(label_evidence),
    )
