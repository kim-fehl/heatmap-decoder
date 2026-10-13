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
from .detection_policy import policy_snapshot
from .extract import extract
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
    ScaleConfig,
)

AUTOMATION_POLICY_VERSION = "review-selective.1"


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
        ) or any(
            k.startswith("labels_") for k in self.hints.requested
        )
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


def automate(
    image: RGBArray,
    request: AutomationRequest,
    *,
    experimental_export: bool = False,
) -> AutomationResult:
    """Deployed classifier used by CLI and benchmark; receives no evaluation truth.

    The concrete config records resolved geometry and counts. Default operation
    is diagnostic; explicit experimental export requires every per-image decision
    and the supplied-count/supplied-scale scope. OCR identity requires review.
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
            else list(r for r in legend.reasons if r != "M12_scale_gate_closed")
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
            start, end = legend.fit.intercept, legend.fit.intercept + legend.fit.slope
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
                    colourbar_start_value=None
                    if scale and scale.source == "manual_ticks"
                    else start,
                    colourbar_end_value=None if scale and scale.source == "manual_ticks" else end,
                    scale=(
                        scale
                        if scale.source == "manual_ticks"
                        else ScaleConfig(source="manual_endpoints", quantity=scale.quantity)
                    )
                    if scale
                    else None,
                )
                values = extract(image, concrete)
                flagged = float(np.mean(values.flags != 0))
                finite = float(np.mean(np.isfinite(values.values)))
                reasons = []
                if flagged > A.maximum_flagged_fraction:
                    reasons.append("excessive_colour_matching_flags")
                if 1 - finite > A.maximum_missing_or_ambiguous_fraction:
                    reasons.append("missing_or_ambiguous_numeric_cells")
                numeric = decision(
                    "numeric_quality",
                    reasons,
                    {
                        "flagged_fraction": flagged,
                        "finite_fraction": finite,
                        **colour_quality_summary(values),
                    },
                )
            except ValueError as error:
                numeric = decision("numeric_quality", [str(error)], {}, unresolved=True)
    else:
        numeric = decision("numeric_quality", ["required_capability_unresolved"], {})
    decisions.append(numeric)
    failures = [f"{d.requirement}:{d.status}" for d in decisions if d.status != "accepted"]
    complete = decision(
        "complete",
        failures,
        {"experimental_only": True},
        unresolved=any(d.status == "unresolved" for d in decisions),
    )
    policy = policy_snapshot()
    release = ["M12_release_gates_not_met"] if not policy["automatic_roi_enabled"] else []
    if hints.infer_rows and hints.rows is None and not policy["automatic_rows_enabled"]:
        release.append("inferred_rows_review_required")
    if hints.infer_columns and hints.columns is None and not policy["automatic_columns_enabled"]:
        release.append("inferred_columns_review_required")
    if scale is not None and scale.source == "ocr_ticks":
        release.append("OCR_scale_release_gate_not_met")
    scope_reasons = []
    if not request.numeric_only:
        scope_reasons.append("experimental_export_requires_numeric_only")
    if hints.rows is None or hints.columns is None:
        scope_reasons.append("experimental_inferred_count_export_not_validated")
    if scale is not None and scale.source == "ocr_ticks":
        scope_reasons.append("experimental_OCR_scale_export_not_validated")
    if hints.detector != DetectorOptions():
        scope_reasons.append("experimental_export_requires_default_detector")
    allowed = experimental_export and not scope_reasons and complete.status == "accepted"
    if experimental_export:
        release = scope_reasons or ([] if allowed else ["required_per_image_capability_failed"])

    if start is not None and end is not None and not all(map(math.isfinite, (start, end))):
        raise ValueError("nonfinite scale")
    return AutomationResult(
        detection=detected,
        legend=legend,
        decisions=tuple(decisions),
        complete=complete,
        automatic_export_allowed=allowed,
        release_reasons=tuple(release),
        diagnostic_config=concrete,
        legend_diagnostics=tuple(legend_diagnostics),
        experimental_export_requested=experimental_export,
        export_maturity="experimental" if allowed else "review_only",
    )
