"""Composable selective decisions. Experimental eligibility never authorises export."""

import math
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .acceptance import decision
from .calibration import fit_linear_scale
from .detection import detect
from .detection_policy import ACCEPTANCE_THRESHOLDS as A
from .detection_policy import policy_snapshot
from .extract import extract
from .legend import inspect_legend
from .models import (
    AcceptanceDecision,
    DetectionHints,
    DetectionResult,
    ExtractionConfig,
    LegendConfig,
    LegendResult,
    RGBArray,
    ScaleConfig,
)

AUTOMATION_POLICY_VERSION = "m12-selective.1"


class AutomationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hints: DetectionHints
    numeric_only: Annotated[bool, Field(strict=True)]
    scale: ScaleConfig | None = None

    @model_validator(mode="after")
    def consistent_request(self) -> "AutomationRequest":
        if "colourbar" not in self.hints.requested and self.hints.colourbar_bbox is None:
            raise ValueError("automation requires a colourbar request or supplied bar")
        identifiers = self.hints.labels is not None or any(
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


def automate(image: RGBArray, request: AutomationRequest) -> AutomationResult:
    """Deployed classifier used by CLI and benchmark; receives no evaluation truth.

    The concrete config is for diagnostics only until independent release gates
    pass. Supplied/reviewed anchors compose with accepted geometry; OCR identity
    remains a separate required review. An inferred axis is never silently filled.
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
    scale = request.scale
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
            fit = fit_linear_scale(
                scale.anchors,
                bar,
                orientation,
                max_value_residual_fraction=scale.max_value_residual_fraction,
                max_position_residual_pixels=scale.max_position_residual_pixels,
            )
            start, end = fit.intercept, fit.intercept + fit.slope
            scale_check = decision(
                "scale", [], {"origin": "reviewed_ticks", "fit": fit.model_dump(mode="json")}
            )
        except ValueError as error:
            scale_check = decision("scale", [str(error)], {}, unresolved=True)
    else:
        legend = inspect_legend(
            image,
            LegendConfig(
                colourbar_bbox=bar, colourbar_orientation=orientation, legend=scale.legend
            ),
        )
        scale_check = decision(
            "scale",
            []
            if legend.development_eligible
            else list(r for r in legend.reasons if r != "M12_scale_gate_closed"),
            {
                "origin": "ocr_ticks",
                "experimental_eligible": legend.development_eligible,
                "fit": legend.fit.model_dump(mode="json") if legend.fit else None,
            },
            unresolved=legend.status == "unresolved",
        )
        if legend.development_eligible and legend.fit is not None:
            start, end = legend.fit.intercept, legend.fit.intercept + legend.fit.slope
    decisions.append(scale_check)
    concrete = None
    required = {"heatmap", "colourbar", "association", "rows", "columns", "scale"}
    if all(d.status == "accepted" for d in decisions if d.requirement in required):
        assert hints.rows is not None and hints.columns is not None
        assert bar is not None and orientation is not None
        heatmap = hints.heatmap_bbox
        if heatmap is None:
            heatmap = candidates[by_kind["heatmap"].candidate_ids[0]].bbox
        if hints.rows * hints.columns > A.maximum_numeric_cells:
            numeric = decision("numeric_quality", ["numeric_evaluation_cell_cap_exceeded"], {})
        else:
            try:
                concrete = ExtractionConfig(
                    heatmap_bbox=heatmap,
                    colourbar_bbox=bar,
                    colourbar_orientation=orientation,
                    rows=hints.rows,
                    columns=hints.columns,
                    colourbar_start_value=start,
                    colourbar_end_value=end,
                    scale=ScaleConfig(source="manual_endpoints", quantity=scale.quantity)
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
                    {"flagged_fraction": flagged, "finite_fraction": finite},
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
    allowed = not release and complete.status == "accepted"
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
    )
