"""M10 measured development decisions. Release permission belongs to runtime policy.

No saved score, OCR confidence or benchmark JSON can enable automatic export.
Thresholds are engineering defaults, not probabilities. Count uniqueness and bar
numerical endpoints cannot be established from palette similarity alone.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .detection import (
    _count_signals,
    _refine_bar,
    _refine_noisy_bar,
    bar_measurements,
    foreground,
    refine_box,
    score_box,
)
from .detection_policy import ACCEPTANCE_THRESHOLDS as A
from .detection_policy import POLICY_VERSION
from .models import (
    AcceptanceDecision,
    AcceptanceReport,
    Axis,
    BoolArray,
    BoundingBox,
    DetectionHints,
    DetectionResult,
    ExtractionConfig,
    RGBArray,
    ROICandidate,
)


@dataclass(frozen=True)
class BoundaryEdgeEvidence:
    contrast_rgb: float
    outside_spread_rgb: float


@dataclass(frozen=True)
class BoundaryEvidence:
    edges: tuple[BoundaryEdgeEvidence, ...]
    touches_image_edge: bool
    closed: bool


@dataclass(frozen=True)
class PhaseEvidence:
    positions: tuple[float, ...]
    alignment: float


@dataclass(frozen=True)
class GridEvidence:
    bbox: BoundingBox
    grid: dict[str, float]
    full_bounds: BoundaryEvidence
    phase: dict[Axis, PhaseEvidence]
    perturbed_boxes: tuple[BoundingBox, ...]
    maximum_edge_shift_cell_pitch: float
    perturbation_threshold_factors: tuple[float, ...]


@dataclass(frozen=True)
class BarEvidence:
    bbox: BoundingBox
    continuity: dict[str, float]
    extent: BoundaryEvidence
    perturbed_boxes: tuple[BoundingBox, ...]
    maximum_edge_shift_pixels: int
    numerical_endpoint_completeness: str = "user_supplied_scale_required"


def decision(
    requirement: str,
    reasons: list[str],
    evidence: dict[str, Any],
    ids: tuple[str, ...] = (),
    *,
    unresolved: bool = False,
) -> AcceptanceDecision:
    return AcceptanceDecision(
        requirement=requirement,
        status="unresolved" if unresolved else "review_required" if reasons else "accepted",
        reasons=tuple(reasons) or ("development_evidence_passed",),
        evidence=evidence,
        candidate_ids=ids,
        policy_version=POLICY_VERSION,
    )


def distinct_candidates(
    candidates: tuple[ROICandidate, ...],
    hints: DetectionHints,
) -> tuple[ROICandidate, ...]:
    """Deduplicate within 0.1 supplied cell pitch (bar edges within one pixel).

    Never combine coordinates or average incompatible hypotheses. Keep losing
    alternatives in DetectionResult; this function is only for competition checks.
    """
    retained: list[ROICandidate] = []
    for candidate in candidates:
        box = np.array(candidate.bbox)
        tolerance = np.ones(4)
        if candidate.kind == "heatmap" and hints.rows and hints.columns:
            tolerance = A.maximum_edge_shift_pitch * np.array(
                [
                    (box[2] - box[0]) / hints.columns,
                    (box[3] - box[1]) / hints.rows,
                ]
                * 2
            )
        if not any(
            candidate.kind == other.kind and np.all(np.abs(box - other.bbox) <= tolerance)
            for other in retained
        ):
            retained.append(candidate)
    return tuple(retained)


def boundary_evidence(image: RGBArray, box: BoundingBox) -> BoundaryEvidence:
    """All four edges must close against a separable background or border stroke.

    A strong internal transition alone does not establish the outside of a grid.
    Work on original pixels, and retain each edge's observation separately.
    """
    left, top, right, bottom = box
    edges = []
    pairs = (
        (image[top:bottom, left, :3], image[top:bottom, max(0, left - 1), :3]),
        (image[top:bottom, right - 1, :3], image[top:bottom, min(image.shape[1] - 1, right), :3]),
        (image[top, left:right, :3], image[max(0, top - 1), left:right, :3]),
        (image[bottom - 1, left:right, :3], image[min(image.shape[0] - 1, bottom), left:right, :3]),
    )
    for inside, outside in pairs:
        a, b = inside.astype(float), outside.astype(float)
        edges.append(
            BoundaryEdgeEvidence(
                contrast_rgb=float(np.mean(np.abs(a - b))),
                outside_spread_rgb=float(np.max(np.ptp(b, axis=0))),
            )
        )
    return BoundaryEvidence(
        edges=tuple(edges),
        touches_image_edge=left == 0
        or top == 0
        or right == image.shape[1]
        or bottom == image.shape[0],
        closed=all(
            e.contrast_rgb >= A.minimum_boundary_contrast and e.outside_spread_rgb <= 8
            for e in edges
        ),
    )


def grid_decision(
    image: RGBArray,
    hints: DetectionHints,
    identifier: str,
    box: BoundingBox,
    masks: tuple[BoolArray, ...],
    *,
    manual: bool = False,
) -> AcceptanceDecision:
    if manual:
        return decision("heatmap", [], {"origin": "manual", "bbox": box}, (identifier,))
    if hints.rows is None or hints.columns is None:
        return decision(
            "heatmap",
            ["supplied_counts_required_for_grid_validation"],
            {"bbox": box},
            (identifier,),
        )
    pitches = np.array([(box[2] - box[0]) / hints.columns, (box[3] - box[1]) / hints.rows] * 2)
    measurements = score_box(image, box, hints)
    reasons = []
    if min(pitches) < A.minimum_pitch:
        reasons.append("unusable_cell_interiors")
    if measurements["equal_grid"] < A.minimum_equal_grid:
        reasons.append("grid_interior_disagreement")
    if measurements["piecewise_constant"] < A.minimum_piecewise_constant:
        reasons.append("nonconstant_cell_interiors")
    edges = boundary_evidence(image, box)
    if edges.touches_image_edge or not edges.closed:
        reasons.append("full_grid_bounds_unverified")
    phase: dict[Axis, PhaseEvidence] = {}
    axes: tuple[tuple[Axis, float, int], ...] = (
        ("columns", pitches[0], box[0]),
        ("rows", pitches[1], box[1]),
    )
    for axis, pitch, start in axes:
        positions = _count_signals(image, box, axis)["colour_transitions"]
        offsets = (np.asarray(positions) - start) / pitch
        agreement = float(np.mean(np.abs(offsets - np.rint(offsets)) <= 0.1)) if positions else 0.0
        phase[axis] = PhaseEvidence(positions=positions, alignment=agreement)
        if agreement < A.minimum_phase_alignment:
            reasons.append(f"{axis}_boundary_phase_unverified")
    refined = [
        refine_box(image, box, 1, hints.detector.background_distance, mask) for mask in masks
    ]
    shift = float(np.max(np.abs(np.asarray(refined) - box) / pitches))
    if shift > A.maximum_edge_shift_pitch:
        reasons.append("unstable_heatmap_bounds")
    return decision(
        "heatmap",
        reasons,
        asdict(
            GridEvidence(
                box,
                measurements,
                edges,
                phase,
                tuple(refined),
                shift,
                A.perturbation_threshold_factors,
            )
        ),
        (identifier,),
    )


def bar_decision(
    image: RGBArray,
    hints: DetectionHints,
    candidate: ROICandidate,
    masks: tuple[BoolArray, ...],
) -> AcceptanceDecision:
    box = candidate.bbox
    left, top, right, bottom = box
    crop = image[top:bottom, left:right, :3].astype(float)
    if candidate.orientation == "horizontal":
        crop = crop.transpose(1, 0, 2)
    measured = bar_measurements(crop)
    reasons = []
    if measured["noise_rgb"] > A.maximum_bar_noise or measured["transverse_error"] > 8:
        reasons.append("bar_transverse_noise_excessive")
    if measured["maximum_step_rgb"] > measured["step_limit_rgb"]:
        reasons.append("bar_discontinuity")
    if measured["longest_plateau_fraction"] > A.maximum_plateau_fraction:
        reasons.append("categorical_or_saturated_strip")
    if measured["variation_rgb"] < 12 or measured["changing_fraction"] < 0.2:
        reasons.append("insufficient_bar_variation")
    edges = boundary_evidence(image, box)
    if edges.touches_image_edge or not edges.closed:
        reasons.append("bar_extent_or_clipping_unverified")
    refined = [_refine_noisy_bar(image, _refine_bar(image, box, mask)) for mask in masks]
    shift = int(np.max(np.abs(np.asarray(refined) - box)))
    if shift > 1:
        reasons.append("unstable_bar_bounds")
    return decision(
        "colourbar",
        reasons,
        asdict(BarEvidence(box, measured, edges, tuple(refined), shift)),
        (candidate.id,),
    )


def count_decision(result: DetectionResult, hints: DetectionHints, axis: str) -> AcceptanceDecision:
    supplied = getattr(hints, axis)
    if supplied is not None:
        return decision(
            axis,
            [],
            {
                "origin": "manual",
                "count": supplied,
                "inference_requested": getattr(hints, f"infer_{axis}"),
            },
        )
    proposals = [p for p in result.count_proposals if p.axis == axis]
    if not proposals:
        return decision(axis, ["no_count_observations"], {}, unresolved=True)
    evidence = []
    for proposal in proposals:
        support = {
            name: signal
            for name, signal in proposal.signals.items()
            if signal.alignment >= A.minimum_count_alignment
            and signal.coverage >= A.minimum_count_coverage
        }
        observations = {
            s.observation for s in support.values() if s.observation != "legacy_correlated_unknown"
        }
        # Repeated OCR on opposite sides is one observation family. Never use
        # engine confidence, repeated preprocessing or assigned slot counts.
        colour = support.get("colour_transitions")
        if colour is not None and not colour.excluded_gridline_pixels:
            observations.discard(colour.observation)
        evidence.append(
            {
                "id": proposal.id,
                "count": proposal.count,
                "pitch": proposal.pitch,
                "phase": proposal.phase,
                "supporting_observations": sorted(observations),
                "agreement": len(observations) >= 2 and not proposal.conflicts,
                "signals": {name: signal.model_dump() for name, signal in proposal.signals.items()},
                "conflicts": proposal.conflicts,
                "harmonic_alternatives": [
                    p.id
                    for p in proposals
                    if p.count != proposal.count
                    and (p.count % proposal.count == 0 or proposal.count % p.count == 0)
                ],
            }
        )
    return decision(
        axis,
        ["count_uniqueness_and_outer_completeness_unestablished"],
        {"hypotheses": evidence, "automatic_axis_enabled": False},
        tuple(p.id for p in proposals),
    )


def evaluate_acceptance(
    image: RGBArray,
    hints: DetectionHints,
    result: DetectionResult,
) -> AcceptanceReport:
    """Request-specific measured decisions; never mutate selections or extraction config."""
    decisions = []
    distinct = distinct_candidates(result.candidates, hints)
    heatmaps = tuple(c for c in distinct if c.kind == "heatmap")
    bars = tuple(c for c in distinct if c.kind == "colourbar")
    masks = tuple(
        foreground(image, hints.detector.background_distance * factor)
        for factor in A.perturbation_threshold_factors
    )
    grid_checks = {}
    if hints.heatmap_bbox is not None:
        grid_checks["heatmap-manual"] = grid_decision(
            image, hints, "heatmap-manual", hints.heatmap_bbox, masks, manual=True
        )
    for candidate in heatmaps[: A.max_shortlisted_heatmaps]:
        grid_checks[candidate.id] = grid_decision(image, hints, candidate.id, candidate.bbox, masks)
    heatmap_ids = tuple(grid_checks)
    if not heatmap_ids:
        heatmap = decision("heatmap", ["no_supported_heatmap"], {}, unresolved=True)
    elif len(heatmaps) > 1 or any(
        "candidate_limit_hides_competition" in c.reasons for c in heatmaps
    ):
        heatmap = decision(
            "heatmap",
            ["competing_panels_or_bounds"],
            {"hypotheses": [d.model_dump() for d in grid_checks.values()]},
            tuple(c.id for c in heatmaps),
        )
    else:
        heatmap = next(iter(grid_checks.values()))
    decisions.append(heatmap)
    bar_required = "colourbar" in hints.requested or hints.colourbar_bbox is not None
    if bar_required:
        if hints.colourbar_bbox is not None:
            bar = decision(
                "colourbar",
                [],
                {"origin": "manual", "bbox": hints.colourbar_bbox},
                ("colourbar-manual",),
            )
            bar_ids = {"colourbar-manual"}
        elif not bars:
            bar = decision("colourbar", ["no_continuous_bar"], {}, unresolved=True)
            bar_ids = set()
        else:
            checks = [bar_decision(image, hints, c, masks) for c in bars]
            bar_ids = {c.id for c in bars}
            bar = (
                checks[0]
                if len(checks) == 1
                and not any("candidate_limit_hides_competition" in c.reasons for c in bars)
                else decision(
                    "colourbar",
                    ["competing_bars"],
                    {"hypotheses": [d.model_dump() for d in checks]},
                    tuple(c.id for c in bars),
                )
            )
        decisions.append(bar)
        associations = [
            a
            for a in result.associations
            if a.heatmap_id in heatmap_ids and a.colourbar_id in bar_ids
        ]
        if hints.heatmap_bbox is not None and hints.colourbar_bbox is not None:
            association = decision("association", [], {"origin": "manual"})
        elif (
            len(associations) != 1 or len(heatmap_ids) * len(bar_ids) > A.maximum_association_pairs
        ):
            association = decision(
                "association",
                ["nonunique_heatmap_bar_association"],
                {"hypotheses": [a.model_dump() for a in associations]},
                unresolved=not associations,
            )
        else:
            a = associations[0]
            reasons = (
                []
                if a.score_components["palette_coverage"] >= A.minimum_association_coverage
                else ["inadequate_palette_coverage"]
            )
            association = decision("association", reasons, a.model_dump(), (a.id,))
        decisions.append(association)
        if hints.colourbar_start_value is None or hints.colourbar_end_value is None:
            scale = decision("scale", ["manual_scale_endpoints_required"], {}, unresolved=True)
        elif hints.colourbar_start_value == hints.colourbar_end_value:
            scale = decision("scale", ["zero_scale_range"], {}, unresolved=True)
        else:
            scale = decision(
                "scale",
                [],
                {
                    "origin": "manual_endpoints",
                    "start": hints.colourbar_start_value,
                    "end": hints.colourbar_end_value,
                },
            )
        decisions.append(scale)
    decisions.extend(count_decision(result, hints, axis) for axis in ("rows", "columns"))
    if bar_required:
        decisions.append(numeric_decision(image, hints, result, decisions))
    identifiers: Literal["not_requested", "review_required", "user_supplied"] = "not_requested"
    for axis in ("rows", "columns"):
        source = getattr(hints.labels, axis) if hints.labels is not None else None
        if f"labels_{axis}" in hints.requested or (source is not None and source.source == "ocr"):
            identifiers = "review_required"
            kind = f"labels_{axis}"
            if source is not None and source.source == "ocr":
                region = decision(kind, [], {"origin": "manual", "bbox": source.bbox})
            else:
                matches = [
                    c
                    for c in distinct
                    if c.kind == kind and any(i in heatmap_ids for i in c.associations)
                ]
                if not matches:
                    region = decision(kind, ["no_supported_label_region"], {}, unresolved=True)
                elif len(matches) != 1:
                    region = decision(
                        kind,
                        ["competing_label_regions"],
                        {"hypotheses": [c.model_dump() for c in matches]},
                        tuple(c.id for c in matches),
                    )
                else:
                    candidate = matches[0]
                    reasons = [
                        r
                        for r in candidate.reasons
                        if r in {"incomplete_label_axis", "ambiguous_label_assignment"}
                    ]
                    region = decision(kind, reasons, candidate.model_dump(), (candidate.id,))
            decisions.append(region)
            decisions.append(
                decision(
                    f"identifiers_{axis}",
                    ["identifier_text_requires_review"],
                    {"heuristic_recognition_is_verification": False},
                )
            )
        elif source is not None and identifiers == "not_requested":
            identifiers = "user_supplied"
    failed = [d for d in decisions if d.status != "accepted"]
    overall = decision(
        "overall",
        [f"{d.requirement}:{d.status}" for d in failed],
        {"release_gate": "M12_not_established", "mode": "development_only"},
        unresolved=any(d.status == "unresolved" for d in failed),
    )
    return AcceptanceReport(
        decisions=tuple(decisions), overall=overall, identifier_verification=identifiers
    )


def numeric_decision(
    image: RGBArray,
    hints: DetectionHints,
    result: DetectionResult,
    decisions: list[AcceptanceDecision],
) -> AcceptanceDecision:
    """Evaluate numeric QC only after concrete geometry/count/scale checks pass."""
    from .extract import extract

    required = {"heatmap", "colourbar", "association", "rows", "columns", "scale"}
    if any(d.status != "accepted" for d in decisions if d.requirement in required):
        return decision("numeric_quality", ["required_geometry_counts_or_scale_unresolved"], {})
    assert hints.rows is not None and hints.columns is not None
    if hints.rows * hints.columns > A.maximum_numeric_cells:
        return decision("numeric_quality", ["numeric_evaluation_cell_cap_exceeded"], {})
    choices = {d.requirement: d.candidate_ids for d in decisions}
    candidates = {c.id: c for c in result.candidates}
    heatmap = hints.heatmap_bbox or candidates[choices["heatmap"][0]].bbox
    bar = hints.colourbar_bbox or candidates[choices["colourbar"][0]].bbox
    orientation = hints.colourbar_orientation
    if orientation is None:
        orientation = (
            candidates[choices["colourbar"][0]].orientation
            if hints.colourbar_bbox is None
            else "vertical"
            if bar[3] - bar[1] >= bar[2] - bar[0]
            else "horizontal"
        )
    try:
        config = ExtractionConfig.model_validate(
            {
                "heatmap_bbox": heatmap,
                "colourbar_bbox": bar,
                "colourbar_orientation": orientation,
                "rows": hints.rows,
                "columns": hints.columns,
                "colourbar_start_value": hints.colourbar_start_value,
                "colourbar_end_value": hints.colourbar_end_value,
            }
        )
        extracted = extract(image, config)
    except ValueError as error:
        return decision(
            "numeric_quality", ["invalid_concrete_extraction_config"], {"error": str(error)}
        )
    flagged = float(np.mean(extracted.flags != 0))
    missing = float(np.mean(~np.isfinite(extracted.values)))
    reasons = []
    if flagged > A.maximum_flagged_fraction:
        reasons.append("excessive_colour_matching_flags")
    if missing > A.maximum_missing_or_ambiguous_fraction:
        reasons.append("missing_or_ambiguous_numeric_cells")
    return decision(
        "numeric_quality",
        reasons,
        {
            "concrete_config": config.model_dump(mode="json"),
            "flagged_fraction": flagged,
            "missing_or_ambiguous_fraction": missing,
            "maximum_colour_distance": float(np.max(extracted.colour_distance)),
            "finite_fraction": 1 - missing,
        },
    )
