"""M10 measured development decisions. Release permission belongs to runtime policy.

No saved score, OCR confidence or benchmark JSON can enable automatic export.
Thresholds are engineering defaults, not probabilities. Count uniqueness and bar
numerical endpoints cannot be established from palette similarity alone.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from .detection import (
    bar_measurements,
    foreground,
    refine_box,
    score_box,
)
from .detection_policy import ACCEPTANCE_THRESHOLDS as A
from .detection_policy import POLICY_VERSION
from .edge_profiles import EdgeProfile as BoundaryEdgeEvidence
from .edge_profiles import measure_edges
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
class BoundaryEvidence:
    edges: tuple[BoundaryEdgeEvidence, ...]
    touches_image_edge: bool
    closed: bool


@dataclass(frozen=True)
class PhaseEvidence:
    positions: tuple[float, ...]
    alignment: float
    widths: tuple[float, ...] = ()
    support: tuple[float, ...] = ()


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


def boundary_evidence(
    image: RGBArray,
    box: BoundingBox,
    tolerance: float = 1.0,
) -> BoundaryEvidence:
    """Fit local transitions against robust background samples beyond the halo."""
    left, top, right, bottom = box
    edges = measure_edges(image, box, tolerance)
    return BoundaryEvidence(
        edges=edges,
        touches_image_edge=left == 0
        or top == 0
        or right == image.shape[1]
        or bottom == image.shape[0],
        closed=all(e.closed for e in edges),
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
            ["counts_unresolved_for_grid_validation"],
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
    edges = boundary_evidence(image, box, float(min(pitches) * A.maximum_edge_shift_pitch))
    if edges.touches_image_edge or not edges.closed:
        reasons.append("full_grid_bounds_unverified")
    phase: dict[Axis, PhaseEvidence] = {}
    axes: tuple[tuple[Axis, float, int], ...] = (
        ("columns", pitches[0], box[0]),
        ("rows", pitches[1], box[1]),
    )
    for axis, pitch, start in axes:
        from .lattice import observe_axis

        observed = observe_axis(image, box, axis)
        positions = observed.transitions
        offsets = (np.asarray(positions) - start) / pitch
        agreement = float(np.mean(np.abs(offsets - np.rint(offsets)) <= 0.1)) if positions else 0.0
        count = hints.columns if axis == "columns" else hints.rows
        if count == 1 and not positions:
            agreement = 1.0
        phase[axis] = PhaseEvidence(
            positions, agreement, observed.transition_widths, observed.transition_support
        )
        minimum_observations = min(2, max(0, count - 1))
        if agreement < A.minimum_phase_alignment or len(positions) < minimum_observations:
            reasons.append(f"{axis}_boundary_phase_unverified")
        if any(w / 2 > max(1.0, 0.1 * pitch) for w in observed.transition_widths):
            reasons.append(f"{axis}_broad_boundary_peaks")
    from .edge_profiles import refine_measured_box

    refined = [
        refine_measured_box(
            image, refine_box(image, box, 1, hints.detector.background_distance, mask)
        )
        for mask in masks
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
    from .bar_cores import search_bar_cores

    refined = []
    for factor in A.perturbation_threshold_factors:
        cores, _ = search_bar_cores(image, box, hints.detector.background_distance * factor)
        matching = [
            b
            for b, o in cores
            if o == candidate.orientation and np.max(np.abs(np.array(b) - box)) <= 4
        ]
        if len(matching) != 1:
            reasons.append("ambiguous_core_refinement")
            refined.append(box)
        else:
            refined.append(matching[0])
    shift = int(np.max(np.abs(np.asarray(refined) - box)))
    if shift > 1:
        reasons.append("unstable_bar_bounds")
    return decision(
        "colourbar",
        reasons,
        asdict(BarEvidence(box, measured, edges, tuple(refined), shift)),
        (candidate.id,),
    )


def count_decision(
    result: DetectionResult,
    hints: DetectionHints,
    axis: str,
    heatmap_id: str | None = None,
) -> AcceptanceDecision:
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
    proposals = [
        p
        for p in result.count_proposals
        if p.axis == axis and (heatmap_id is None or p.heatmap_id == heatmap_id)
    ]
    if not proposals:
        return decision(axis, ["no_count_observations"], {}, unresolved=True)
    evidence = []
    eligible = []
    for proposal in proposals:
        support = {
            name: signal
            for name, signal in proposal.signals.items()
            if signal.alignment >= A.minimum_count_alignment and signal.coverage >= 0.8
        }
        observations = {
            s.observation for s in support.values() if s.observation != "legacy_correlated_unknown"
        }
        # Repeated OCR on opposite sides is one observation family. Never use
        # engine confidence, repeated preprocessing or assigned slot counts.
        colour = support.get("colour_transitions")
        if colour is not None and not colour.excluded_gridline_pixels:
            observations.discard(colour.observation)
        complete = any(s.coverage >= A.minimum_count_coverage for s in support.values())
        accepted_model = (
            len(observations) >= 2
            and complete
            and not proposal.conflicts
            and proposal.pitch >= A.minimum_pitch
            and proposal.count >= 3
        )
        if accepted_model:
            eligible.append(proposal)
        evidence.append(
            {
                "id": proposal.id,
                "count": proposal.count,
                "pitch": proposal.pitch,
                "phase": proposal.phase,
                "supporting_observations": sorted(observations),
                "agreement": accepted_model,
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
    if len(eligible) == 1 and not any(
        "count_limit_hides_supported_competition" in p.reasons for p in proposals
    ):
        selected = eligible[0]
        return decision(
            axis,
            [],
            {
                "origin": "measured_inference",
                "count": selected.count,
                "model": "regular grid with complete corroborating lattice or label centres",
                "outer_completeness": "conditional on measured extent and cell-interior validation",
                "hypotheses": evidence,
            },
            (selected.id,),
        )
    return decision(
        axis,
        [
            "competing_count_lattices"
            if len(eligible) > 1
            else "insufficient_independent_count_support"
        ],
        {"hypotheses": evidence, "model": "regular rectangular grid"},
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
    axis_checks = {}
    boxes = [(c.id, c.bbox) for c in heatmaps[: A.max_shortlisted_heatmaps]]
    if hints.heatmap_bbox is not None:
        boxes = [("heatmap-manual", hints.heatmap_bbox)]
    for identifier, box in boxes:
        row = count_decision(result, hints, "rows", identifier)
        column = count_decision(result, hints, "columns", identifier)
        axis_checks[identifier] = (row, column)
        resolved = hints
        if row.status == column.status == "accepted":
            resolved = hints.model_copy(
                update={"rows": row.evidence["count"], "columns": column.evidence["count"]}
            )
        grid_checks[identifier] = grid_decision(
            image,
            resolved,
            identifier,
            box,
            masks,
            manual=hints.heatmap_bbox is not None
            and hints.rows is not None
            and hints.columns is not None,
        )
    accepted_grids = [d for d in grid_checks.values() if d.status == "accepted"]
    truncated = any("candidate_limit_hides_competition" in c.reasons for c in heatmaps)
    chosen_heatmap = None
    if len(accepted_grids) == 1 and not truncated:
        heatmap = accepted_grids[0]
        chosen_heatmap = heatmap.candidate_ids[0]
    elif len(accepted_grids) > 1 or truncated:
        heatmap = decision(
            "heatmap",
            ["competing_panels_or_bounds"],
            {"hypotheses": [d.model_dump() for d in grid_checks.values()]},
            tuple(c.id for c in heatmaps),
        )
    elif len(grid_checks) == 1:
        heatmap = next(iter(grid_checks.values()))
    else:
        heatmap = decision(
            "heatmap",
            ["no_validated_grid"],
            {"hypotheses": [d.model_dump() for d in grid_checks.values()]},
            unresolved=True,
        )
    heatmap_ids = (chosen_heatmap,) if chosen_heatmap is not None else tuple(grid_checks)
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
    selected_axes = axis_checks.get(chosen_heatmap) if chosen_heatmap else None
    decisions.extend(
        selected_axes or tuple(count_decision(result, hints, axis) for axis in ("rows", "columns"))
    )
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
    axes = {d.requirement: d for d in decisions}
    rows, columns = axes["rows"].evidence["count"], axes["columns"].evidence["count"]
    if rows * columns > A.maximum_numeric_cells:
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
                "rows": rows,
                "columns": columns,
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
