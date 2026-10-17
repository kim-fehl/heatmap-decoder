"""Measured common-pitch hypotheses; proposals never silently change sampling.

A separator needs transverse constancy, two edges and an interrupted lattice.
Integer-pitch bands retain the missing/constant-cell interpretation. Large spaces
retain an independent-panel interpretation. This is a bounded support model, not
universal identifiability from raster pixels.
"""

import itertools
from typing import Any

import numpy as np

from .geometry import interior_slice
from .lattice import observe_axis
from .models import (
    Axis,
    AxisSegment,
    BlockAxisProposal,
    BlockGridProposal,
    BoundingBox,
    RGBArray,
)

BLOCK_POLICY: dict[str, Any] = {
    "version": "blocked-common-pitch.1",
    "minimum_pitch": 6.0,
    "minimum_no_gap_inferred_count": 6,
    "minimum_no_gap_transition_coverage": 0.95,
    "maximum_count": 128,
    "maximum_gap_bands": 6,
    "transverse_spread_rgb": 10.0,
    "minimum_gap_width": 4,
    "boundary_tolerance_pixels": 1.5,
    "minimum_transition_coverage": 0.60,
    "maximum_interior_error_rgb": 4.0,
    "maximum_gap_pitch_ratio": 1.5,
    "integer_gap_tolerance": 0.15,
    "maximum_block_transition_width": 1.0,
    "minimum_separator_edge_contrast_ratio": 0.5,
}


def separator_bands(
    image: RGBArray, box: BoundingBox, axis: Axis
) -> tuple[tuple[float, float], ...]:
    x, y, r, b = box
    crop = image[y:b, x:r, :3]
    if axis == "columns":
        crop = crop.transpose(1, 0, 2)
    origin = y if axis == "rows" else x
    pixels = crop.astype(float)
    # Unlike background masks this admits any colour, including palette white.
    spread = np.max(np.quantile(pixels, 0.95, axis=1) - np.quantile(pixels, 0.05, axis=1), axis=1)
    profile = np.median(pixels, axis=1)
    uniform = spread <= BLOCK_POLICY["transverse_spread_rgb"]
    # Split uniform runs when their colours differ; annotation gradients are not gaps.
    changing = np.r_[False, np.max(np.abs(np.diff(profile, axis=0)), axis=1) > 10]
    bands = []
    start = None
    for i in range(len(pixels) + 1):
        stop = i == len(pixels) or not uniform[i] or changing[i]
        if start is not None and stop:
            if i - start >= BLOCK_POLICY["minimum_gap_width"] and start > 0 and i < len(pixels):
                before = pixels[max(0, start - 3) : start]
                after = pixels[i : min(len(pixels), i + 3)]
                colour = np.median(profile[start:i], axis=0)
                # Both edges must separate the band from variable matrix interiors.
                if (
                    min(
                        float(np.mean(np.abs(before - colour))),
                        float(np.mean(np.abs(after - colour))),
                    )
                    > 12
                ):
                    bands.append((float(origin + start), float(origin + i)))
            start = None
        if i < len(pixels) and uniform[i] and start is None:
            start = i
    return tuple(bands)


def _hypotheses(
    image: RGBArray,
    box: BoundingBox,
    axis: Axis,
    gaps: tuple[tuple[float, float], ...],
    rejected: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    coord = 1 if axis == "rows" else 0
    start, end = float(box[coord]), float(box[coord + 2])
    bounds = list(zip([start] + [g[1] for g in gaps], [g[0] for g in gaps] + [end], strict=True))
    lengths = np.array([b - a for a, b in bounds])
    if np.any(lengths < BLOCK_POLICY["minimum_pitch"]):
        return []
    observations = []
    for a, b in bounds:
        sub = list(box)
        sub[coord], sub[coord + 2] = int(a), int(b)
        observations.append(observe_axis(image, tuple(sub), axis))  # type: ignore[arg-type]
    longest = int(np.argmax(lengths))
    crop = image[box[1] : box[3], box[0] : box[2], :3]
    if axis == "columns":
        crop = crop.transpose(1, 0, 2)
    pixels = crop[:, :: max(1, crop.shape[1] // 32)].astype(float)
    found = []
    for n in range(
        1,
        min(
            int(lengths[longest] / BLOCK_POLICY["minimum_pitch"]),
            int(BLOCK_POLICY["maximum_count"]),
        )
        + 1,
    ):
        pitch = float(lengths[longest] / n)
        counts = np.maximum(1, np.rint(lengths / pitch).astype(int))
        if counts.sum() > BLOCK_POLICY["maximum_count"]:
            continue
        endpoint_residual = float(np.max(np.abs(lengths - counts * pitch)))
        if endpoint_residual > BLOCK_POLICY["boundary_tolerance_pixels"]:
            if rejected is not None:
                rejected.append(
                    dict(
                        gaps=gaps,
                        counts=counts.tolist(),
                        pitch=pitch,
                        reason="incompatible_segment_endpoint_pitches",
                        endpoint_residual=endpoint_residual,
                    )
                )
            continue
        residuals: list[float] = []
        coverages: list[float] = []
        errors: list[float] = []
        transitions: list[float] = []
        strokes: list[float] = []
        plausible = True
        for (a, b), count, obs in zip(bounds, counts, observations, strict=True):
            offsets = (np.asarray(obs.transitions) - a) / ((b - a) / count)
            residual = np.abs(offsets - np.rint(offsets)) * ((b - a) / count)
            coverage = (
                len(set(np.rint(offsets).astype(int)) - {0, int(count)}) / (count - 1)
                if count > 1
                else 1
            )
            if (
                max(residual, default=0) > BLOCK_POLICY["boundary_tolerance_pixels"]
                or coverage < BLOCK_POLICY["minimum_transition_coverage"]
            ):
                plausible = False
                break
        if not plausible:
            if rejected is not None:
                rejected.append(
                    dict(
                        gaps=gaps,
                        counts=counts.tolist(),
                        pitch=pitch,
                        reason="boundary_cuts_or_unobserved_harmonic_subdivisions",
                        maximum_boundary_residual=float(max(residual, default=0)),
                        transition_coverage=float(coverage),
                    )
                )
            continue
        for (a, b), count, obs in zip(bounds, counts, observations, strict=True):
            local_pitch = (b - a) / count
            positions = np.asarray(obs.transitions)
            offsets = (positions - a) / local_pitch
            residuals.extend((np.abs(offsets - np.rint(offsets)) * local_pitch).tolist())
            boundaries = set(np.rint(offsets).astype(int)) - {0, int(count)}
            coverages.append(len(boundaries) / (count - 1) if count > 1 else 1.0)
            transitions.extend(obs.transitions)
            strokes.extend(obs.strokes)
            for i in range(int(count)):
                sl = interior_slice(
                    a - start + i * local_pitch, a - start + (i + 1) * local_pitch, 0.25
                )
                interior = pixels[sl]
                errors.append(float(np.mean(np.abs(interior - np.median(interior, axis=0)))))
        maximum_residual = max(residuals, default=0)
        # Constancy and endpoint fitting test the declared model. The transition
        # list is one observation family, never several independent votes.
        if (
            maximum_residual > BLOCK_POLICY["boundary_tolerance_pixels"]
            or max(errors, default=100) > BLOCK_POLICY["maximum_interior_error_rgb"]
            or min(coverages) < BLOCK_POLICY["minimum_transition_coverage"]
            or len(transitions) < 2
        ):
            continue
        found.append(
            dict(
                count=int(counts.sum()),
                pitch=pitch,
                segments=[
                    dict(start=a, end=b, count=int(c))
                    for (a, b), c in zip(bounds, counts, strict=True)
                ],
                gaps=gaps,
                transitions=transitions,
                strokes=strokes,
                maximum_boundary_residual=maximum_residual,
                endpoint_residual=endpoint_residual,
                transition_coverage=coverages,
                maximum_interior_error=max(errors),
                maximum_transition_width=max(
                    (w for o in observations for w in o.transition_widths), default=0
                ),
                transition_widths=[list(o.transition_widths) for o in observations],
                transition_uncertainties=[list(o.transition_uncertainties) for o in observations],
            )
        )
    return found


def propose_axis(
    image: RGBArray, box: BoundingBox, axis: Axis, supplied_count: int | None = None
) -> BlockAxisProposal:
    bands = separator_bands(image, box, axis)
    capped = len(bands) > BLOCK_POLICY["maximum_gap_bands"]
    bands = bands[: int(BLOCK_POLICY["maximum_gap_bands"])]
    hypotheses = []
    rejected: list[dict[str, Any]] = []
    # Keep no-gap and every bounded subset, including ignored annotation bands.
    for flags in itertools.product((False, True), repeat=len(bands)):
        gaps = tuple(g for g, take in zip(bands, flags, strict=True) if take)
        hypotheses.extend(_hypotheses(image, box, axis, gaps, rejected))
    # Manual count is a constraint, not an extra confirming observation.
    eligible = [h for h in hypotheses if supplied_count is None or h["count"] == supplied_count]
    eligible.sort(key=lambda h: (len(h["gaps"]), h["count"]))
    reasons = []
    if capped:
        reasons.append("separator_budget_hides_alternatives")
    if len(eligible) != 1:
        reasons.append("competing_segment_counts" if eligible else "no_supported_segment_lattice")
    chosen = eligible[0] if eligible else None
    if chosen:
        for a, b in chosen["gaps"]:
            ratio = (b - a) / chosen["pitch"]
            if abs(ratio - round(ratio)) <= BLOCK_POLICY["integer_gap_tolerance"]:
                reasons.append("gap_or_uniform_missing_cells_indistinguishable")
            if ratio > BLOCK_POLICY["maximum_gap_pitch_ratio"]:
                reasons.append("independent_panel_spacing_alternative")
    coord = 1 if axis == "rows" else 0
    segments = tuple(AxisSegment.model_validate(s) for s in chosen["segments"]) if chosen else ()
    return BlockAxisProposal(
        axis=axis,
        segments=segments,
        gaps=chosen["gaps"] if chosen else (),
        count=chosen["count"] if chosen else supplied_count or 0,
        pitch=chosen["pitch"] if chosen else 0.0,
        status="review_required" if reasons and chosen else "unresolved" if reasons else "accepted",
        reasons=tuple(dict.fromkeys(reasons)) or ("unique_supported_common_pitch_model",),
        evidence=dict(
            policy=BLOCK_POLICY,
            rejected_models=rejected,
            observed_separator_bands=bands,
            supplied_count=supplied_count,
            count_is_confirming_signal=False,
            selected=chosen,
            axis_extent=(box[coord], box[coord + 2]),
        ),
        alternatives=tuple(hypotheses),
    )


def propose_grid(
    image: RGBArray, box: BoundingBox, identifier: str, rows: int | None, columns: int | None
) -> BlockGridProposal:
    axes = (propose_axis(image, box, "rows", rows), propose_axis(image, box, "columns", columns))
    reasons = [f"{a.axis}:{r}" for a in axes if a.status != "accepted" for r in a.reasons]
    if not any(a.gaps for a in axes):
        reasons.append("no_resolved_separator")
    elif any(
        a.evidence.get("selected")
        and a.evidence["selected"]["maximum_transition_width"]
        > BLOCK_POLICY["maximum_block_transition_width"]
        for a in axes
    ):
        reasons.append("block_boundary_localisation_requires_review")
    return BlockGridProposal(
        id=f"blocks-{identifier}",
        heatmap_id=identifier,
        bbox=box,
        axes=axes,
        status="review_required" if reasons else "accepted",
        reasons=tuple(reasons) or ("unique_block_lattice",),
        evidence={"policy_version": BLOCK_POLICY["version"], "legend_association_required": True},
    )


def aligned_unions(boxes: set[BoundingBox]) -> set[BoundingBox]:
    """Bounded union proposals of aligned components; acceptance is separate."""
    joined = set(boxes)
    for _ in range(2):  # permits the four quadrants of a two-axis blocked matrix
        added = set()
        for a, b in itertools.combinations(sorted(joined), 2):
            for cross, along in ((0, 1), (1, 0)):
                if max(abs(a[cross] - b[cross]), abs(a[cross + 2] - b[cross + 2])) > 2:
                    continue
                gap = max(a[along], b[along]) - min(a[along + 2], b[along + 2])
                if not 3 <= gap <= 0.30 * max(a[along + 2] - a[along], b[along + 2] - b[along]):
                    continue
                added.add((min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])))
        joined.update(sorted(added)[:128])
        if len(joined) > 256:
            break
    return joined - boxes


def check_gap_palette(
    image: RGBArray, proposal: BlockGridProposal, bars: tuple[BoundingBox, ...]
) -> BlockGridProposal:
    """A palette-compatible strip is an unresolved annotation/data alternative."""
    if not any(a.gaps for a in proposal.axes):
        return proposal
    reasons = list(proposal.reasons) if proposal.status != "accepted" else []
    if not bars:
        reasons.append("shared_legend_required_to_resolve_separator")
    palette = []
    for x, y, r, b in bars:
        strip = image[y:b, x:r, :3].astype(float)
        profile = np.median(strip, axis=1 if b - y >= r - x else 0)
        palette.extend(profile)
    x, y, r, b = proposal.bbox
    for axis in proposal.axes:
        for start, end in axis.gaps:
            strip = (
                image[int(start) : int(end), x:r, :3]
                if axis.axis == "rows"
                else image[y:b, int(start) : int(end), :3]
            )
            colour = np.median(strip.reshape(-1, 3), axis=0)
            # Resolve only sharp separator endpoints. A uniform inner plateau
            # inside a blurred halo is not the original separator boundary.
            crop = image[y:b, x:r, :3].astype(float)
            if axis.axis == "columns":
                crop = crop.transpose(1, 0, 2)
            origin = y if axis.axis == "rows" else x
            contrasts = []
            for edge, direction in ((int(start) - origin, -1), (int(end) - origin, 1)):
                near = edge - 1 if direction == -1 else edge
                far = near + direction
                if 0 <= far < len(crop):
                    dnear = float(np.max(np.abs(np.median(crop[near], axis=0) - colour)))
                    dfar = float(np.max(np.abs(np.median(crop[far], axis=0) - colour)))
                    contrasts.append(dnear / max(dfar, 1.0))
            if contrasts and min(contrasts) < BLOCK_POLICY["minimum_separator_edge_contrast_ratio"]:
                reasons.append(f"{axis.axis}:separator_boundary_halo_requires_review")
            if palette and np.min(np.max(np.abs(np.asarray(palette) - colour), axis=1)) <= 12:
                reasons.append(f"{axis.axis}:separator_or_palette_annotation_indistinguishable")
    return proposal.model_copy(
        update={
            "status": "review_required" if reasons else "accepted",
            "reasons": tuple(dict.fromkeys(reasons)) or proposal.reasons,
            "evidence": {**proposal.evidence, "legend_boxes": bars, "gap_palette_checked": True},
        }
    )
