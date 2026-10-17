"""Bounded geometric coarse-to-fine orientation proposals, independent of OCR scores."""

import math
from typing import Any

import numpy as np

from .labels import CropTransform
from .models import BoundingBox, RGBArray, normalise_text_angle
from .oriented_labels import LabelBudget

ANGLE_POLICY: dict[str, Any] = {
    "version": "geometric-angle-search.1",
    "coarse_step_degrees": 15,
    "fine_steps_degrees": [5, 2, 1, 0.5],
    "maximum_geometric_trials": 21,
    "minimum_score_ratio": 1.05,
    "minimum_ink_pixels": 30,
    "reading_direction_from_confidence": False,
}


def estimate_angle(image: RGBArray, box: BoundingBox, budget: LabelBudget) -> dict[str, Any]:
    """Concentration of deskewed transverse ink profiles; deterministic finite trials."""
    from PIL import Image
    from scipy import ndimage  # type: ignore[import-untyped]

    raw = np.asarray(Image.fromarray(image[:, :, :3]).crop(box).convert("L")).copy()
    if not budget.reserve(raw.size):
        return dict(
            policy=ANGLE_POLICY,
            status="capped",
            trials=[dict(status="skipped", reason="shared_budget_exhausted")],
            reason="shared_budget_exhausted",
        )
    ink = raw < 175
    components, _ = ndimage.label(ink)
    excluded = []
    for index, region in enumerate(ndimage.find_objects(components), 1):
        if region is None:
            continue
        mask = components[region] == index
        height, width = mask.shape
        if mask.sum() > 80 and mask.mean() > 0.9 and max(height, width) > 4 * min(height, width):
            raw[region][mask] = 255
            excluded.append(
                dict(
                    bbox=(region[1].start, region[0].start, region[1].stop, region[0].stop),
                    reason="solid_strip_not_text_orientation_evidence",
                )
            )
    if np.count_nonzero(raw < 175) < ANGLE_POLICY["minimum_ink_pixels"]:
        return dict(
            policy=ANGLE_POLICY,
            status="unresolved",
            trials=[],
            reason="no_orientation_structure",
            excluded_components=excluded,
        )
    source_image = Image.fromarray(raw)
    local_box = (0, 0, raw.shape[1], raw.shape[0])
    trials: list[dict[str, Any]] = []

    def measure(angle: float) -> None:
        transform = CropTransform(local_box, 1, angle)
        pixels = math.prod(transform.crop_size)
        if not budget.reserve(pixels):
            trials.append(
                dict(angle=angle, status="skipped", reason="shared_budget_exhausted", pixels=pixels)
            )
            return
        prepared = transform.prepare(source_image)
        ink = np.asarray(prepared) < 175
        count = float(ink.sum())
        glyphs, _ = ndimage.label(ink)
        glyph_regions = ndimage.find_objects(glyphs)
        heights = [
            r[0].stop - r[0].start
            for r in glyph_regions
            if r is not None and r[0].stop - r[0].start >= 3
        ]
        height = float(np.median(heights)) if heights else 3.0
        gap = max(3, min(65, round(2.0 * height)))
        joined, _ = ndimage.label(ndimage.binary_dilation(ink, structure=np.ones((3, gap), bool)))
        groups = ndimage.find_objects(joined)
        weighted, used = 0.0, 0
        line_shapes = []
        for index, region in enumerate(groups, 1):
            if region is None:
                continue
            yy, xx = np.nonzero(ink[region] & (joined[region] == index))
            if len(xx) < 3:
                continue
            width, span = int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)
            if width < 3 * height:
                continue
            # A merged diagonal assembly can have a wide bounding box. Test
            # glyph baseline agreement so its aspect ratio cannot masquerade as
            # one upright line; both reading directions use the same geometry.
            ids = np.unique(glyphs[region][ink[region]])
            glyph_boxes = [
                glyph_regions[int(i) - 1]
                for i in ids
                if i > 0 and glyph_regions[int(i) - 1] is not None
            ]
            tall = [g for g in glyph_boxes if g[0].stop - g[0].start >= 0.55 * height]
            dispersion = 100.0
            if len(tall) >= 6:
                tops = np.array([g[0].start for g in tall], float)
                bottoms = np.array([g[0].stop for g in tall], float)
                dispersion = min(
                    float(np.quantile(np.abs(tops - np.median(tops)), 0.8)),
                    float(np.quantile(np.abs(bottoms - np.median(bottoms)), 0.8)),
                )
            weighted += (width / max(span, 1)) * len(xx) / (1 + dispersion)
            used += len(xx)
            line_shapes.append(
                dict(width=width, height=span, ink_pixels=len(xx), baseline_dispersion=dispersion)
            )
        score = weighted / max(count, 1) if count >= ANGLE_POLICY["minimum_ink_pixels"] else 0.0
        trials.append(
            dict(
                angle=angle,
                status="measured",
                score=score,
                pixels=pixels,
                ink_pixels=int(count),
                line_shapes=line_shapes,
                eligible_ink_fraction=used / max(count, 1),
            )
        )

    # [-90,90) identifies an unoriented axis; the 180-degree alternative is explicit below.
    for coarse_angle in range(-90, 90, 15):
        measure(float(coarse_angle))
    measured = [t for t in trials if t["status"] == "measured"]
    if not measured or max(t["score"] for t in measured) <= 0:
        return dict(
            policy=ANGLE_POLICY,
            status="unresolved",
            trials=trials,
            reason="no_orientation_structure",
        )
    coarse = sorted(measured, key=lambda t: (-t["score"], t["angle"]))
    best = coarse[0]
    for step in ANGLE_POLICY["fine_steps_degrees"]:
        centre = best["angle"]
        for delta in (-step, step):
            candidate = (centre + delta + 90) % 180 - 90
            if not any(abs(t["angle"] - candidate) < 1e-9 for t in trials):
                measure(candidate)
        measured = [t for t in trials if t["status"] == "measured"]
        best = sorted(measured, key=lambda t: (-t["score"], t["angle"]))[0]
    distant = [t for t in measured if abs((t["angle"] - best["angle"] + 90) % 180 - 90) >= 12]
    competitor = max((t["score"] for t in distant), default=0)
    supported = (
        best["score"] >= ANGLE_POLICY["minimum_score_ratio"] * competitor
        and best["score"] > 0
        and len(best["line_shapes"]) >= 2
        and best["eligible_ink_fraction"] >= 0.75
    )
    complete = all(t["status"] == "measured" for t in trials)
    angle = normalise_text_angle(best["angle"])
    return dict(
        policy=ANGLE_POLICY,
        excluded_components=excluded,
        source_bbox=box,
        status="unique_axis" if supported and complete else "review_required",
        angle=angle,
        angle_alternatives=(angle, normalise_text_angle(angle + 180)),
        reading_direction="unresolved_180_degree_alternative",
        trials=trials,
        score_kind="elongated-line geometry; not probability or OCR confidence",
        search_resolution_degrees=0.5,
        reason="geometric_axis_supported"
        if supported and complete
        else "weak_or_incomplete_orientation_geometry",
    )
