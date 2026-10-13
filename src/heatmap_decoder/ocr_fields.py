"""Image-bound OCR fields: shared text band, independent of numeric tick values."""

from typing import Any

import numpy as np
from PIL import Image

from .models import BoundingBox, LegendConfig, LegendTick, RGBArray


def _runs(positions: np.ndarray[Any, Any], missing_gap: int = 0) -> list[np.ndarray[Any, Any]]:
    if not len(positions):
        return []
    return list(np.split(positions, np.flatnonzero(np.diff(positions) > missing_gap + 1) + 1))


def aligned_tick_fields(
    image: RGBArray,
    config: LegendConfig,
    side: str,
    region: BoundingBox,
    strokes: tuple[LegendTick, ...],
) -> tuple[dict[float, BoundingBox], dict[str, Any]]:
    """Shared perpendicular text band and separate ink clusters including observed signs.

    No equal tick spacing, known tick count, OCR spelling or numeric values enter
    this measurement. Competing bands, merged labels and missing ink abstain.
    """
    vertical = config.colourbar_orientation == "vertical"
    axial = 1 if vertical else 0
    bar = config.colourbar_bbox
    relevant = [
        t
        for t in strokes
        if bar[axial] <= t.position < bar[axial + 2] and t.stroke_bbox is not None
    ]
    evidence: dict[str, Any] = {
        "assumption": "shared_left" if vertical else "shared_top",
        "source": "observed_ink_projection",
        "reason": None,
    }
    if len(relevant) < 3:
        evidence["reason"] = "fewer_than_three_strokes_for_shared_band"
        return {}, evidence
    boxes = [t.stroke_bbox for t in relevant if t.stroke_bbox is not None]
    cross = 0 if vertical else 1
    before = side in {"left", "above"}
    cross_start = region[cross] if before else int(max(b[cross + 2] for b in boxes)) + 2
    cross_end = int(min(b[cross] for b in boxes)) - 2 if before else region[cross + 2]
    if cross_end <= cross_start:
        evidence["reason"] = "no_space_beyond_strokes"
        return {}, evidence
    x0, y0, x1, y1 = (
        (cross_start, bar[1], cross_end, bar[3])
        if vertical
        else (bar[0], cross_start, bar[2], cross_end)
    )
    grayscale = np.asarray(Image.fromarray(image[:, :, :3]).convert("L"))
    mask = grayscale[y0:y1, x0:x1] < 140
    if vertical:
        mask = mask.T
    bands = [
        r
        for r in _runs(np.flatnonzero(np.sum(mask, axis=1) >= 3), 6 if vertical else 1)
        if len(r) >= 3
    ]
    bands.sort(key=len, reverse=True)
    if not bands or (len(bands) > 1 and (vertical or len(bands[1]) >= 0.8 * len(bands[0]))):
        evidence["reason"] = "missing_or_competing_shared_text_bands"
        return {}, evidence
    band = bands[0]
    a, b = int(band[0]), int(band[-1]) + 1
    # Bridge a measured glyph-sized gap: signed text stays one observation.
    gap = min(6, max(2, round((b - a) * 0.35)))
    groups = _runs(np.flatnonzero(np.any(mask[a:b], axis=0)), gap)
    evidence.update(
        {
            "cross_band": [cross_start + a, cross_start + b],
            "glyph_gap_pixels": gap,
            "dark_luminance_threshold": 140,
            "clusters": [[bar[axial] + int(g[0]), bar[axial] + int(g[-1]) + 1] for g in groups],
        }
    )
    fields: dict[float, BoundingBox] = {}
    matches: dict[float, list[tuple[int, int]]] = {t.position: [] for t in relevant}
    offset = config.legend.max_axis_offset_pixels
    for group in groups:
        first, last = bar[axial] + int(group[0]), bar[axial] + int(group[-1]) + 1
        owners = [t for t in relevant if first - offset <= t.position <= last + offset]
        if len(owners) == 1:
            matches[owners[0].position].append((first, last))
    pad = config.legend.crop_padding_pixels
    for position, matched in matches.items():
        if len(matched) != 1:
            continue
        first, last = matched[0]
        box = (
            (cross_start + a, first, cross_start + b, last)
            if vertical
            else (first, cross_start + a, last, cross_start + b)
        )
        fields[position] = (
            max(region[0], box[0] - pad),
            max(region[1], box[1] - pad),
            min(region[2], box[2] + pad),
            min(region[3], box[3] + pad),
        )
    evidence["matched_fields"] = len(fields)
    if len(fields) != len(relevant):
        evidence["reason"] = "missing_or_ambiguous_tick_fields"
    return fields, evidence
