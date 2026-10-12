"""Find coherent colour-bearing strips inside component SEARCH windows.

No chroma assumption, component-centre seed, tick-text preference or learned model.
Original pixels determine the final borders; bounded sampling only proposes cores.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .detection_policy import THRESHOLDS as T
from .models import BoundingBox, RGBArray

CORE_SEARCH_VERSION = "core-profile.2"
MAX_AXIS_SAMPLES = 512
MAX_CORES_PER_WINDOW = 16


def _groups(indices: NDArray[np.intp]) -> list[NDArray[np.intp]]:
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)) if indices.size else []


def search_bar_cores(
    image: RGBArray,
    window: BoundingBox,
    background_distance: float = 12,
) -> tuple[tuple[tuple[BoundingBox, str], ...], dict[str, Any]]:
    """Search both orientations; every coherent band is retained, never rank-one chosen."""
    x, y, r, s = window
    margin = 6
    x, y, r, s = (
        max(0, x - margin),
        max(0, y - margin),
        min(image.shape[1], r + margin),
        min(image.shape[0], s + margin),
    )
    background = np.median(
        np.concatenate((image[0, :, :3], image[-1, :, :3], image[:, 0, :3], image[:, -1, :3])),
        axis=0,
    ).astype(float)
    found: list[tuple[BoundingBox, str]] = []
    audit: list[dict[str, Any]] = []
    truncated = False
    for vertical in (True, False):
        crop = image[y:s, x:r, :3]
        if not vertical:
            crop = crop.transpose(1, 0, 2)
        length, width = crop.shape[:2]
        if length < 24 or width < 2:
            continue
        # Ignore only the window's bounded end margin for seed measurements.
        # Refinement below measures the whole original strip, including both ends.
        inset = min(16, length // 8)
        sampled = crop[
            inset : length - inset : max(1, (length - 2 * inset) // MAX_AXIS_SAMPLES)
        ].astype(float)
        if len(sampled) < 8:
            continue
        steps = np.max(np.abs(np.diff(sampled, axis=0)), axis=2)
        variation = np.mean(np.std(sampled, axis=0), axis=1)
        valid = (
            (variation >= 12)
            & (np.mean(steps > 0.25, axis=0) >= 0.2)
            & (np.quantile(steps, 0.99, axis=0) <= 20)
        )
        groups = [g for g in _groups(np.flatnonzero(valid)) if len(g) >= 2]
        audit.append(
            dict(
                orientation="vertical" if vertical else "horizontal",
                coherent_bands=len(groups),
                sampled_long_axis=len(sampled),
            )
        )
        for group in groups:
            if len(found) >= MAX_CORES_PER_WINDOW:
                truncated = True
                break
            a, b = int(group[0]), int(group[-1]) + 1
            if length / (b - a) < 4:
                continue
            # The coherent band itself provides a reference, independent of the
            # component midpoint. Estimate coverage versus local background.
            core = crop[:, a:b].astype(float)
            profile = np.median(core[:, (b - a) // 4 : (b - a) - (b - a) // 4], axis=1)
            foreground = np.max(np.abs(profile - background), axis=1) > background_distance
            runs = [g for g in _groups(np.flatnonzero(foreground)) if len(g) >= 24]
            # White may be a legitimate interior palette colour. Join only
            # low-contrast gaps whose complete original-pixel profile is smooth.
            merged: list[NDArray[np.intp]] = []
            for run in runs:
                if merged:
                    previous = merged[-1]
                    gap = int(run[0] - previous[-1] - 1)
                    bridge = profile[int(previous[-1]) : int(run[0]) + 1]
                    varying_bridge = (
                        gap <= T.bar_join_gap_fraction * length
                        and np.max(np.ptp(bridge, axis=0)) >= background_distance
                        and np.mean(np.max(np.abs(np.diff(bridge, axis=0)), axis=1) > 0.25) >= 0.5
                    )
                    if (gap <= max(4, length // 10) or varying_bridge) and np.max(
                        np.abs(np.diff(bridge, axis=0))
                    ) <= 20:
                        audit.append(
                            dict(joined_gap_pixels=gap, varying_bridge=bool(varying_bridge))
                        )
                        merged[-1] = np.arange(previous[0], run[-1] + 1, dtype=np.intp)
                        continue
                merged.append(run)
            for run in merged:
                first, last = int(run[0]), int(run[-1]) + 1
                if (last - first) / (b - a) < 4:
                    continue
                lo, hi = max(0, a - 3), min(width, b + 3)
                interior = profile[first + 2 : last - 2]
                delta = interior - background
                denominator = np.sum(delta**2, axis=1)
                useful = denominator > 100
                if not np.any(useful):
                    continue
                columns = crop[first + 2 : last - 2, lo:hi].astype(float) - background
                alpha = np.median(
                    np.sum(columns[useful] * delta[useful, None], axis=2)
                    / denominator[useful, None],
                    axis=0,
                )
                covered = np.flatnonzero((alpha > 0.5) & (alpha < 1.3))
                if not len(covered):
                    continue
                ca, cb = lo + int(covered[0]), lo + int(covered[-1]) + 1
                if cb - ca < 2:
                    continue
                p = np.median(crop[:, ca:cb], axis=1).astype(float)
                # At either endpoint use an interior colour reference to locate
                # the half-height transition, tolerating blur halos and ringing.
                for beginning in (True, False):
                    reference = np.median(
                        p[first + 3 : min(last, first + 9)]
                        if beginning
                        else p[max(first, last - 9) : last - 3],
                        axis=0,
                    )
                    d = reference - background
                    denom = float(d @ d)
                    if denom <= 100:
                        continue
                    low, high = (
                        (max(0, first - 4), min(last, first + 5))
                        if beginning
                        else (max(first, last - 5), min(length, last + 4))
                    )
                    fractions = (p[low:high] - background) @ d / denom
                    present = np.flatnonzero(fractions > 0.5)
                    if len(present):
                        if beginning:
                            first = low + int(present[0])
                        else:
                            last = low + int(present[-1]) + 1
                # Constant endcap strokes are not the colour-bearing core.
                # Trim only bounded caps with an abrupt colour discontinuity;
                # smoothly varying grayscale endpoints are preserved.
                for _ in range(3):
                    if last - first < 24:
                        break
                    if np.max(p[first]) <= 20 and np.max(np.abs(p[first] - p[first + 1])) > 20:
                        first += 1
                    elif (
                        first + 2 < last
                        and np.max(p[first : first + 2]) <= 20
                        and np.max(np.abs(p[first + 1] - p[first + 2])) > 20
                    ):
                        first += 2
                    else:
                        break
                for _ in range(3):
                    if last - first < 24:
                        break
                    if np.max(p[last - 1]) <= 20 and np.max(np.abs(p[last - 1] - p[last - 2])) > 20:
                        last -= 1
                    elif (
                        last - 3 >= first
                        and np.max(p[last - 2 : last]) <= 20
                        and np.max(np.abs(p[last - 2] - p[last - 3])) > 20
                    ):
                        last -= 2
                    else:
                        break
                bbox = (
                    (x + ca, y + first, x + cb, y + last)
                    if vertical
                    else (x + first, y + ca, x + last, y + cb)
                )
                if min(bbox[2] - bbox[0], bbox[3] - bbox[1]) >= 2:
                    found.append((bbox, "vertical" if vertical else "horizontal"))
    unique = tuple(dict.fromkeys(found))
    return unique, dict(
        version=CORE_SEARCH_VERSION,
        window=window,
        search_bbox=(x, y, r, s),
        orientations=audit,
        cores=[dict(bbox=b, orientation=o) for b, o in unique],
        truncated=truncated,
    )
