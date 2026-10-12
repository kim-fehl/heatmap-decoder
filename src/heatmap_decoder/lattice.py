"""Robust lattice observations, with colour evidence excluding measured strokes."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .models import Axis, BoundingBox, RGBArray


@dataclass(frozen=True)
class AxisObservations:
    transitions: tuple[float, ...]
    transition_widths: tuple[float, ...]
    transition_support: tuple[float, ...]
    strokes: tuple[float, ...]


def observe_axis(image: RGBArray, box: BoundingBox, axis: Axis) -> AxisObservations:
    x, y, r, s = box
    crop = image[y:s, x:r, :3]
    if axis == "columns":
        crop = crop.transpose(1, 0, 2)
    pixels = crop[:, :: max(1, crop.shape[1] // 32)].astype(float)
    origin = y if axis == "rows" else x
    dark = np.max(pixels, axis=2) < 48
    stroke_rows = np.mean(dark, axis=1) >= 0.85
    bands = []
    strokes = []
    for group in _groups(np.flatnonzero(stroke_rows)):
        a, b = int(group[0]), int(group[-1]) + 1
        if b - a > 3 or a == 0 or b == len(pixels):
            continue
        before = np.median(pixels[max(0, a - 3) : a], axis=0)
        after = np.median(pixels[b : min(len(pixels), b + 3)], axis=0)
        if min(np.mean(np.abs(pixels[a:b] - before)), np.mean(np.abs(pixels[a:b] - after))) <= 8:
            continue
        bands.append((a, b))
        strokes.append(float(origin + (a + b) / 2 + 0.5))
    # Perpendicular strokes are independently removed from colour samples.
    colours = pixels[:, np.mean(dark, axis=0) < 0.85]
    positions, widths, supports = [], [], []
    if colours.shape[1] and len(colours) > 1:
        differences = np.max(np.abs(np.diff(colours, axis=0)), axis=2)
        gradient = np.mean(differences, axis=1)
        noise = float(np.median(gradient))
        threshold = max(8.0, 4 * noise)
        cross_support = np.mean(differences > max(6.0, 2 * noise), axis=1)
        for a, b in bands:
            gradient[max(0, a - 1) : min(len(gradient), b)] = 0
            before = np.median(colours[max(0, a - 3) : a], axis=0)
            after = np.median(colours[b : min(len(colours), b + 3)], axis=0)
            jumps = np.max(np.abs(before - after), axis=1)
            support = float(np.mean(jumps > 8))
            if np.mean(jumps) > 8 and support >= 0.4:
                positions.append(float(origin + (a + b) / 2 + 0.5))
                widths.append(float(b - a))
                supports.append(support)
        for group in _groups(np.flatnonzero((gradient > threshold) & (cross_support >= 0.4))):
            # Broad peaks are preserved, never sharpened to fake a precise boundary.
            if len(group) > 7:
                continue
            position = origin + float(np.average(group + 1, weights=gradient[group]))
            if position <= origin + 2 or position >= origin + len(pixels) - 2:
                continue  # Outer stroke/halo observations belong to extent, not internal lattice.
            positions.append(position)
            widths.append(float(len(group)))
            supports.append(float(np.mean(cross_support[group])))
    ordered = sorted(zip(positions, widths, supports, strict=True))
    return AxisObservations(
        tuple(p for p, w, c in ordered),
        tuple(w for p, w, c in ordered),
        tuple(c for p, w, c in ordered),
        tuple(strokes),
    )


def _groups(indices: NDArray[np.intp]) -> list[NDArray[np.intp]]:
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)) if len(indices) else []
