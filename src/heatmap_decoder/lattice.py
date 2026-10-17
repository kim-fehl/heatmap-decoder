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
    transition_uncertainties: tuple[float, ...] = ()


def _stroke_centres(pixels: NDArray[np.float64]) -> list[tuple[int, float]]:
    """Locate narrow dark troughs corroborated across the transverse samples.

    A blurred stroke is a trough between two independently measured cell
    interiors; its centre dispersion is distinct from its visible width.
    """
    candidates = []
    for i in range(3, len(pixels) - 3):
        before = np.median(pixels[i - 3 : i - 1], axis=0)
        after = np.median(pixels[i + 2 : i + 4], axis=0)
        centre = pixels[i]
        depression = np.minimum(np.mean(before - centre, axis=1), np.mean(after - centre, axis=1))
        supported = depression > 12
        if np.mean(supported) < 0.85:
            continue
        if np.median(np.max(centre, axis=1)) > 0.75 * min(
            np.median(np.max(before, axis=1)), np.median(np.max(after, axis=1))
        ):
            continue
        candidates.append(i)
    found = []
    for group in _groups(np.array(candidates, dtype=np.intp)):
        if len(group) > 3:
            continue
        i = int(group[np.argmin(np.mean(pixels[group], axis=(1, 2)))])
        # Per-sample quadratic trough localisation, bounded to half a pixel.
        a, b, c = (np.mean(pixels[j], axis=1) for j in (i - 1, i, i + 1))
        curvature = a - 2 * b + c
        usable = (curvature > 12) & (b < a) & (b < c)
        if np.mean(usable) < 0.8:
            continue
        shifts = np.clip((a[usable] - c[usable]) / (2 * curvature[usable]), -0.5, 0.5)
        uncertainty = 0.5 + float(np.quantile(np.abs(shifts), 0.9))
        found.append((i, uncertainty))
    return found


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
    # Exclude only thin, context-supported perpendicular strokes, using their
    # ORIGINAL pixel width. A broad dark sample band is data, not a stroke.
    transverse = crop.transpose(1, 0, 2).astype(float)
    excluded = np.zeros(crop.shape[1], dtype=bool)
    transverse_dark = np.mean(np.max(transverse, axis=2) < 48, axis=1) >= 0.85
    for group in _groups(np.flatnonzero(transverse_dark)):
        a, b = int(group[0]), int(group[-1]) + 1
        if b - a > 3 or a == 0 or b == len(transverse):
            continue
        before = np.median(transverse[max(0, a - 3) : a], axis=0)
        after = np.median(transverse[b : min(len(transverse), b + 3)], axis=0)
        if (
            min(np.mean(np.abs(transverse[a:b] - before)), np.mean(np.abs(transverse[a:b] - after)))
            > 8
        ):
            excluded[a:b] = True
    step = max(1, crop.shape[1] // 32)
    colours = pixels[:, ~excluded[::step]]
    positions, widths, supports, uncertainties = [], [], [], []
    if colours.shape[1] and len(colours) > 1:
        differences = np.max(np.abs(np.diff(colours, axis=0)), axis=2)
        gradient = np.mean(differences, axis=1)
        noise = float(np.median(gradient))
        threshold = max(8.0, 4 * noise)
        cross_support = np.mean(differences > max(6.0, 2 * noise), axis=1)
        measured_strokes = _stroke_centres(colours)
        for centre, uncertainty in measured_strokes:
            before = np.median(colours[max(0, centre - 5) : centre - 2], axis=0)
            after = np.median(colours[centre + 3 : min(len(colours), centre + 6)], axis=0)
            jumps = np.max(np.abs(before - after), axis=1)
            support = float(np.mean(jumps > 8))
            measured_width = float(
                max(
                    1,
                    int(
                        np.count_nonzero(
                            gradient[max(0, centre - 3) : min(len(gradient), centre + 3)]
                            > threshold
                        )
                    ),
                )
            )
            gradient[max(0, centre - 3) : min(len(gradient), centre + 3)] = 0
            bands = [(u, v) for u, v in bands if not u <= centre < v]
            strokes.append(float(origin + centre + 1))
            if np.mean(jumps) > 8 and support >= 0.4:
                positions.append(float(origin + centre + 1))
                widths.append(measured_width)
                supports.append(support)
                uncertainties.append(uncertainty)
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
                uncertainties.append(float(b - a) / 2)
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
            uncertainties.append(float(len(group)) / 2)
    ordered = sorted(zip(positions, widths, supports, uncertainties, strict=True))
    return AxisObservations(
        tuple(p for p, w, c, u in ordered),
        tuple(w for p, w, c, u in ordered),
        tuple(c for p, w, c, u in ordered),
        tuple(sorted(set(strokes))),
        tuple(u for p, w, c, u in ordered),
    )


def _groups(indices: NDArray[np.intp]) -> list[NDArray[np.intp]]:
    return list(np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)) if len(indices) else []
