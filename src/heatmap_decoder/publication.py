"""Raster heatmap decoder for regular grids and line/rectangle annotations.

Coordinates are image pixels; boxes and column spans are end-exclusive.
Values recovered from colors are estimates in displayed scale units, not GEO data.
"""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

cv2 = importlib.import_module("cv2")
pytesseract = importlib.import_module("pytesseract")
find_peaks = importlib.import_module("scipy.signal").find_peaks
cKDTree = importlib.import_module("scipy.spatial").cKDTree


class DecodeError(ValueError):
    """The image does not supply enough evidence for a supported extraction."""


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def json(self) -> list[int]:
        return [int(self.x), int(self.y), int(self.right), int(self.bottom)]


def _components(mask: Any) -> list[Box]:
    return [
        Box(*map(int, s[:4]))
        for s in cv2.connectedComponentsWithStats(mask.astype(np.uint8))[2][1:]
    ]


def _overlap(a: Box, b: Box) -> bool:
    return a.x < b.right and b.x < a.right and a.y < b.bottom and b.y < a.bottom


def _regions(rgb: Any) -> tuple[Box, list[Box]]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 1] > 15) & (hsv[:, :, 2] > 80)).astype(np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    boxes = _components(closed)
    large = [b for b in boxes if b.w > 50 and b.h > 50]
    if not large:
        raise DecodeError("No large colored heatmap region found")
    matrix = max(large, key=lambda b: b.w * b.h)
    # Closing at an image edge can fill the small white margin below a matrix.
    # Refine to the original colored footprint before fitting periodic spacing.
    footprint = mask[matrix.y : matrix.bottom, matrix.x : matrix.right]
    yy = np.where(footprint.sum(axis=1) > max(1, matrix.w * 0.02))[0]
    xx = np.where(footprint.sum(axis=0) > max(1, matrix.h * 0.02))[0]
    if len(xx) and len(yy):
        matrix = Box(
            matrix.x + int(xx[0]),
            matrix.y + int(yy[0]),
            int(xx[-1] - xx[0] + 1),
            int(yy[-1] - yy[0] + 1),
        )
    # A zero/white segment can split a diverging colorbar into two components.
    outside = [b for b in boxes if not _overlap(b, matrix) and b.w * b.h > 150]
    merged = []
    while outside:
        a = outside.pop(0)
        changed = True
        while changed:
            changed = False
            for b in outside[:]:
                horizontal = (
                    abs(a.y - b.y) <= 2
                    and abs(a.h - b.h) <= 3
                    and min(abs(a.right - b.x), abs(b.right - a.x)) <= max(a.h, b.h)
                )
                vertical = (
                    abs(a.x - b.x) <= 2
                    and abs(a.w - b.w) <= 3
                    and min(abs(a.bottom - b.y), abs(b.bottom - a.y)) <= max(a.w, b.w)
                )
                if horizontal or vertical:
                    x, y = min(a.x, b.x), min(a.y, b.y)
                    a = Box(x, y, max(a.right, b.right) - x, max(a.bottom, b.bottom) - y)
                    outside.remove(b)
                    changed = True
        merged.append(a)
    bars = [
        b
        for b in merged
        if max(b.w, b.h) / min(b.w, b.h) >= 3
        and not (
            b.h >= matrix.h * 0.8
            and abs(b.y - matrix.y) < 20
            and 0 <= matrix.x - b.right < max(30, matrix.w * 0.05)
        )
    ]
    return matrix, bars


def infer_grid(profile: Any, length: int) -> tuple[int, float]:
    """Fit a periodic grid using both edge support and coverage (avoid harmonics)."""
    peaks, _ = find_peaks(profile, prominence=max(0.6, float(profile.max()) * 0.035), distance=3)
    if len(peaks) < 2:
        raise DecodeError("Not enough cell transitions to infer a grid")
    weights = profile[peaks]
    candidates = []
    for n in range(2, min(1000, length // 4) + 1):
        pitch = length / n
        boundary = np.rint((peaks + 1) / pitch).astype(int)
        hit = (
            (abs(peaks + 1 - boundary * pitch) <= min(2, pitch * 0.15))
            & (boundary > 0)
            & (boundary < n)
        )
        precision = float(weights[hit].sum() / weights.sum())
        coverage = len(set(boundary[hit])) / (n - 1)
        score = 2 * precision * coverage / (precision + coverage + 1e-9)
        candidates.append((score, n))
    score, n = max(candidates)
    if score < 0.7:
        raise DecodeError(
            f"Ambiguous grid (best support {score:.2f}); provide higher resolution or sample counts"
        )
    return n, float(score)


def _ocr(rgb: Any, box: Box, psm: int = 11, scale: int = 2) -> list[dict[str, Any]]:
    """OCR words with boxes transformed back into original image coordinates."""
    x0, y0 = max(0, box.x), max(0, box.y)
    x1, y1 = min(rgb.shape[1], box.right), min(rgb.shape[0], box.bottom)
    if x1 <= x0 or y1 <= y0:
        return []
    crop = cv2.resize(rgb[y0:y1, x0:x1], None, fx=scale, fy=scale)
    data = pytesseract.image_to_data(
        crop, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    out = []
    for i, raw in enumerate(data["text"]):
        raw = raw.strip()
        if not raw:
            continue
        b = Box(
            round(x0 + data["left"][i] / scale),
            round(y0 + data["top"][i] / scale),
            max(1, round(data["width"][i] / scale)),
            max(1, round(data["height"][i] / scale)),
        )
        out.append({"text": raw, "bbox": b.json(), "confidence": float(data["conf"][i])})
    return out


def _runs(positions: Any, gap: int = 1) -> list[Any]:
    if len(positions) == 0:
        return []
    return np.split(positions, np.where(np.diff(positions) > gap)[0] + 1)


def _number(crop: Any) -> float | None:
    ink = crop.min(axis=2) < 150
    ys, xs = np.where(ink)
    if not len(xs):
        return None
    crop = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    negative = False
    ink = crop.min(axis=2) < 150
    # Small minus signs sometimes touch the following glyph and OCR drops them.
    # Accept a short leading horizontal stroke only when the glyph is wide enough.
    h, w = ink.shape
    leading = np.where(ink[:, : max(1, w // 5)].any(axis=1))[0]
    if (
        w > 0.8 * h
        and len(leading)
        and leading[-1] - leading[0] + 1 <= 0.3 * h
        and 0.35 * h <= leading.mean() <= 0.8 * h
    ):
        negative = True
    enlarged = cv2.resize(crop, None, fx=5, fy=5)
    enlarged = cv2.copyMakeBorder(
        enlarged, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    for psm in (7, 8):
        text = pytesseract.image_to_string(
            enlarged, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.-"
        ).strip()
        if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            value = float(text)
            return -abs(value) if negative else value
    return None


def _bar_labels(rgb: Any, bar: Box) -> list[dict[str, Any]]:
    vertical = bar.h > bar.w
    if vertical:
        roi = Box(bar.right + 3, bar.y, max(0, min(90, rgb.shape[1] - bar.right - 3)), bar.h)
    else:
        roi = Box(max(0, bar.x - 15), bar.bottom + 1, bar.w + 30, 90)
    roi = Box(roi.x, roi.y, min(roi.w, rgb.shape[1] - roi.x), min(roi.h, rgb.shape[0] - roi.y))
    crop = rgb[roi.y : roi.bottom, roi.x : roi.right]
    if not crop.size:
        return []
    ink = crop.min(axis=2) < 150
    if not vertical:
        # Search separate text bands: ticks and a scale title can both lie
        # between/around the numeric labels. Do not clip labels at a fixed offset.
        candidates: list[list[dict[str, Any]]] = []
        for band in _runs(np.where(ink.any(axis=1))[0]):
            if len(band) < 5:
                continue
            y0, y1 = int(band[0]), int(band[-1] + 1)
            strip = crop[y0:y1]
            line_labels = []
            for run in _runs(
                np.where((strip.min(axis=2) < 150).any(axis=0))[0], gap=max(3, int((y1 - y0) * 0.6))
            ):
                lo, hi = int(run[0]), int(run[-1] + 1)
                value = _number(strip[:, lo:hi])
                if value is not None:
                    b = Box(roi.x + lo, roi.y + y0, hi - lo, y1 - y0)
                    line_labels.append(
                        {
                            "value": value,
                            "bbox": b.json(),
                            "pixel": float(b.x + b.w / 2),
                            "anchor_source": "label_center",
                        }
                    )
            if len(line_labels) >= 2 and len({a["value"] for a in line_labels}) >= 2:
                candidates.append(line_labels)
        labels = max(candidates, key=len) if candidates else []
    else:
        labels = []
    positions = np.where(ink.any(axis=1))[0] if vertical else []
    for run in _runs(positions, gap=3 if vertical else 10):
        lo, hi = int(run[0]), int(run[-1] + 1)
        sub = crop[lo:hi] if vertical else crop[:, lo:hi]
        ys, xs = np.where(sub.min(axis=2) < 150)
        if not len(xs):
            continue
        b = Box(
            roi.x + int(xs.min()) + (0 if vertical else lo),
            roi.y + int(ys.min()) + (lo if vertical else 0),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        )
        value = _number(sub)
        if value is not None:
            labels.append(
                {
                    "value": value,
                    "bbox": b.json(),
                    "pixel": float(b.y + b.h / 2 if vertical else b.x + b.w / 2),
                    "anchor_source": "label_center",
                }
            )
    # Look immediately outside the bar, before any label pixels, for perpendicular ticks.
    if labels:
        if vertical:
            stop = min(b["bbox"][0] for b in labels) - 1
            strip = rgb[bar.y : bar.bottom, bar.right + 1 : stop]
        else:
            stop = min(b["bbox"][1] for b in labels) - 1
            strip = rgb[bar.bottom + 1 : stop, bar.x : bar.right]
        if strip.size:
            black = strip.max(axis=2) < 100
            projection = black.mean(axis=1 if vertical else 0)
            ticks = [
                float(g.mean() + (bar.y if vertical else bar.x))
                for g in _runs(np.where(projection > 0.25)[0])
                if len(g) <= 5
            ]
            for label in labels:
                if ticks:
                    tick = min(ticks, key=lambda t: abs(t - label["pixel"]))
                    if abs(tick - label["pixel"]) <= 6:
                        label["pixel"] = tick
                        label["anchor_source"] = "tick"
    return labels


def calibrate(labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit value = slope * pixel + intercept without assigning bar endpoints."""
    if len(labels) < 2:
        raise DecodeError("At least two readable colorbar values are required")
    pixels = np.array([v["pixel"] for v in labels])
    values = np.array([v["value"] for v in labels])
    if np.ptp(pixels) < 1 or np.ptp(values) == 0:
        raise DecodeError("Degenerate colorbar calibration")
    slope, intercept = np.polyfit(pixels, values, 1)
    residual = float(np.max(abs(values - (slope * pixels + intercept))))
    if residual > 0.08 * np.ptp(values):
        raise DecodeError("Colorbar labels do not support a linear displayed scale")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "max_label_residual": residual,
        "labeled_value_range": [float(values.min()), float(values.max())],
    }


def extend_group_spans(lines: list[Box], edges: Any) -> list[tuple[int, int]]:
    """Recover shared group boundaries from gaps between rules, not text widths."""
    lines = sorted(lines, key=lambda b: b.x)
    pitch = float(np.median(np.diff(edges)))
    if not lines:
        return []
    if lines[0].x - edges[0] > pitch or edges[-1] - lines[-1].right > pitch:
        raise DecodeError("Annotation lines do not establish the outer group spans")
    boundaries = [0]
    for a, b in zip(lines, lines[1:]):
        if b.x - a.right > 2 * pitch or a.right > b.x:
            raise DecodeError("Ambiguous gap or overlap between annotation lines")
        midpoint = (a.right + b.x) / 2
        index = int(np.argmin(abs(edges - midpoint)))
        if abs(edges[index] - midpoint) > 0.35 * pitch:
            raise DecodeError("Group boundary is ambiguous relative to the column grid")
        boundaries.append(index)
    boundaries.append(len(edges) - 1)
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        raise DecodeError("Group lines map to overlapping or empty column spans")
    return [(int(a), int(b)) for a, b in zip(boundaries, boundaries[1:])]


def _line_tracks(
    rgb: Any, matrix: Box, edges: Any, *, rectangle_tracks: Any = ()
) -> list[dict[str, Any]]:
    black = (rgb.max(axis=2) < 100).astype(np.uint8)
    kernel = np.ones((1, max(20, round(matrix.w * 0.035))), np.uint8)
    rules = _components(cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel))
    tolerance = max(5, float(np.median(np.diff(edges))) * 0.5)
    rules = [
        b
        for b in rules
        if b.h <= 10
        and b.w >= kernel.shape[1]
        and b.x >= matrix.x - tolerance
        and b.right <= matrix.right + tolerance
        and (b.bottom <= matrix.y or b.y >= matrix.bottom)
    ]
    # Outlined rectangle borders are already consumed by rectangle detection;
    # they are not extra annotation rows (nor are nearby figure titles labels).
    rectangle_bands = [
        (
            min(s["header_bbox"][1] for s in t["segments"]),
            max(s["header_bbox"][3] for s in t["segments"]),
        )
        for t in rectangle_tracks
    ]
    rules = [b for b in rules if not any(y0 - 5 <= b.y <= y1 + 5 for y0, y1 in rectangle_bands)]
    bands: list[list[Box]] = []
    for b in sorted(rules, key=lambda b: b.y):
        if bands and abs(b.y - bands[-1][0].y) <= 5:
            bands[-1].append(b)
        else:
            bands.append([b])
    words = []
    for roi in (
        Box(matrix.x, 0, matrix.w, matrix.y),
        Box(matrix.x, matrix.bottom, matrix.w, rgb.shape[0] - matrix.bottom),
    ):
        words.extend(_ocr(rgb, roi, scale=1))
    tracks: list[dict[str, Any]] = []
    for band in bands:
        band.sort(key=lambda b: b.x)
        spans = extend_group_spans(band, edges)
        segments = []
        for line, (start, end) in zip(band, spans):
            matched = []
            for word in words:
                x0, y0, x1, y1 = word["bbox"]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                nearest = min(bands, key=lambda other: abs(other[0].y - cy))
                if nearest is band and edges[start] <= cx < edges[end] and abs(line.y - cy) < 70:
                    matched.append(word)
            matched.sort(key=lambda w: w["bbox"][0])
            text = " ".join(w["text"] for w in matched) or None
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "label": text,
                    "raw_text": text,
                    "line_bbox": line.json(),
                    "inferred_span_pixels": [float(edges[start]), float(edges[end])],
                    "ocr_words": matched,
                }
            )
        if any(s["label"] for s in segments):
            tracks.append(
                {
                    "track_id": f"track_{len(tracks) + 1}",
                    "display_name": None,
                    "kind": "line",
                    "segments": segments,
                }
            )
    return tracks


def _row_labels(rgb: Any, matrix: Box, rows: int, bar: Box) -> list[str | None]:
    right = bar.x - 3 if bar.h > bar.w and bar.x > matrix.right else rgb.shape[1]
    words = _ocr(
        rgb, Box(matrix.right + 3, matrix.y, right - matrix.right - 3, matrix.h), psm=6, scale=3
    )
    labels: list[list[dict[str, Any]]] = [[] for _ in range(rows)]
    for word in words:
        y = (word["bbox"][1] + word["bbox"][3]) / 2
        index = int((y - matrix.y) / matrix.h * rows)
        if 0 <= index < rows:
            labels[index].append(word)
    return [
        " ".join(w["text"] for w in sorted(line, key=lambda w: w["bbox"][0])) or None
        for line in labels
    ]


def _panel_layout(rgb: Any) -> Any:
    """Find colored header rectangles paired with separate tall matrix blocks."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    boxes = _components((hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 80))
    headers = [b for b in boxes if b.w > 40 and 8 <= b.h <= 50 and b.w / b.h > 3]
    bands: list[list[Box]] = []
    for b in sorted(headers, key=lambda b: b.y):
        if bands and abs(b.y - bands[-1][0].y) < 5:
            bands[-1].append(b)
        else:
            bands.append([b])
    layouts = []
    for band in bands:
        if len(band) < 2:
            continue
        pairs = []
        for header in sorted(band, key=lambda b: b.x):
            matches = [
                b
                for b in boxes
                if b.h > 100
                and 0 <= b.y - header.bottom < 15
                and abs(b.x - header.x) <= 5
                and abs(b.right - header.right) <= 5
            ]
            if matches:
                pairs.append((header, max(matches, key=lambda b: b.h)))
        if len(pairs) == len(band):
            layouts.append(pairs)
    return max(layouts, key=lambda pairs: sum(b.w * b.h for _, b in pairs)) if layouts else None


def _rectangle_words(rgb: Any, box: Box) -> tuple[list[dict[str, Any]], list[int], str]:
    """Isolate text by contrast with the fill, independent of text polarity."""
    crop = rgb[box.y : box.bottom, box.x : box.right]
    background = np.median(crop, axis=(0, 1))
    contrast = np.linalg.norm(crop.astype(float) - background, axis=2)
    dark = (crop.max(axis=2) < 120) & (contrast > 70)
    light = (crop.min(axis=2) > 200) & (contrast > 70)
    polarity = "light" if light.sum() > dark.sum() else "dark"
    # A contrast-only mask also admits JPEG fringes and neighboring panel
    # colors. Restrict it to the observed text polarity before OCR.
    foreground = light if polarity == "light" else dark
    text_mask = np.where(foreground, 0, 255).astype(np.uint8)
    text_mask = cv2.copyMakeBorder(text_mask, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    prepared = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2RGB)
    words = _ocr(prepared, Box(0, 0, prepared.shape[1], prepared.shape[0]), psm=11, scale=3)
    for word in words:
        word["bbox"] = [
            value + (box.x - 8 if index % 2 == 0 else box.y - 8)
            for index, value in enumerate(word["bbox"])
        ]
    return words, background.astype(int).tolist(), polarity


def _rectangle_tracks(rgb: Any, matrix: Box, edges: Any) -> list[dict[str, Any]]:
    """Find stacked filled or outlined rectangles above a regular matrix.

    Quantized flat-fill components include gray, black and white, which HSV
    saturation detection alone would miss. White outlined cells are separate
    components bounded by their printed borders.
    """
    crop = rgb[: matrix.y, matrix.x : matrix.right]
    if not crop.size:
        return []
    codes = (
        (crop[:, :, 0] // 16).astype(int) * 256
        + (crop[:, :, 1] // 16).astype(int) * 16
        + crop[:, :, 2] // 16
    )
    colors, counts = np.unique(codes, return_counts=True)
    rectangles = []
    for color, count in zip(colors, counts):
        if count < 400:
            continue
        mask = (codes == color).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, w, h, area in stats[1:]:
            if w < 20 or not 12 <= h <= 120 or area / (w * h) < 0.65 or y == 0 or y + h >= matrix.y:
                continue
            rectangles.append(Box(matrix.x + int(x), int(y), int(w), int(h)))
    bands: list[list[Box]] = []
    for b in sorted(rectangles, key=lambda b: b.y):
        match = next(
            (
                band
                for band in bands
                if min(b.bottom, band[0].bottom) - max(b.y, band[0].y) > 0.7 * min(b.h, band[0].h)
            ),
            None,
        )
        if match is None:
            bands.append([b])
        else:
            match.append(b)
    tracks: list[dict[str, Any]] = []
    pitch = float(np.median(np.diff(edges)))
    for band in bands:
        band.sort(key=lambda b: b.x)
        if len(band) < 2 or band[0].x - matrix.x > pitch or matrix.right - band[-1].right > pitch:
            continue
        # Candidate headers must form a partition; unrelated title/text blobs
        # cannot be promoted into sample groups.
        if any(abs(a.right - b.x) > pitch * 0.7 for a, b in zip(band, band[1:])):
            continue
        boundaries = (
            [0]
            + [int(np.argmin(abs(edges - (a.right + b.x) / 2))) for a, b in zip(band, band[1:])]
            + [len(edges) - 1]
        )
        if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
            continue
        segments = []
        for box, start, end in zip(band, boundaries, boundaries[1:]):
            words, color, polarity = _rectangle_words(rgb, box)
            text = " ".join(w["text"] for w in words) or None
            segments.append(
                {
                    "start": start,
                    "end": end,
                    "label": text,
                    "raw_text": text,
                    "header_bbox": box.json(),
                    "color_rgb": color,
                    "text_polarity": polarity,
                    "ocr_words": words,
                    "inferred_span_pixels": [float(edges[start]), float(edges[end])],
                }
            )
        if all(s["label"] for s in segments):
            tracks.append(
                {
                    "track_id": f"track_{len(tracks) + 1}",
                    "display_name": None,
                    "kind": "colored_panel",
                    "segments": segments,
                }
            )
    return tracks


def _decode_panels(
    rgb: Any, path: str | Path, layout: Any, expected_columns: int | None
) -> dict[str, Any]:
    headers, blocks = zip(*layout)
    matrix = Box(
        min(b.x for b in blocks),
        min(b.y for b in blocks),
        max(b.right for b in blocks) - min(b.x for b in blocks),
        max(b.bottom for b in blocks) - min(b.y for b in blocks),
    )
    # The white gaps between blocks are layout separators, never samples.
    block = max(blocks, key=lambda b: b.w)
    sub = rgb[block.y : block.bottom, block.x : block.right].astype(float)
    rows, row_score = infer_grid(np.abs(np.diff(sub, axis=0)).mean(axis=(1, 2)), block.h)
    _, bars = _regions(rgb)
    if not bars:
        raise DecodeError("No colorbar found beside colored-panel heatmap")
    bar = max(bars, key=lambda b: b.w * b.h)
    anchors = _bar_labels(rgb, bar)
    calibration = calibrate(anchors)
    segments = []
    for header, block in layout:
        words, color, polarity = _rectangle_words(rgb, header)
        text = " ".join(w["text"] for w in words) or None
        segments.append(
            {
                "start": None,
                "end": None,
                "label": text,
                "raw_text": text,
                "header_bbox": header.json(),
                "matrix_block_bbox": block.json(),
                "color_rgb": color,
                "text_polarity": polarity,
                "ocr_words": words,
            }
        )
    last = headers[-1]
    title = _ocr(rgb, Box(last.right + 2, last.y, 150, last.h + 5), psm=7, scale=2)
    name = " ".join(w["text"] for w in title) or None
    return {
        "image": str(Path(path)),
        "status": "resolution_limited",
        "matrix_bbox": matrix.json(),
        "matrix_blocks": [b.json() for b in blocks],
        "n_rows": rows,
        "n_columns": None,
        "expected_n_columns": expected_columns,
        "sample_count_source": "caller_provided_metadata" if expected_columns is not None else None,
        "grid_support": {"rows": row_score, "columns": None},
        "row_edges": np.linspace(matrix.y, matrix.bottom, rows + 1).tolist(),
        "column_edges": None,
        "columns": None,
        "row_labels": _row_labels(rgb, matrix, rows, bar),
        "column_annotations": [
            {
                "track_id": "track_1",
                "display_name": name,
                "kind": "colored_panel",
                "segments": segments,
            }
        ],
        "colorbar": {
            "bbox": bar.json(),
            "orientation": "vertical" if bar.h > bar.w else "horizontal",
            "anchors": anchors,
            "calibration": calibration,
        },
        "reconstructed_values": None,
        "warnings": [
            "Colored panels and their text were recovered directly without a separate legend.",
            "Individual column counts and per-sample values are unresolved in this dense raster; "
            "pixel widths are not sample counts.",
            "Expected sample count is external metadata, not an image-derived measurement.",
            "OCR text may contain substitutions.",
        ],
    }


def _values(
    rgb: Any, row_edges: Any, col_edges: Any, bar: Box, calibration: dict[str, Any]
) -> tuple[list[list[float]], dict[str, Any]]:
    if bar.h > bar.w:
        palette = np.median(rgb[bar.y : bar.bottom, bar.x + 2 : bar.right - 2], axis=1)
        coordinates = np.arange(bar.y, bar.bottom)
    else:
        palette = np.median(rgb[bar.y + 2 : bar.bottom - 2, bar.x : bar.right], axis=0)
        coordinates = np.arange(bar.x, bar.right)
    colors = []
    for y0, y1 in zip(row_edges, row_edges[1:]):
        for x0, x1 in zip(col_edges, col_edges[1:]):
            # Sample cell interiors, excluding printed grid lines and antialiased edges.
            xa, xb = (
                int(x0 + 0.25 * (x1 - x0)),
                max(int(x0 + 0.75 * (x1 - x0)), int(x0 + 0.25 * (x1 - x0)) + 1),
            )
            ya, yb = (
                int(y0 + 0.25 * (y1 - y0)),
                max(int(y0 + 0.75 * (y1 - y0)), int(y0 + 0.25 * (y1 - y0)) + 1),
            )
            colors.append(np.median(rgb[ya:yb, xa:xb], axis=(0, 1)))
    distance, index = cKDTree(palette).query(np.array(colors))
    values = calibration["slope"] * coordinates[index] + calibration["intercept"]
    shape = (len(row_edges) - 1, len(col_edges) - 1)
    return values.reshape(shape).tolist(), {
        "median_rgb_distance": float(np.median(distance)),
        "max_rgb_distance": float(distance.max()),
        "cells_outside_labeled_range": int(
            (
                (values < calibration["labeled_value_range"][0])
                | (values > calibration["labeled_value_range"][1])
            ).sum()
        ),
    }


def decode(path: str | Path, *, expected_columns: int | None = None) -> dict[str, Any]:
    if expected_columns is not None and (
        not isinstance(expected_columns, int) or expected_columns < 1
    ):
        raise DecodeError("expected_columns must be a positive integer")
    rgb = np.array(Image.open(path).convert("RGB"))
    layout = _panel_layout(rgb)
    if layout:
        return _decode_panels(rgb, path, layout, expected_columns)
    matrix, bars = _regions(rgb)
    if not bars:
        raise DecodeError("No separate colorbar detected")
    sub = rgb[matrix.y : matrix.bottom, matrix.x : matrix.right].astype(float)
    rows, row_score = infer_grid(np.abs(np.diff(sub, axis=0)).mean(axis=(1, 2)), matrix.h)
    cols, col_score = infer_grid(np.abs(np.diff(sub, axis=1)).mean(axis=(0, 2)), matrix.w)
    if expected_columns is not None and expected_columns != cols:
        raise DecodeError(
            f"Image grid ({cols}) conflicts with expected sample count ({expected_columns})"
        )
    row_edges = np.linspace(matrix.y, matrix.bottom, rows + 1)
    col_edges = np.linspace(matrix.x, matrix.right, cols + 1)
    bar = max(bars, key=lambda b: b.w * b.h)
    anchors = _bar_labels(rgb, bar)
    calibration = calibrate(anchors)
    values, quality = _values(rgb, row_edges, col_edges, bar, calibration)
    rectangles = _rectangle_tracks(rgb, matrix, col_edges)
    tracks = rectangles + _line_tracks(rgb, matrix, col_edges, rectangle_tracks=rectangles)
    for index, track in enumerate(tracks):
        track["track_id"] = f"track_{index + 1}"
    return {
        "image": str(Path(path)),
        "matrix_bbox": matrix.json(),
        "n_rows": rows,
        "n_columns": cols,
        "grid_support": {"rows": row_score, "columns": col_score},
        "row_edges": row_edges.tolist(),
        "column_edges": col_edges.tolist(),
        "columns": [
            {"column_id": f"column_{i + 1:03d}", "display_label": None, "geo_sample_id": None}
            for i in range(cols)
        ],
        "row_labels": _row_labels(rgb, matrix, rows, bar),
        "column_annotations": tracks,
        "colorbar": {
            "bbox": bar.json(),
            "orientation": "vertical" if bar.h > bar.w else "horizontal",
            "anchors": anchors,
            "calibration": calibration,
        },
        "reconstructed_values": values,
        "value_quality": quality,
        "warnings": [
            "Values are estimates from image colors in displayed scale units; "
            "not validated against deposited expression data.",
            "OCR text may contain character substitutions; inspect raw text and boxes.",
            "Sample display labels and GEO identities are unresolved; internal IDs are positional.",
        ],
    }
