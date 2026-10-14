"""Pillow rendering of sampled colours, nearest candidates and categorical QC."""

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw

from .geometry import axis_slots, resolve_grid
from .models import CellFlag, ExtractionResult, LabelExtractionResult, OCRLabelSource, RGBArray

if TYPE_CHECKING:
    from .models import GridConfig, LegendResult


def diagnostic_image(result: ExtractionResult) -> Image.Image:
    """Compare source sampling to matched bar colours, with an explicit QC panel.

    Missing colours are grey/crossed. The middle panel shows nearest candidates,
    including ambiguous candidates whose exported numerical values are NaN.
    """
    rows, columns = result.values.shape
    size, gap, header, footer = 18, 20, 35, 55
    width, height = columns * size, rows * size
    panel_width = max(width, 220)
    canvas = Image.new("RGB", (3 * panel_width + 4 * gap, height + header + footer), "white")
    draw = ImageDraw.Draw(canvas)
    titles = ["Source sampling", "Nearest bar candidate", "QC flags"]
    for panel, title in enumerate(titles):
        x0 = gap + panel * (panel_width + gap)
        draw.text((x0, 10), title, fill="black")
        for row in range(rows):
            for column in range(columns):
                flag = int(result.flags[row, column])
                if panel == 2:
                    colour = (
                        (80, 80, 80)
                        if flag & CellFlag.UNCALIBRATED
                        else (150, 150, 150)
                        if flag & CellFlag.MISSING
                        else (180, 70, 210)
                        if flag & CellFlag.AMBIGUOUS
                        else (240, 135, 35)
                        if flag & CellFlag.POOR_MATCH
                        else (70, 170, 90)
                    )
                else:
                    rgb = (result.sampled_rgb if panel == 0 else result.reconstructed_rgb)[
                        row, column
                    ]
                    colour = (
                        (int(round(rgb[0])), int(round(rgb[1])), int(round(rgb[2])))
                        if np.isfinite(rgb).all()
                        else (150, 150, 150)
                    )
                x, y = x0 + column * size, header + row * size
                draw.rectangle((x, y, x + size - 1, y + size - 1), fill=colour)
                if panel == 2 and flag & CellFlag.UNCALIBRATED:
                    draw.text((x + 3, y + 2), "U", fill="white")
                if flag & CellFlag.MISSING:
                    draw.line((x, y, x + size - 1, y + size - 1), fill="black")
                # Outline poor matches even when ambiguity takes colour priority.
                if panel == 2 and flag & CellFlag.POOR_MATCH:
                    draw.rectangle((x, y, x + size - 1, y + size - 1), outline="black", width=2)
    draw.text(
        (gap, header + height + 8),
        "QC: green OK; orange poor; purple ambiguous; grey/cross missing; U uncalibrated",
        fill="black",
    )
    draw.text(
        (gap, header + height + 25),
        "Black QC outline: poor match. Colour distance is not numerical uncertainty.",
        fill="black",
    )
    return canvas


def labels_diagnostic_image(image: RGBArray, result: LabelExtractionResult) -> Image.Image:
    """Original-coordinate ROI, centre, token and candidate overlay plus slot ledger."""
    from .diagnostic_style import DiagnosticCanvas

    count = sum(len(a.tokens) + len(a.candidates) + 2 * len(a.slots) for a in result.axes)
    extra = sum(
        len(a.geometry_evidence.get("lines", [])) if a.geometry_evidence else 0 for a in result.axes
    )
    canvas = DiagnosticCanvas(
        Image.fromarray(image),
        lines=count + result.config.rows + result.config.columns + extra + 24,
    )
    canvas.roi(result.config.heatmap_bbox, "heatmap", "manual")
    grid = resolve_grid(result.config)
    gx, gy, gr, gb = grid.bbox
    for name in ("rows", "columns"):
        axis = getattr(grid, name)
        for start, end in axis.gaps:
            gap_box = (gx, start, gr, end) if name == "rows" else (start, gy, end, gb)
            canvas.roi(gap_box, "heatmap", "manual", f"{name} gap")
        for slot in axis.slots:
            cell_box = (
                (gx, slot.start, gr, slot.end) if name == "rows" else (slot.start, gy, slot.end, gb)
            )
            canvas.roi(cell_box, "heatmap", "manual", f"{name} {slot.index}")
    canvas.note(f"Resolved grid: {grid.metadata()['sha256']}")
    settings = result.config.labels.ocr if result.config.labels is not None else None
    minimum, high = (settings.min_confidence, settings.high_confidence) if settings else (80, 90)
    for axis in result.axes:
        if axis.geometry_evidence:
            for line in axis.geometry_evidence.get("lines", []):
                polygon = line.get("polygon", ())
                if polygon:
                    canvas.draw.line([*polygon, polygon[0]], fill="#007e80", width=2)
                baseline = line.get("baseline", ())
                if baseline:
                    canvas.draw.line(baseline, fill="#9370bb", width=2)
                attachment = line.get("attachment")
                if attachment:
                    ax, ay = attachment
                    canvas.draw.ellipse(
                        (ax - 3, ay - 3, ax + 3, ay + 3), outline="#c52647", width=2
                    )
                canvas.note(
                    f"Line {line['index']} -> column {line.get('slot_index')} {line.get('issues')}"
                )
            canvas.note(
                f"Label geometry: {axis.geometry_evidence.get('status')}; "
                "identifiers require review"
            )
        roi = getattr(result.config.labels, axis.axis) if result.config.labels else None
        if isinstance(roi, OCRLabelSource):
            expanded = (
                roi.bbox[0],
                roi.bbox[1],
                min(image.shape[1], roi.bbox[2] + roi.crop_right_padding_pixels),
                roi.bbox[3],
            )
            canvas.roi(expanded, f"labels_{axis.axis}", "manual")
        if isinstance(roi, OCRLabelSource):
            for slot in axis_slots(result.config, axis.axis):
                box = (
                    (
                        roi.bbox[0],
                        max(roi.bbox[1], slot.start),
                        min(image.shape[1], roi.bbox[2] + roi.crop_right_padding_pixels),
                        min(roi.bbox[3], slot.end),
                    )
                    if axis.axis == "rows"
                    else (
                        max(roi.bbox[0], slot.start),
                        roi.bbox[1],
                        min(roi.bbox[2], slot.end),
                        roi.bbox[3],
                    )
                )
                if box[2] > box[0] and box[3] > box[1]:
                    canvas.roi(box, f"labels_{axis.axis}", "manual", f"{axis.axis}-F{slot.index}")
            canvas.note(
                "Fields follow equal grid pitch; raw glyph boxes are separate observations."
            )
        for index, token in enumerate(axis.tokens):
            if token.text.strip():
                canvas.ocr(
                    token.bbox,
                    token.confidence,
                    f"{axis.axis}-T{index}",
                    token.text,
                    "raw",
                    minimum,
                    high,
                )
        for index, candidate in enumerate(axis.candidates):
            labels = [s for s in axis.slots if index in s.candidate_indices]
            status = next(
                (s.status for s in labels if s.status == "ambiguous"),
                labels[0].status if labels else "unmatched",
            )
            canvas.ocr(
                candidate.bbox,
                candidate.confidence,
                f"{axis.axis}-C{index}",
                candidate.text,
                status,
                minimum,
                high,
            )
        for slot, label in zip(axis_slots(result.config, axis.axis), axis.slots, strict=True):
            canvas.note(
                f"{axis.axis}-S{label.index} [{label.status}] "
                f"{(label.raw_text or label.final_text or 'missing').replace(chr(10), ' | ')} "
                f"{','.join(label.issues)}"
            )
            if label.status == "missing":
                x, y = (
                    (result.config.heatmap_bbox[0], slot.centre)
                    if axis.axis == "rows"
                    else (slot.centre, result.config.heatmap_bbox[1])
                )
                canvas.draw.line((x - 3, y - 3, x + 3, y + 3), fill="black", width=2)
                canvas.draw.line((x - 3, y + 3, x + 3, y - 3), fill="black", width=2)
    return canvas.image


def legend_fields_diagnostic_image(image: RGBArray, result: "LegendResult") -> Image.Image:
    """Enlarge actual tick OCR fields, even when no text/value was recognised."""
    from .diagnostic_style import DiagnosticCanvas

    boxes = [result.config.colourbar_bbox] + [t.ocr_bbox for t in result.ticks if t.ocr_bbox]
    x = max(0, min(b[0] for b in boxes) - 8)
    y = max(0, min(b[1] for b in boxes) - 8)
    r = min(image.shape[1], max(b[2] for b in boxes) + 8)
    s = min(image.shape[0], max(b[3] for b in boxes) + 8)
    source = Image.fromarray(image[:, :, :3]).crop((x, y, r, s))
    source = source.resize((source.width * 4, source.height * 4), Image.Resampling.NEAREST)
    canvas = DiagnosticCanvas(
        source, lines=3 * sum(t.ocr_bbox is not None for t in result.ticks) + 8
    )
    canvas.note("4x original pixels; fields remain linked to original coordinates.")

    def mapped(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return ((box[0] - x) * 4, (box[1] - y) * 4, (box[2] - x) * 4, (box[3] - y) * 4)

    canvas.roi(mapped(result.config.colourbar_bbox), "colourbar", "manual")
    settings = result.config.legend.ocr
    for tick in result.ticks:
        if tick.ocr_bbox is not None:
            canvas.roi(mapped(tick.ocr_bbox), "tick_search", "proposed", f"tick-{tick.index}-field")
            canvas.note(f"Original field: {tick.ocr_bbox}")
            canvas.ocr(
                mapped(tick.text_bbox) if tick.text_bbox else None,
                tick.confidence,
                f"tick-{tick.index}",
                (
                    f"{tick.normalised_text} (raw: {tick.raw_text})"
                    if tick.normalised_text != tick.raw_text
                    else tick.raw_text
                )
                or "missing",
                tick.status,
                settings.min_confidence,
                settings.high_confidence,
            )
    return canvas.image


def grid_diagnostic_image(image: RGBArray, config: "GridConfig") -> Image.Image:
    """Original pixels and the exact resolved intervals used by numbers and OCR."""
    from .diagnostic_style import DiagnosticCanvas

    grid = resolve_grid(config)
    canvas = DiagnosticCanvas(Image.fromarray(image), lines=config.rows + config.columns + 20)
    canvas.roi(grid.bbox, "heatmap", "selected")
    x, y, r, b = grid.bbox
    for name, axis in (("rows", grid.rows), ("columns", grid.columns)):
        for slot, (group, local) in zip(axis.slots, axis.ownership, strict=True):
            box = (x, slot.start, r, slot.end) if name == "rows" else (slot.start, y, slot.end, b)
            canvas.roi(box, "heatmap", "selected", f"{name} {slot.index} group {group}:{local}")
        for start, end in axis.gaps:
            box = (x, start, r, end) if name == "rows" else (start, y, end, b)
            canvas.roi(box, "heatmap", "selected", f"{name} gap")
    canvas.note(f"Resolved grid: {grid.metadata()['sha256']}")
    return canvas.image
