"""Pillow rendering of sampled colours, nearest candidates and categorical QC."""

import numpy as np
from PIL import Image, ImageDraw

from .geometry import axis_slots
from .models import CellFlag, ExtractionResult, LabelExtractionResult, OCRLabelSource, RGBArray


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

    count = sum(len(a.tokens) + len(a.candidates) + len(a.slots) for a in result.axes)
    canvas = DiagnosticCanvas(Image.fromarray(image), lines=count + 12)
    canvas.roi(result.config.heatmap_bbox, "heatmap", "manual")
    settings = result.config.labels.ocr if result.config.labels is not None else None
    minimum, high = (settings.min_confidence, settings.high_confidence) if settings else (80, 90)
    for axis in result.axes:
        roi = getattr(result.config.labels, axis.axis) if result.config.labels else None
        if isinstance(roi, OCRLabelSource):
            canvas.roi(roi.bbox, f"labels_{axis.axis}", "manual")
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
                f"{label.raw_text or label.final_text or 'missing'} "
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
