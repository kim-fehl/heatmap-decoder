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
                        (150, 150, 150)
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
                if flag & CellFlag.MISSING:
                    draw.line((x, y, x + size - 1, y + size - 1), fill="black")
                # Outline poor matches even when ambiguity takes colour priority.
                if panel == 2 and flag & CellFlag.POOR_MATCH:
                    draw.rectangle((x, y, x + size - 1, y + size - 1), outline="black", width=2)
    draw.text(
        (gap, header + height + 8),
        "QC: green OK; orange poor; purple ambiguous; grey/cross missing",
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
    source = Image.fromarray(image).convert("RGB")
    count = sum(len(axis.slots) for axis in result.axes)
    canvas = Image.new(
        "RGB", (source.width + 380, max(source.height + 70, 90 + count * 20)), "white"
    )
    canvas.paste(source, (0, 0))
    draw = ImageDraw.Draw(canvas)
    colours = {
        "accepted": "green",
        "low_confidence": "darkorange",
        "missing": "red",
        "ambiguous": "purple",
        "manual": "blue",
        "generated": "gray",
    }
    x, y = source.width + 12, 10
    draw.text((x, y), "Label review: axis / index / status", fill="black")
    y += 25
    left, top, right, bottom = result.config.heatmap_bbox
    draw.rectangle((left, top, right - 1, bottom - 1), outline="black", width=2)
    for axis in result.axes:
        roi = getattr(result.config.labels, axis.axis) if result.config.labels is not None else None
        if isinstance(roi, OCRLabelSource):
            x0, y0, x1, y1 = roi.bbox
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="cyan", width=2)
            draw.text((x0, max(0, y0 - 14)), f"{axis.axis} ROI", fill="blue")
        for slot, label in zip(axis_slots(result.config, axis.axis), axis.slots, strict=True):
            colour = colours[label.status]
            if axis.axis == "rows":
                bounds = roi.bbox if isinstance(roi, OCRLabelSource) else result.config.heatmap_bbox
                draw.line((bounds[0], slot.centre, bounds[2] - 1, slot.centre), fill=colour)
                draw.ellipse((left - 3, slot.centre - 3, left + 3, slot.centre + 3), fill=colour)
            else:
                bounds = roi.bbox if isinstance(roi, OCRLabelSource) else result.config.heatmap_bbox
                draw.line((slot.centre, bounds[1], slot.centre, bounds[3] - 1), fill=colour)
                draw.ellipse((slot.centre - 3, top - 3, slot.centre + 3, top + 3), fill=colour)
            draw.text((x, y), f"{axis.axis} {label.index}: {label.status}", fill=colour)
            if label.duplicate:
                draw.text((x + 260, y), "duplicate", fill="purple")
            y += 20
        for token in axis.tokens:
            if token.text.strip() and token.confidence >= 0:
                draw.rectangle(token.bbox, outline="gray")
        for index, candidate in enumerate(axis.candidates):
            labels = [s for s in axis.slots if index in s.candidate_indices]
            status = next(
                (s.status for s in labels if s.status == "ambiguous"),
                labels[0].status if labels else "ambiguous",
            )
            draw.rectangle(candidate.bbox, outline=colours[status], width=2)
        if axis.unmatched_candidate_indices:
            draw.text(
                (x, y),
                f"{axis.axis}: unmatched candidates {list(axis.unmatched_candidate_indices)}",
                fill="purple",
            )
            y += 20
    draw.text(
        (10, source.height + 12),
        "Cyan: configured ROI. Grey: raw tokens. Coloured boxes: candidates.",
        fill="black",
    )
    draw.text(
        (10, source.height + 32),
        "Centres and ledger preserve missing/ambiguous slots. Verify all OCR text.",
        fill="black",
    )
    return canvas
