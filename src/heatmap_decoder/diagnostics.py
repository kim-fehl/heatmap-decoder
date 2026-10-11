"""Pillow rendering of sampled colours, nearest candidates and categorical QC."""

import numpy as np
from PIL import Image, ImageDraw

from .models import CellFlag, ExtractionResult


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
