"""MIT development fixtures: ordinary 20–30 gene x 6–10 sample raster layouts.

These are synthetic development/regression data, never an M12 hold-out corpus.
Both orientations on all four bar sides are declared before evaluation.
"""

import argparse
import itertools
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

GENES = (
    "TP53",
    "BRCA1",
    "BRCA2",
    "EGFR",
    "MYC",
    "PTEN",
    "KRAS",
    "NRAS",
    "BRAF",
    "AKT1",
    "GAPDH",
    "ACTB",
    "HLA-DRA",
    "CD4",
    "CD8A",
    "IL6",
    "TNF",
    "VEGFA",
    "ESR1",
    "ERBB2",
    "MAPK1",
    "STAT3",
    "JUN",
    "FOS",
    "BCL2",
    "BAX",
    "MTOR",
    "RB1",
    "APOE",
    "SOX2",
)


def palette(t, kind):
    t = np.asarray(t)[..., None]
    if kind == "grayscale":
        return np.repeat(np.rint(30 + 180 * t).astype(np.uint8), 3, axis=-1)
    if kind == "sequential":
        return np.rint([30, 50, 80] + t * np.array([150, 130, 100])).astype(np.uint8)
    return np.rint(
        np.where(
            t <= 0.5,
            [25, 50, 200] + 2 * t * np.array([210, 185, 35]),
            [235, 235, 235] + (2 * t - 1) * np.array([-35, -200, -210]),
        )
    ).astype(np.uint8)


def make_layout(
    rows=25,
    columns=8,
    row_side="left",
    column_side="top",
    bar_side="right",
    orientation="vertical",
    kind="diverging",
    degradation="clean",
    gridlines=False,
):
    left, top, px, py = 320, 220, 72, 22
    right, bottom = left + columns * px, top + rows * py
    box = (left, top, right, bottom)
    values = np.random.default_rng(101026 + rows + columns).uniform(0.06, 0.94, (rows, columns))
    image = np.full((bottom + 220, right + 240, 3), 255, dtype=np.uint8)
    cells = np.repeat(np.repeat(palette(values, kind), py, axis=0), px, axis=1)
    if gridlines:
        cells[py - 1 :: py] = 0
        cells[:, px - 1 :: px] = 0
    image[top:bottom, left:right] = cells
    if orientation == "vertical":
        if bar_side in {"left", "right"}:
            x = 40 if bar_side == "left" else right + 160
            bar = (x, top, x + 14, bottom)
        else:
            y = 20 if bar_side == "top" else bottom + 90
            bar = ((left + right) // 2, y, (left + right) // 2 + 14, y + 100)
    elif bar_side in {"top", "bottom"}:
        y = 100 if bar_side == "top" else bottom + 100
        bar = (left, y, right, y + 14)
    else:
        x = 30 if bar_side == "left" else right + 100
        y = (top + bottom) // 2
        bar = (x, y, x + 120, y + 14)
    x0, y0, x1, y1 = bar
    length = y1 - y0 if orientation == "vertical" else x1 - x0
    profile = palette(np.linspace(0, 1, length), kind)
    image[y0:y1, x0:x1] = profile[:, None] if orientation == "vertical" else profile[None]
    figure = Image.fromarray(image)
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default(size=14)
    tokens = {"rows": [], "columns": []}
    for index, text in enumerate(GENES[:rows]):
        centre = top + (index + 0.5) * py
        width = draw.textlength(text, font=font)
        x = left - 20 - width if row_side == "left" else right + 20
        y = centre - 7
        b = draw.textbbox((x, y), text, font=font)
        # Centre the ink on the slot rather than Pillow's font ascent box.
        y += centre - (b[1] + b[3]) / 2
        draw.text((x, y), text, fill="black", font=font)
        b = draw.textbbox((x, y), text, font=font)
        tokens["rows"].append({"text": text, "bbox": b})
    samples = [f"S_{index + 1:02d}" for index in range(columns)]
    for index, text in enumerate(samples):
        centre = left + (index + 0.5) * px
        x = centre - draw.textlength(text, font=font) / 2
        y = top - 40 if column_side == "top" else bottom + 20
        draw.text((x, y), text, fill="black", font=font)
        tokens["columns"].append({"text": text, "bbox": draw.textbbox((x, y), text, font=font)})
    label_boxes = {}
    for axis, records in tokens.items():
        boxes = np.array([record["bbox"] for record in records])
        label_boxes[axis] = [
            int(np.floor(boxes[:, 0].min())) - 4,
            int(np.floor(boxes[:, 1].min())) - 4,
            int(np.ceil(boxes[:, 2].max())) + 4,
            int(np.ceil(boxes[:, 3].max())) + 4,
        ]
    if degradation == "blur":
        figure = figure.filter(ImageFilter.GaussianBlur(0.65))
    elif degradation == "jpeg":
        buffer = BytesIO()
        figure.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        figure = Image.open(buffer).convert("RGB")
    return np.asarray(figure), {
        "heatmap_bbox": box,
        "colourbar_bbox": bar,
        "rows": rows,
        "columns": columns,
        "orientation": orientation,
        "values": values.tolist(),
        "genes": GENES[:rows],
        "samples": samples,
        "tokens": tokens,
        "label_boxes": label_boxes,
        "row_side": row_side,
        "column_side": column_side,
        "bar_side": bar_side,
        "kind": kind,
        "degradation": degradation,
        "licence": "MIT",
        "split": "development",
    }


def layouts():
    for index, (row_side, column_side, bar_side, orientation) in enumerate(
        itertools.product(
            ("left", "right"),
            ("top", "bottom"),
            ("left", "right", "top", "bottom"),
            ("vertical", "horizontal"),
        )
    ):
        rows, columns = ((20, 6), (25, 8), (30, 10))[index % 3]
        yield dict(
            rows=rows,
            columns=columns,
            row_side=row_side,
            column_side=column_side,
            bar_side=bar_side,
            orientation=orientation,
            kind=("sequential", "diverging", "grayscale")[index % 3],
            gridlines=index % 4 == 0,
        )


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, options in enumerate(layouts()):
        for degradation in ("clean", "blur", "jpeg"):
            image, annotation = make_layout(**options, degradation=degradation)
            name = f"layout-{index:02d}-{degradation}"
            Image.fromarray(image).save(directory / f"{name}.png")
            manifest.append(dict(id=name, image=f"{name}.png", **annotation))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("work/m10-fixtures"))
    generate(parser.parse_args().out)
