"""Original MIT synthetic M8 layouts, annotations and predeclared base splits."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Transformations stay with their base layout; fixed before detector evaluation.
LAYOUTS = {
    "vertical_sequential": ("development", "vertical", "sequential", 0),
    "horizontal_diverging": ("development", "horizontal", "diverging", 2),
    "left_grayscale": ("held_out", "vertical", "grayscale", 1),
    "below_sequential": ("held_out", "horizontal", "sequential", 1),
}


def palette(t, name):
    t = np.asarray(t)[..., None]
    if name == "grayscale":
        return np.repeat(np.rint(30 + 180 * t).astype(np.uint8), 3, axis=-1)
    if name == "sequential":
        return np.rint([30, 50, 80] + t * np.array([150, 130, 100])).astype(np.uint8)
    return np.rint(
        np.where(
            t < 0.5,
            [25, 50, 200] + 2 * t * np.array([210, 185, 35]),
            [235, 235, 235] + (2 * t - 1) * np.array([-35, -200, -210]),
        )
    ).astype(np.uint8)


def make_layout(name):
    split, orientation, colour, border = LAYOUTS[name]
    image = np.full((430, 530, 3), 250, dtype=np.uint8)
    box = (130, 100, 370, 280)
    if name == "left_grayscale":
        bar = (65, 100, 79, 280)
    elif orientation == "vertical":
        bar = (410, 100, 424, 280)
    elif name == "below_sequential":
        bar = (130, 330, 370, 344)
    else:
        bar = (130, 40, 370, 54)
    values = np.random.default_rng(82026 + list(LAYOUTS).index(name)).uniform(0.08, 0.92, (6, 8))
    image[100:280, 130:370] = np.repeat(np.repeat(palette(values, colour), 30, axis=0), 30, axis=1)
    x0, y0, x1, y1 = bar
    if border:
        image[y0 - border : y1 + border, x0 - border : x1 + border] = 0
    length = y1 - y0 if orientation == "vertical" else x1 - x0
    profile = palette(np.linspace(0, 1, length), colour)
    image[y0:y1, x0:x1] = profile[:, None] if orientation == "vertical" else profile[None]
    figure = Image.fromarray(image)
    draw = ImageDraw.Draw(figure)
    # Separable categorical track and unrelated text explicitly confound the search.
    for col in range(8):
        draw.rectangle(
            (130 + col * 30, 80, 159 + col * 30, 87),
            fill=(220, 30, 30) if col % 2 else (30, 160, 30),
        )
    draw.text((15, 15), "Panel A: unrelated caption", fill="black")
    draw.text((440, 100), "2", fill="black")
    draw.text((440, 270), "-2", fill="black")
    return np.asarray(figure), {
        "split": split,
        "heatmap_bbox": box,
        "colourbar_bbox": bar,
        "orientation": orientation,
        "rows": 6,
        "columns": 8,
        "licence": "MIT",
        "edge_tolerance_pixels": 1,
    }


def count_fixture(kind):
    rows, columns, pitch = 6, 8, 20
    image = np.full((170, 210, 3), 255, dtype=np.uint8)
    values = np.random.default_rng(8299).uniform(0.08, 0.92, (rows, columns))
    if kind == "merged":
        values[1::2] = values[::2]
        values[:, 1::2] = values[:, ::2]
    if kind == "uniform":
        values[:] = 0.5
    cells = np.repeat(np.repeat(palette(values, "diverging"), pitch, axis=0), pitch, axis=1)
    if kind in {"complete", "omitted"}:
        for i in range(1, rows):
            if kind == "complete" or i % 2:
                cells[i * pitch - 1] = 0
        for i in range(1, columns):
            if kind == "complete" or i % 2:
                cells[:, i * pitch - 1] = 0
    image[25:145, 25:185] = cells
    return image, (25, 25, 185, 145), rows, columns


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in LAYOUTS:
        image, annotation = make_layout(name)
        Image.fromarray(image).save(directory / f"{name}.png")
        manifest[name] = annotation
    for name in ("complete", "omitted", "merged", "uniform"):
        image, box, rows, columns = count_fixture(name)
        Image.fromarray(image).save(directory / f"count_{name}.png")
        manifest[f"count_{name}"] = {
            "split": "held_out",
            "heatmap_bbox": box,
            "rows": rows,
            "columns": columns,
            "licence": "MIT",
        }
    (directory / "fixtures.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
