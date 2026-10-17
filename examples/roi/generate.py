"""MIT synthetic fixtures; split by base layout before detector evaluation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

# Each base layout belongs to one split, including any future degradations.
FIXTURES = {
    "sequential": ("development", 8, 12, 12, 10, "sequential", 0, False),
    "diverging": ("development", 12, 8, 10, 14, "diverging", 0, False),
    "grayscale": ("held_out", 9, 11, 13, 11, "grayscale", 0, False),
    "gridlines": ("held_out", 10, 8, 15, 12, "diverging", 0, True),
    "border": ("held_out", 7, 13, 11, 16, "sequential", 2, False),
    "large": ("held_out", 14, 9, 47, 39, "grayscale", 1, True),
}


def make_fixture(name: str) -> tuple[np.ndarray, tuple[int, int, int, int], int, int]:
    _, rows, columns, px, py, palette, border, lines = FIXTURES[name]
    values = np.random.default_rng(7026 + list(FIXTURES).index(name)).uniform(
        0.08, 0.92, (rows, columns)
    )
    if palette == "grayscale":
        rgb = np.repeat((30 + values[..., None] * 180), 3, axis=2)
    elif palette == "sequential":
        rgb = np.array([30, 55, 90]) + values[..., None] * np.array([150, 130, 100])
    else:
        t = values[..., None]
        rgb = np.where(
            t < 0.5,
            [25, 50, 200] + 2 * t * np.array([210, 185, 35]),
            [235, 235, 235] + (2 * t - 1) * np.array([-35, -200, -210]),
        )
    cells = np.repeat(np.repeat(np.rint(rgb).astype(np.uint8), py, axis=0), px, axis=1)
    if lines:
        cells[py - 1 : -1 : py] = 255
        cells[:, px - 1 : -1 : px] = 255
    left, top = 37, 29
    right, bottom = left + columns * px, top + rows * py
    image = np.full((bottom + 53, right + 67, 3), 245, dtype=np.uint8)
    if border:
        image[top - border : bottom + border, left - border : right + border] = 0
    image[top:bottom, left:right] = cells
    return image, (left, top, right, bottom), rows, columns


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, definition in FIXTURES.items():
        image, box, rows, columns = make_fixture(name)
        Image.fromarray(image).save(directory / f"{name}.png")
        manifest[name] = {
            "split": definition[0],
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
