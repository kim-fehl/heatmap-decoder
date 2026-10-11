"""Recreate the bundled reference without using the extraction implementation."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    positions = np.random.default_rng(2026).uniform(0, 1, (20, 20))
    positions[0, :3] = [0, 0.5, 1]
    truth = 2 - 4 * positions

    def palette(t: np.ndarray) -> np.ndarray:
        t = t[..., None]
        blue = np.array([25, 60, 210])
        white = np.array([245, 245, 245])
        red = np.array([210, 35, 25])
        rgb = np.where(t <= 0.5, blue + 2 * t * (white - blue), white + (2 * t - 1) * (red - white))
        return np.rint(rgb).astype(np.uint8)

    image = np.full((576, 350, 3), 255, dtype=np.uint8)
    image[32:272, 24:264] = np.repeat(np.repeat(palette(positions), 12, axis=0), 12, axis=1)
    image[32:544, 292:308] = palette(np.linspace(0, 1, 512))[:, None, :]
    figure = Image.fromarray(image)
    draw = ImageDraw.Draw(figure)
    draw.text((24, 10), "Synthetic 20 x 20 reference (linear scale)", fill="black")
    draw.text((316, 32), "2", fill="black")
    draw.text((316, 532), "-2", fill="black")
    draw.text((24, 292), "Known truth: 2 - 4 * palette position", fill="black")
    figure.save(directory / "figure.png")
    np.savetxt(directory / "truth.csv", truth, delimiter=",", fmt="%.17g")
    config = {
        "heatmap_bbox": [24, 32, 264, 272],
        "colourbar_bbox": [292, 32, 308, 544],
        "rows": 20,
        "columns": 20,
        "colourbar_orientation": "vertical",
        "colourbar_start_value": 2.0,
        "colourbar_end_value": -2.0,
        "sampling_inset_fraction": 0.25,
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
