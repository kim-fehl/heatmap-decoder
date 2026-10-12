"""MIT biological-layout synthetic ticks, 25 genes × 8 samples; no real counts."""

import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
BASE = runpy.run_path(str(ROOT / "examples/m10/generate.py"))


def make_fixture(
    orientation="vertical",
    count=5,
    descending=True,
    nonuniform=False,
    interior=False,
    nonlinear=False,
    side=None,
    strings=None,
):
    image, truth = BASE["make_layout"](
        rows=25,
        columns=8,
        row_side="left",
        column_side="bottom",
        bar_side="right" if orientation == "vertical" else "top",
        orientation=orientation,
        kind="diverging",
    )
    # Simulate the scientific transformation explicitly. The recovered table is
    # the displayed/clipped row z-score, never the pre-transformation input.
    raw = np.random.default_rng(111026).normal(size=(25, 8))
    zscores = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    displayed = np.clip(zscores, -2, 2)
    positions_in_palette = (2 - displayed) / 4 if descending else (displayed + 2) / 4
    image = image.copy()
    hx, hy, hr, hs = truth["heatmap_bbox"]
    image[hy:hs, hx:hr] = np.repeat(
        np.repeat(BASE["palette"](positions_in_palette, "diverging"), 22, axis=0), 72, axis=1
    )
    truth["values"] = positions_in_palette.tolist()
    truth["unclipped_row_zscores"] = zscores.tolist()
    truth["transformation"] = "per-row mean centering / population standard deviation"
    truth["clipped_cells"] = int(np.count_nonzero(np.abs(zscores) > 2))
    image = np.pad(image, ((0, 0), (0, 100), (0, 0)), constant_values=255)
    figure = Image.fromarray(image)
    draw = ImageDraw.Draw(figure)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    x, y, r, s = truth["colourbar_bbox"]
    start, last = (y, s - 1) if orientation == "vertical" else (x, r - 1)
    if nonuniform:
        fractions = np.linspace(0, 1, count) ** 1.35
    else:
        fractions = np.linspace(0, 1, count)
    if interior:
        fractions = 0.1 + 0.8 * fractions
    positions = np.rint(start + fractions * (last - start)).astype(int)
    t = (positions - start) / (last - start)
    values = (2 - 4 * t) if descending else (-2 + 4 * t)
    if nonlinear:
        values = 1 + 9 * t**2
    side = side or ("right" if orientation == "vertical" else "below")
    token_boxes = []
    for i, (position, value) in enumerate(zip(positions, values, strict=True)):
        text = strings[i] if strings is not None else f"{value:.4g}"
        if orientation == "vertical":
            a, b = (r + 2, r + 8) if side == "right" else (x - 8, x - 2)
            draw.line((a, int(position), b, int(position)), fill="black", width=1)
            width = draw.textlength(text, font=font)
            px = r + 14 if side == "right" else x - 14 - width
            py = float(position) - 10
            bounds = draw.textbbox((px, py), text, font=font)
            py += position - (bounds[1] + bounds[3]) / 2
        else:
            a, b = (s + 2, s + 8) if side == "below" else (y - 8, y - 2)
            draw.line((int(position), a, int(position), b), fill="black", width=1)
            px = position - draw.textlength(text, font=font) / 2
            py = s + 14 if side == "below" else y - 34
        draw.text((px, py), text, font=font, fill="black")
        token_boxes.append({"text": text, "bbox": draw.textbbox((px, py), text, font=font)})
    truth.update(
        positions=positions.tolist(),
        tick_values=values.tolist(),
        tick_tokens=token_boxes,
        tick_side=side,
        quantity="row z-score",
        scale_type="nonlinear" if nonlinear else "linear",
    )
    truth["expected_matrix"] = displayed.tolist()
    return np.asarray(figure), truth


def generate(directory):
    directory.mkdir(parents=True, exist_ok=True)
    image, truth = make_fixture()
    Image.fromarray(image).save(directory / "figure.png")
    config = {
        "heatmap_bbox": truth["heatmap_bbox"],
        "colourbar_bbox": truth["colourbar_bbox"],
        "colourbar_orientation": truth["orientation"],
        "rows": 25,
        "columns": 8,
        "scale": {"source": "ocr_ticks", "quantity": "row z-score"},
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (directory / "legend.json").write_text(
        json.dumps(
            {
                "colourbar_bbox": truth["colourbar_bbox"],
                "colourbar_orientation": truth["orientation"],
            },
            indent=2,
        )
        + "\n"
    )
    (directory / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    # Identity is explicitly supplied; numeric tick review remains separate.
    config["labels"] = {
        "rows": {"source": "file", "path": "genes.txt"},
        "columns": {"source": "file", "path": "samples.txt"},
    }
    (directory / "labelled_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (directory / "genes.txt").write_text("\n".join(truth["genes"]) + "\n")
    (directory / "samples.txt").write_text("\n".join(truth["samples"]) + "\n")


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
