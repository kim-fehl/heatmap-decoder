"""Deterministic M13 quarter-turn fixtures; no palette dependency."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
STRINGS = [
    "Sample_Alpha-01",
    "Sample_Bravo-02",
    "Sample_Charlie-03",
    "Sample_Delta-04",
    "Sample_Echo-05",
    "Sample_Foxtrot-06",
    "Sample_Golf-07",
    "Sample_Hotel-08",
]
FONTS = {
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "serif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}


CLEAN_STRINGS = [
    "Sample Alpha-A",
    "Sample Bravo-B",
    "Sample Charlie-C",
    "Sample Delta-D",
    "Sample Echo-E",
    "Sample Foxtrot-F",
    "Sample Golf-G",
    "Sample Hotel-H",
]


def fixture(angle=90, placement="above", font="sans", variant="clean", strings=None):
    strings = STRINGS if strings is None else strings
    pitch, left, top = 64, 40, 380 if placement == "above" else 40
    matrix_box = (left, top, left + pitch * 8, top + 160)
    label_box = (
        left,
        20 if placement == "above" else top + 180,
        left + pitch * 8,
        360 if placement == "above" else top + 520,
    )
    image = Image.new("RGB", (640, 600), "white")
    typeface = ImageFont.truetype(FONTS[font], 24)
    for index, text in enumerate(strings):
        patch = Image.new("RGB", (340, 56), "white")
        draw = ImageDraw.Draw(patch)
        draw.text((12, 12), text, font=typeface, fill="black")
        patch = patch.transpose(
            Image.Transpose.ROTATE_90 if angle == 90 else Image.Transpose.ROTATE_270
        )
        image.paste(patch, (left + index * pitch + 4, label_box[1]))
    # Observed injective grayscale LUT, direction left -> right, 0 -> 1.
    draw = ImageDraw.Draw(image)
    truth = np.linspace(0.1, 0.9, 32).reshape(4, 8)
    for r in range(4):
        for c in range(8):
            gray = round(float(truth[r, c]) * 255)
            x, y = left + c * pitch, top + r * 40
            draw.rectangle((x, y, x + pitch - 1, y + 39), fill=(gray,) * 3)
    bar = (40, 570, 296, 590)
    for i in range(256):
        draw.line((40 + i, 570, 40 + i, 589), fill=(i,) * 3)
    if variant == "blur":
        image = image.filter(ImageFilter.GaussianBlur(0.65))
    config = {
        "heatmap_bbox": matrix_box,
        "rows": 4,
        "columns": 8,
        "colourbar_bbox": bar,
        "colourbar_orientation": "horizontal",
        "colourbar_start_value": 0,
        "colourbar_end_value": 1,
        "labels": {
            "columns": {
                "source": "ocr",
                "bbox": label_box,
                "text_rotation_degrees": angle,
                "recognition": "slots",
            }
        },
    }
    return image, config, truth


if __name__ == "__main__":
    image, config, truth = fixture()
    image.save(ROOT / "figure.png")
    (ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (ROOT / "truth.json").write_text(
        json.dumps(
            {
                "columns": STRINGS,
                "values": truth.tolist(),
                "generator_version": 1,
                "palette": "RGB (0,0,0) to (255,255,255), linear RGB, 256 entries",
                "font": FONTS["sans"],
                "font_size": 24,
            },
            indent=2,
        )
        + "\n"
    )
    (ROOT / "reviewed_labels.tsv").write_text(
        "axis\tindex\ttext\n" + "".join(f"columns\t{i}\t{s}\n" for i, s in enumerate(STRINGS))
    )
