"""Generate fixed, upright horizontal OCR references independently of extraction.

Requires DejaVuSans.ttf (Debian/Ubuntu: fonts-dejavu-core). Checked-in fixtures
are the integration inputs; tests never regenerate or search for better variants.
"""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROWS = ["TP53", "HLA-DRA", "Gapdh", "BRCA2"]
COLUMNS = ["Control-1", "Tumor_2", "Sample A"]


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("DejaVuSans.ttf", 30)
    image = Image.new("RGB", (1200, 550), "white")
    draw = ImageDraw.Draw(image)
    for index, text in enumerate(ROWS):
        draw.text((110, 180 + index * 80), text, font=font, fill="black", anchor="mm")
    for index, text in enumerate(COLUMNS):
        draw.text((350 + index * 260, 92), text, font=font, fill="black", anchor="mm")
    for row in range(4):
        for column in range(3):
            draw.rectangle(
                (220 + column * 260, 140 + row * 80, 479 + column * 260, 219 + row * 80),
                fill=(35 + row * 45, 60 + column * 60, 190 - row * 25),
            )
    image.save(directory / "clean.png")
    image.filter(ImageFilter.GaussianBlur(2)).save(directory / "blur.png")
    image.save(directory / "compressed.jpg", quality=35, subsampling=2)
    image.filter(ImageFilter.GaussianBlur(12)).save(directory / "unreadable.png")
    config = {
        "heatmap_bbox": [220, 140, 1000, 460],
        "rows": 4,
        "columns": 3,
        "labels": {
            "rows": {"source": "ocr", "bbox": [10, 140, 210, 460]},
            "columns": {"source": "ocr", "bbox": [220, 60, 1000, 125]},
        },
    }
    for name, data in [("config.json", config), ("truth.json", {"rows": ROWS, "columns": COLUMNS})]:
        (directory / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    # Record generation inputs without requiring a font file at extraction time.
    (directory / "fixture.json").write_text(
        json.dumps(
            {
                "font": "DejaVu Sans 30 px",
                "font_file": Path(font.path).name,
                "blur_radius": 2,
                "jpeg_quality": 35,
                "jpeg_subsampling": 2,
                "unreadable_blur_radius": 12,
                "ocr_thresholds": "unchanged engineering defaults (confidence 80, offset 0.4)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
