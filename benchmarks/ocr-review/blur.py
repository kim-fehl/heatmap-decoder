"""Small paired development sweep and same-zoom visual blur comparison."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.m19.evaluate import measured, workflow  # noqa: E402
from benchmarks.m19.fixtures import fixture  # noqa: E402

records = []
for family, seed in zip(
    ("none", "horizontal", "vertical", "both"), range(291001, 291005), strict=True
):
    pixels, config, truth, _ = fixture(seed=seed, family=family)
    for radius in (0, 0.3, 0.6, 1.0):
        image = Image.fromarray(pixels).filter(ImageFilter.GaussianBlur(radius))
        automatic = workflow(np.asarray(image), config, truth)
        record = dict(
            family=family,
            seed=seed,
            radius=radius,
            explicit=measured(np.asarray(image), config, truth),
            automatic_correct=automatic["correct"],
            blocker=automatic["earliest_blocker"],
        )
        records.append(record)
        print(
            family,
            radius,
            record["explicit"]["accurate"],
            record["automatic_correct"],
            record["blocker"],
            flush=True,
        )
(ROOT / "benchmarks/ocr-review/blur-sweep.json").write_text(json.dumps(records, indent=2) + "\n")
real = Image.open(ROOT / "examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg").convert("RGB")
pixels, config, *_ = fixture(seed=291002, family="both")
synthetic = Image.fromarray(pixels)
x, y, r, b = config.colourbar_bbox
box = (x, y - 12, min(synthetic.width, r + 65), y + 35)
crops = [
    ("Publication: original ticks", real.crop((12, 105, 218, 155))),
    ("Publication: original genes", real.crop((850, 580, 1040, 630))),
]
for radius in (0, 0.6, 2):
    crops.append(
        (
            f"Synthetic ticks: Gaussian radius {radius}px",
            synthetic.filter(ImageFilter.GaussianBlur(radius)).crop(box),
        )
    )
font = ImageFont.load_default(size=16)
canvas = Image.new("RGB", (900, sum(c.height * 4 + 40 for _, c in crops) + 35), "white")
draw = ImageDraw.Draw(canvas)
draw.text(
    (12, 8),
    "All crops at 4x nearest-neighbour display zoom; no added blur on publication.",
    font=font,
    fill="black",
)
y = 35
for title, crop in crops:
    draw.text((12, y), title, font=font, fill="black")
    canvas.paste(
        crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.NEAREST), (12, y + 28)
    )
    y += crop.height * 4 + 40
canvas.save(ROOT / "docs/images/blur-comparison.png")
