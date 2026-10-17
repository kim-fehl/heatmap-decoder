"""Independent MIT synthetic renderer with split-specific matrices and exact annotations."""

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
GENES = (
    "HSP70A HSP70B HSP70D TP53 BRCA1 BRCA2 EGFR MYC PTEN KRAS NRAS BRAF "
    "AKT1 GAPDH ACTB HLA-DRA CD4 CD8A IL6 TNF VEGFA ESR1 ERBB2 MAPK1 "
    "STAT3 JUN FOS BCL2 BAX MTOR"
).split()


def palette(t):
    t = np.asarray(t)[..., None]
    return np.rint(
        np.where(
            t <= 0.5,
            [25, 50, 200] + 2 * t * [210, 185, 35],
            [235, 235, 235] + (2 * t - 1) * [-35, -200, -210],
        )
    ).astype(np.uint8)


def cases(split):
    for i in range(45 if split == "test" else 15):
        seed = {"development": 120000, "validation": 121000, "test": 122000}[split] + i
        family = ("vertical_flank", "horizontal_cap", "cross_axis")[i % 3]
        for degradation in ("clean", "blur", "jpeg"):
            yield dict(
                id=f"{split}-{i:02d}-{degradation}",
                base_id=f"{split}-{i:02d}",
                split=split,
                seed=seed,
                family=family,
                index=i,
                degradation=degradation,
            )


def make_fixture(case):
    i = case["index"]
    rows, columns = ((20, 6), (25, 8), (30, 10))[(i // 3) % 3]
    lo, hi = ((-4, 4), (-3, 3), (-2, 2), (-4, 2), (-3, 4))[(i // 3) % 5]
    descending = i % 2 == 0
    start_value, end_value = (hi, lo) if descending else (lo, hi)
    left, top, px, py = 340, 260, 80, 26
    right, bottom = left + columns * px, top + rows * py
    family = case["family"]
    if family == "vertical_flank":
        orientation, side = "vertical", "left" if (i // 3) % 2 else "right"
    elif family == "horizontal_cap":
        orientation, side = "horizontal", "top" if (i // 3) % 2 else "bottom"
    else:
        orientation = "horizontal" if (i // 3) % 2 else "vertical"
        side = (
            ("left" if (i // 6) % 2 else "right")
            if orientation == "horizontal"
            else ("top" if (i // 6) % 2 else "bottom")
        )
    width, height = right + 340, bottom + 330
    image = np.full((height, width, 3), 255, np.uint8)
    raw = np.random.default_rng(case["seed"]).normal(size=(rows, columns))
    z = (raw - raw.mean(1, keepdims=True)) / raw.std(1, keepdims=True)
    # A range-relative synthetic displayed quantity: never original expression counts.
    values = np.clip((lo + hi) / 2 + z * (hi - lo) / 5, lo, hi)
    t = (values - start_value) / (end_value - start_value)
    cells = np.repeat(np.repeat(palette(t), py, 0), px, 1)
    if (i // 3) % 2:
        cells[py - 1 :: py] = 0
        cells[:, px - 1 :: px] = 0
    image[top:bottom, left:right] = cells
    if orientation == "vertical":
        x = 65 if side == "left" else right + 190 if side == "right" else left + columns * px // 2
        y = top if side in ("left", "right") else 25 if side == "top" else bottom + 65
        length = rows * py if side in ("left", "right") else 200
        bar = (x, y, x + 16, y + length)
    else:
        x = left if side in ("top", "bottom") else 20 if side == "left" else right + 35
        y = 80 if side == "top" else bottom + 150 if side == "bottom" else top + rows * py // 2
        length = columns * px if side in ("top", "bottom") else 280
        bar = (x, y, x + length, y + 16)
    x, y, r, s = bar
    profile = palette(np.linspace(0, 1, length))
    image[y:s, x:r] = profile[:, None] if orientation == "vertical" else profile[None]
    figure = Image.fromarray(image)
    draw = ImageDraw.Draw(figure)
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    tick_font = ImageFont.truetype("DejaVuSans.ttf", 20)
    tokens = {"rows": [], "columns": []}
    row_side = "left" if (i // 2) % 2 else "right"
    column_side = "top" if (i // 2) % 2 else "bottom"
    samples = [f"S_{j + 1:02d}" for j in range(columns)]
    for j, text in enumerate(GENES[:rows]):
        cx = left - 20 - draw.textlength(text, font=font) if row_side == "left" else right + 20
        cy = top + (j + 0.5) * py
        b = draw.textbbox((cx, cy), text, font=font)
        cy -= (b[3] - cy + b[1] - cy) / 2
        draw.text((cx, cy), text, font=font, fill="black")
        tokens["rows"].append(dict(text=text, bbox=draw.textbbox((cx, cy), text, font=font)))
    for j, text in enumerate(samples):
        cx = left + (j + 0.5) * px - draw.textlength(text, font=font) / 2
        cy = top - 45 if column_side == "top" else bottom + 25
        draw.text((cx, cy), text, font=font, fill="black")
        tokens["columns"].append(dict(text=text, bbox=draw.textbbox((cx, cy), text, font=font)))
    count = (3, 5, 7)[(i // 5) % 3]
    axis_start, axis_last = (y, s - 1) if orientation == "vertical" else (x, r - 1)
    tick_values = np.linspace(start_value, end_value, count)
    positions = np.rint(np.linspace(axis_start, axis_last, count)).astype(int)
    ticks = []
    tick_side = (
        ("right" if side != "right" else "left")
        if orientation == "vertical"
        else ("below" if side != "bottom" else "above")
    )
    for p, v in zip(positions, tick_values, strict=True):
        text = f"{v:.6g}"
        if orientation == "vertical":
            a, b = (r + 2, r + 8) if tick_side == "right" else (x - 8, x - 2)
            draw.line((a, int(p), b, int(p)), fill="black")
            tx = r + 14 if tick_side == "right" else x - 14 - draw.textlength(text, font=tick_font)
            ty = float(p)
            b = draw.textbbox((tx, ty), text, font=tick_font)
            ty -= (b[1] + b[3]) / 2 - p
        else:
            a, b = (s + 2, s + 8) if tick_side == "below" else (y - 8, y - 2)
            draw.line((int(p), a, int(p), b), fill="black")
            tx = p - draw.textlength(text, font=tick_font) / 2
            ty = s + 14 if tick_side == "below" else y - 34
        draw.text((tx, ty), text, font=tick_font, fill="black")
        ticks.append(
            dict(
                text=text,
                value=float(text),
                position=int(p),
                bbox=draw.textbbox((tx, ty), text, font=tick_font),
            )
        )
    truth = dict(
        heatmap_bbox=(left, top, right, bottom),
        colourbar_bbox=bar,
        rows=rows,
        columns=columns,
        orientation=orientation,
        start_value=start_value,
        end_value=end_value,
        expected_matrix=values.tolist(),
        ticks=ticks,
        genes=GENES[:rows],
        samples=samples,
        tokens=tokens,
        row_side=row_side,
        column_side=column_side,
        bar_side=side,
        scale_type="linear",
        quantity="synthetic displayed score",
        licence="MIT",
    )
    if case["degradation"] == "blur":
        figure = figure.filter(ImageFilter.GaussianBlur(0.65))
    elif case["degradation"] == "jpeg":
        stream = BytesIO()
        figure.save(stream, format="JPEG", quality=85)
        stream.seek(0)
        figure = Image.open(stream).convert("RGB")
    return np.asarray(figure), truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=("development", "validation", "test"), default="development"
    )
    parser.add_argument("--out", type=Path, default=Path("work/m12-fixtures"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for case in cases(args.split):
        image, truth = make_fixture(case)
        Image.fromarray(image).save(args.out / f"{case['id']}.png")
        (args.out / f"{case['id']}.json").write_text(
            json.dumps(dict(case=case, truth=truth), indent=2) + "\n"
        )
