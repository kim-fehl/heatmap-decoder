"""QA-first synthetic biological layouts, independently rasterised by Pillow or Agg."""

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

GENES = (
    "HSP70A HSP70B HSP70D TP53 BRCA1 BRCA2 EGFR MYC PTEN KRAS NRAS BRAF "
    "AKT1 GAPDH ACTB HLA-DRA CD4 CD8A IL6 TNF VEGFA ESR1 ERBB2 MAPK1 "
    "STAT3 JUN FOS BCL2 BAX MTOR"
).split()
FONT_NAMES = ("DejaVuSans", "DejaVuSerif", "DejaVuSansMono")


def palette(values, kind):
    t = np.asarray(values)[..., None]
    if kind == "grayscale":
        return np.repeat(np.rint(30 + 180 * t).astype(np.uint8), 3, axis=-1)
    if kind == "sequential":
        return np.rint([30, 50, 80] + t * [150, 130, 100]).astype(np.uint8)
    return np.rint(
        np.where(
            t <= 0.5,
            [25, 50, 200] + 2 * t * [210, 185, 35],
            [235, 235, 235] + (2 * t - 1) * [-35, -200, -210],
        )
    ).astype(np.uint8)


def cases(split):
    for i in range(12 if split == "development" else 24):
        for degradation in ("clean", "blur", "jpeg"):
            yield dict(
                id=f"{split}-{i:02d}-{degradation}",
                base_id=f"{split}-{i:02d}",
                split=split,
                index=i,
                seed=(231000 if split == "development" else 241000) + i,
                renderer="pillow" if (i // 8) % 2 == 0 else "matplotlib",
                degradation=degradation,
            )


def overlaps(a, b, padding=0):
    return bool(
        min(a[2], b[2]) + padding > max(a[0], b[0]) and min(a[3], b[3]) + padding > max(a[1], b[1])
    )


def assert_layout(truth, shape):
    """Run before degradation/freeze; disallow accidental text collisions or clipping."""
    texts = truth["text_boxes"]
    cores = [truth["heatmap_bbox"], truth["colourbar_bbox"]]
    assert not overlaps(*cores), "heatmap/bar overlap"
    for i, a in enumerate(texts):
        box = a["bbox"]
        assert box[0] >= 4 and box[1] >= 4 and box[2] <= shape[1] - 4 and box[3] <= shape[0] - 4, (
            a,
            shape,
        )
        assert all(not overlaps(box, b, 2) for b in cores), "text/core overlap"
        assert all(not overlaps(box, b["bbox"], 2) for b in texts[i + 1 :]), "text/text overlap"
    assert all(len(t["text"]) <= 6 for t in truth["ticks"]), "unreadable numeric precision"


def make_fixture(case):
    i = case["index"]
    rows, columns = ((20, 6), (25, 8), (30, 10))[(i // 2) % 3]
    px, py = (72, 26) if i % 2 == 0 else (64, 24)
    orientation = "vertical" if i % 2 == 0 else "horizontal"
    side = ("left", "right", "top", "bottom")[(i // 2) % 4]
    kind = ("diverging", "sequential", "grayscale")[(i // 3) % 3]
    font_name = FONT_NAMES[(i // 4) % 3]
    font = ImageFont.truetype(font_name + ".ttf", 18)
    tick_font = ImageFont.truetype(font_name + ".ttf", 20)
    lo, hi = ((-4, 4), (-3, 3), (-2, 2), (-4, 2), (-3, 4))[(i // 2) % 5]
    start, end = (hi, lo) if i % 3 == 0 else (lo, hi)
    tick_count = (3, 5, 7)[(i // 3) % 3]
    values = np.random.default_rng(case["seed"]).uniform(0.04, 0.96, (rows, columns))
    scores = start + (end - start) * values
    strings = [
        f"{v:.2f}".rstrip("0").rstrip(".") if v else "0"
        for v in np.linspace(start, end, tick_count)
    ]
    max_text = max(tick_font.getlength(t) for t in strings)
    bar_length = max(200, int((tick_count - 1) * (max_text + 18)) + 1)
    left = bar_length + 300 if orientation == "horizontal" and side == "left" else 340
    top = bar_length + 200 if orientation == "vertical" and side == "top" else 240
    right, bottom = left + columns * px, top + rows * py
    if orientation == "vertical":
        if side in ("left", "right"):
            x = 120 if side == "left" else right + 230
            y, length = top, rows * py
        else:
            x = (left + right) // 2
            y = 40 if side == "top" else bottom + 100
            length = bar_length
        bar = (x, y, x + 16, y + length)
    else:
        if side in ("top", "bottom"):
            length = max(columns * px, bar_length)
            x, y = left + (columns * px - length) // 2, 80 if side == "top" else bottom + 120
        else:
            x, y, length = (60 if side == "left" else right + 160), (top + bottom) // 2, bar_length
        bar = (x, y, x + length, y + 16)
    x, y, r, s = bar
    width, height = max(right + 400, r + 120), max(bottom + 220, s + 100)
    image = np.full((height, width, 3), 255, np.uint8)
    cells = np.repeat(np.repeat(palette(values, kind), py, axis=0), px, axis=1)
    gridlines = i % 2 == 0
    if gridlines:
        cells[py - 1 :: py] = 0
        cells[:, px - 1 :: px] = 0
    profile = palette(np.linspace(0, 1, length), kind)
    colourbar = (
        np.repeat(profile[:, None], 16, axis=1)
        if orientation == "vertical"
        else np.repeat(profile[None], 16, axis=0)
    )
    texts = []
    row_side = "left" if (i // 3) % 2 else "right"
    column_side = "top" if (i // 3) % 2 else "bottom"
    samples = [f"S_{j + 1:02d}" for j in range(columns)]
    for j, text in enumerate(GENES[:rows]):
        tx = left - 20 - font.getlength(text) if row_side == "left" else right + 20
        texts.append(
            dict(kind="gene", text=text, x=tx, y=top + (j + 0.5) * py, anchor="lm", size=18)
        )
    for j, text in enumerate(samples):
        texts.append(
            dict(
                kind="sample",
                text=text,
                x=left + (j + 0.5) * px,
                y=top - 30 if column_side == "top" else bottom + 30,
                anchor="mm",
                size=18,
            )
        )
    tick_side = (
        ("left" if i % 3 else "right")
        if orientation == "vertical"
        else ("above" if i % 3 else "below")
    )
    positions = np.rint(
        np.linspace(
            y if orientation == "vertical" else x,
            s - 1 if orientation == "vertical" else r - 1,
            tick_count,
        )
    ).astype(int)
    lines = []
    ticks = []
    for position, text in zip(positions, strings, strict=True):
        if orientation == "vertical":
            line = (
                (x - 8, int(position), x - 2, int(position))
                if tick_side == "left"
                else (r + 2, int(position), r + 8, int(position))
            )
            tx = x - 14 - tick_font.getlength(text) if tick_side == "left" else r + 14
            record = dict(kind="tick", text=text, x=tx, y=float(position), anchor="lm", size=20)
        else:
            line = (
                (int(position), y - 8, int(position), y - 2)
                if tick_side == "above"
                else (int(position), s + 2, int(position), s + 8)
            )
            record = dict(
                kind="tick",
                text=text,
                x=float(position),
                y=y - 24 if tick_side == "above" else s + 24,
                anchor="mm",
                size=20,
            )
        texts.append(record)
        lines.append(line)
        ticks.append(dict(text=text, value=float(text), position=int(position)))
    if case["renderer"] == "pillow":
        image[top:bottom, left:right] = cells
        image[y:s, x:r] = colourbar
        figure = Image.fromarray(image)
        draw = ImageDraw.Draw(figure)
        for line in lines:
            draw.line(line, fill="black")
        for record in texts:
            f = font if record["size"] == 18 else tick_font
            xy = record["x"], record["y"]
            draw.text(xy, record["text"], font=f, anchor=record["anchor"], fill="black")
            record["bbox"] = draw.textbbox(xy, record["text"], font=f, anchor=record["anchor"])
    else:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_axis_off()
        ax.imshow(cells, extent=(left, right, bottom, top), interpolation="nearest", aspect="auto")
        ax.imshow(colourbar, extent=(x, r, s, y), interpolation="nearest", aspect="auto")
        for a, b, c, d in lines:
            ax.plot((a, c), (b, d), color="black", linewidth=0.72, antialiased=False)
        artists = []
        families = {
            "DejaVuSans": "DejaVu Sans",
            "DejaVuSerif": "DejaVu Serif",
            "DejaVuSansMono": "DejaVu Sans Mono",
        }
        for record in texts:
            artists.append(
                ax.text(
                    record["x"],
                    record["y"],
                    record["text"],
                    ha="left" if record["anchor"] == "lm" else "center",
                    va="center",
                    fontsize=record["size"] * 0.72,
                    fontfamily=families[font_name],
                )
            )
        canvas.draw()
        for record, artist in zip(texts, artists, strict=True):
            b = artist.get_window_extent(canvas.get_renderer())
            record["bbox"] = (b.x0, height - b.y1, b.x1, height - b.y0)
        figure = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
    truth = dict(
        heatmap_bbox=(left, top, right, bottom),
        colourbar_bbox=bar,
        orientation=orientation,
        rows=rows,
        columns=columns,
        start_value=start,
        end_value=end,
        expected_matrix=scores.tolist(),
        ticks=ticks,
        text_boxes=texts,
        genes=GENES[:rows],
        samples=samples,
        gridlines=gridlines,
        font=font_name,
        bar_side=side,
        row_side=row_side,
        column_side=column_side,
        family=f"{case['renderer']}-{kind}",
        scale_type="linear",
        licence="MIT",
        quantity="synthetic displayed score; not original biological counts",
    )
    assert_layout(truth, (height, width))
    if case["degradation"] == "blur":
        figure = figure.filter(ImageFilter.GaussianBlur(0.65))
    elif case["degradation"] == "jpeg":
        stream = BytesIO()
        figure.save(stream, format="JPEG", quality=85)
        stream.seek(0)
        figure = Image.open(stream).convert("RGB")
    return np.array(figure), truth


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    count = 0
    for case in cases(args.split):
        image, truth = make_fixture(case)
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(args.out / f"{case['id']}.png")
            (args.out / f"{case['id']}.json").write_text(json.dumps(truth, indent=2) + "\n")
        count += 1
    print(f"Layout QA passed: {count} {args.split} images")
