"""M14 generator v1: observed LUTs, fixed numerical scales, no recogniser palette input."""

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

GENERATOR_VERSION = 1
PALETTES = (
    "viridis",
    "magma",
    "cividis",
    "grayscale",
    "blue_white_red",
    "teal_black_red",
    "asymmetric",
)
ANCHORS = {
    "grayscale": ([0, 1], [[25, 25, 25], [220, 220, 220]]),
    "blue_white_red": ([0, 0.5, 1], [[20, 60, 210], [245, 245, 245], [210, 35, 25]]),
    "teal_black_red": ([0, 0.5, 1], [[0, 180, 180], [0, 0, 0], [210, 0, 0]]),
    "asymmetric": (
        [0, 0.18, 0.6, 1],
        [[15, 40, 110], [15, 180, 140], [80, 30, 80], [230, 160, 15]],
    ),
    "repeated": ([0, 0.5, 1], [[20, 60, 210], [245, 245, 245], [20, 60, 210]]),
}


def lut(name):
    t = np.linspace(0, 1, 256)
    if name in ("viridis", "magma", "cividis"):
        from matplotlib import colormaps

        table = np.rint(colormaps[name](t)[:, :3] * 255).astype(np.uint8)
        source = {"catalogue": "Matplotlib", "name": name}
    else:
        positions, anchors = ANCHORS[name if name != "plateau" else "grayscale"]
        table = np.rint(
            np.column_stack([np.interp(t, positions, np.array(anchors)[:, c]) for c in range(3)])
        ).astype(np.uint8)
        if name == "plateau":
            table[179:] = table[179]
        source = {"anchors": anchors, "positions": positions}
    return table, {
        **source,
        "generator_version": GENERATOR_VERSION,
        "interpolation": "linear RGB between 256 rounded 8-bit LUT entries",
        "lut": table.tolist(),
        "lut_sha256": hashlib.sha256(table.tobytes()).hexdigest(),
    }


def colours(t, table):
    return (
        np.rint(
            np.column_stack(
                [
                    np.interp(np.asarray(t).ravel(), np.linspace(0, 1, len(table)), table[:, c])
                    for c in range(3)
                ]
            )
        )
        .astype(np.uint8)
        .reshape(*np.asarray(t).shape, 3)
    )


def cases(split):
    for palette in PALETTES:
        for layout in range(4):
            # Both partitions span orientation/background/grid/direction.
            for variant in ("clean", "jpeg", "blur"):
                yield {
                    "id": f"{split}-{palette}-{layout}-{variant}",
                    "base_id": f"{split}-{palette}-{layout}",
                    "split": split,
                    "palette": palette,
                    "layout": layout,
                    "variant": variant,
                    "seed": (314000 if split == "development" else 414000) + layout,
                    "negative": None,
                }


def challenges(split):
    for name in (
        "repeated",
        "plateau",
        "categorical",
        "competing_bars",
        "off_bar",
        "quantised",
        "black_border",
        "dark_band",
        "clipped",
    ):
        yield {
            "id": f"{split}-challenge-{name}",
            "base_id": f"{split}-challenge-{name}",
            "split": split,
            "palette": name
            if name in ("repeated", "plateau")
            else "teal_black_red"
            if name == "dark_band"
            else "viridis",
            "layout": 0,
            "variant": "clean",
            "seed": 514000 if split == "development" else 614000,
            "negative": name,
        }


def fixture(case):
    table, palette = lut(case["palette"])
    layout = case["layout"]
    orientation = "vertical" if layout % 2 == 0 else "horizontal"
    dark = layout >= 2
    grid = layout in (1, 2)
    start, end = (-3, 5) if layout % 2 == 0 else (5, -3)
    rows, columns, px, py = 24, 12, 22, 16
    x, y, r, s = 80, 80, 80 + columns * px, 80 + rows * py
    box = (r + 60, y, r + 76, s) if orientation == "vertical" else (x, s + 50, r, s + 66)
    image = np.full((s + 120, r + 180, 3), 8 if dark else 255, dtype=np.uint8)
    t = np.random.default_rng(case["seed"]).uniform(0.02, 0.98, (rows, columns))
    t[0, :3] = [0, 0.5, 1]
    if case["negative"] == "dark_band":
        t[:, :2] = 0.5
    rgb = colours(t, table)
    if case["negative"] == "quantised":
        rgb = np.rint(rgb.astype(float) / 32) * 32
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cells = np.repeat(np.repeat(rgb, py, axis=0), px, axis=1)
    if grid:
        cells[py - 1 :: py] = 0
        cells[:, px - 1 :: px] = 0
    image[y:s, x:r] = cells
    bx, by, br, bs = box
    length = bs - by if orientation == "vertical" else br - bx
    profile = colours(np.linspace(0, 1, length), table)
    image[by:bs, bx:br] = profile[:, None] if orientation == "vertical" else profile[None]
    name = case["negative"]
    if name == "categorical":
        image[by : (by + bs) // 2, bx:br] = profile[20]
        image[(by + bs) // 2 : bs, bx:br] = profile[-20]
    elif name == "competing_bars":
        image[by:bs, 20:36] = image[by:bs, bx:br]
    elif name == "off_bar":
        image[y : y + 3 * py, x : x + 3 * px] = [255, 0, 255]
    elif name == "black_border":
        image[by - 1 : bs + 1, bx - 1] = image[by - 1 : bs + 1, br] = 0
        image[by - 1, bx - 1 : br + 1] = image[bs, bx - 1 : br + 1] = 0
    elif name == "clipped":
        image[by:bs, bx:br] = colours(np.linspace(0.2, 0.8, length), table)[:, None]
    figure = Image.fromarray(image)
    if case["variant"] == "blur":
        figure = figure.filter(ImageFilter.GaussianBlur(0.65))
    elif case["variant"] == "jpeg":
        stream = BytesIO()
        figure.save(stream, "JPEG", quality=85)
        stream.seek(0)
        figure = Image.open(stream).convert("RGB")
    config = dict(
        heatmap_bbox=(x, y, r, s),
        colourbar_bbox=box,
        colourbar_orientation=orientation,
        rows=rows,
        columns=columns,
        colourbar_start_value=start,
        colourbar_end_value=end,
    )
    truth = {
        "config": config,
        "values": (start + (end - start) * t).tolist(),
        "palette": palette,
        "background": 8 if dark else 255,
        "grid": grid,
    }
    assert bx >= 0 and by >= 0 and br <= image.shape[1] and bs <= image.shape[0]
    assert r <= bx or s <= by
    return np.array(figure), truth
