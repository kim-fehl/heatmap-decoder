"""M15 fresh base figures; bounded rotations, fonts, palettes and explicit challenges."""

import hashlib
import runpy
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

M14 = runpy.run_path(str(Path(__file__).resolve().parents[1] / "m14/fixtures.py"))
FONTS = {
    "sans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "mono": "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
}
STRINGS = [
    "Alpha Case-A",
    "Bravo Case-B",
    "Charlie Case-C",
    "Delta Case-D",
    "Echo Case-E",
    "Foxtrot Case-F",
    "Golf Case-G",
    "Hotel Case-H",
]


def cases(split):
    i = 0
    for angle in (0, 90, -90):
        for placement in ("above", "below"):
            for font in FONTS:
                for palette in ("viridis", "teal_black_red"):
                    base = {
                        "id": f"{split}-{i:02d}",
                        "base_id": f"{split}-{i:02d}",
                        "split": split,
                        "angle": angle,
                        "placement": placement,
                        "font": font,
                        "palette": palette,
                        "seed": (151000 if split == "development" else 152000) + i,
                        "challenge": None,
                    }
                    for variant in ("clean", "jpeg", "blur"):
                        yield {**base, "id": base["id"] + "-" + variant, "variant": variant}
                    i += 1
    for i, name in enumerate(
        (
            "diagonal",
            "conflicting_sides",
            "missing",
            "underscores",
            "unlabelled",
            "multiple_panels",
            "nonlinear",
            "shared_bar",
        )
    ):
        yield {
            "id": f"{split}-challenge-{name}",
            "base_id": f"{split}-challenge-{name}",
            "split": split,
            "angle": 90,
            "placement": "above",
            "font": "sans",
            "palette": "viridis",
            "seed": (153000 if split == "development" else 154000) + i,
            "challenge": name,
            "variant": "clean",
        }


def fixture(case):
    image = Image.new("RGB", (1480, 790), "white")
    draw = ImageDraw.Draw(image)
    x, y, px, py = 40, 300, 160, 24
    box = (x, y, x + px * 8, y + py * 6)
    label_box = (
        (x, y - 240, box[2], y)
        if case["placement"] == "above"
        else (x, box[3], box[2], box[3] + 240)
    )
    font = ImageFont.truetype(FONTS[case["font"]], 16)
    strings = (
        STRINGS
        if case["challenge"] != "underscores"
        else [s.replace(" ", "_") + "_01" for s in STRINGS]
    )
    token_boxes = []
    for i, text in enumerate(strings):
        if case["challenge"] == "unlabelled" or case["challenge"] == "missing" and i == 3:
            token_boxes.append(None)
            continue
        text_width = font.getbbox(text)[2] - font.getbbox(text)[0] + 10
        patch = Image.new("RGB", (text_width if case["angle"] == 0 else 225, 36), "white")
        ImageDraw.Draw(patch).text((5, 6), text, font=font, fill="black")
        angle = 45 if case["challenge"] == "diagonal" else case["angle"]
        if angle:
            patch = (
                patch.rotate(angle, expand=True)
                if angle == 45
                else patch.transpose(
                    Image.Transpose.ROTATE_90 if angle == 90 else Image.Transpose.ROTATE_270
                )
            )
        left = x + i * px + (px - patch.width) // 2
        top = label_box[1] + 8 if case["placement"] == "below" else label_box[3] - patch.height - 8
        image.paste(patch, (left, top))
        token_boxes.append((left, top, left + patch.width, top + patch.height))
        if case["challenge"] == "conflicting_sides":
            image.paste(patch.transpose(Image.Transpose.ROTATE_180), (left, box[3] + 8))
    table, palette = M14["lut"](case["palette"])
    t = np.random.default_rng(case["seed"]).uniform(0.05, 0.95, (6, 8))
    t[0, :3] = [0, 0.5, 1]
    rgb = M14["colours"](t, table)
    image.paste(Image.fromarray(np.repeat(np.repeat(rgb, py, axis=0), px, axis=1)), (x, y))
    bar = (1360, y, 1376, box[3])
    profile = M14["colours"](np.linspace(0, 1, bar[3] - bar[1]), table)
    image.paste(Image.fromarray(np.repeat(profile[:, None], 16, axis=1)), (bar[0], bar[1]))
    draw = ImageDraw.Draw(image)
    for position, value in (
        (bar[1], "-3"),
        ((bar[1] + bar[3]) // 2, "-1" if case["challenge"] == "nonlinear" else "1"),
        (bar[3] - 1, "5"),
    ):
        draw.text((1382, position - 8), value, font=font, fill="black")
    if case["challenge"] in ("multiple_panels", "shared_bar"):
        image.paste(image.crop(box), (40, 630))
    if case["variant"] == "blur":
        image = image.filter(ImageFilter.GaussianBlur(0.65))
    elif case["variant"] == "jpeg":
        stream = BytesIO()
        image.save(stream, "JPEG", quality=85)
        stream.seek(0)
        image = Image.open(stream).convert("RGB")
    config = {
        "heatmap_bbox": box,
        "rows": 6,
        "columns": 8,
        "colourbar_bbox": bar,
        "colourbar_orientation": "vertical",
        "colourbar_start_value": -3,
        "colourbar_end_value": 5,
    }
    expected = [
        None
        if case["challenge"] == "missing" and i == 3 or case["challenge"] == "unlabelled"
        else s
        for i, s in enumerate(strings)
    ]
    return np.asarray(image), {
        "config": config,
        "labels": expected,
        "angle": case["angle"],
        "label_bbox": label_box,
        "values": (-3 + 8 * (t**2 if case["challenge"] == "nonlinear" else t)).tolist(),
        "scale_model": "nonlinear_unsupported" if case["challenge"] == "nonlinear" else "linear",
        "palette": palette,
        "label_patch_boxes": token_boxes,
        "font_sha256": hashlib.sha256(Path(FONTS[case["font"]]).read_bytes()).hexdigest(),
        "qa": {
            "patches_fit_column_bands": all(
                b is None or x + i * px <= b[0] < b[2] <= x + (i + 1) * px
                for i, b in enumerate(token_boxes)
            ),
            "labels_do_not_overlap_matrix": all(
                b is None or b[3] <= box[1] or b[1] >= box[3] for b in token_boxes
            ),
        },
    }
