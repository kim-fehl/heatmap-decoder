"""Create the original MIT M9 hold-out corpus without importing the detector.

Families/base IDs/splits are declared before evaluation. Never tune the detector
on this corpus; any future tuning requires a new held-out dataset and audit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
FAMILIES = ("sequential_right", "diverging_top", "grayscale_left", "gridlines_below")


def palette(values, family):
    t = np.asarray(values)[..., None]
    if family == "grayscale_left":
        return np.repeat(np.rint(25 + 190 * t).astype(np.uint8), 3, axis=-1)
    if family == "sequential_right":
        return np.rint([25, 45, 70] + t * np.array([160, 140, 100])).astype(np.uint8)
    return np.rint(
        np.where(
            t < 0.5,
            [25, 50, 200] + 2 * t * np.array([210, 185, 35]),
            [235, 235, 235] + (2 * t - 1) * np.array([-35, -200, -210]),
        )
    ).astype(np.uint8)


def base_figure(family, index):
    rows, columns = ((6, 8), (9, 5), (5, 11))[index]
    labels = index == 0
    px, py = (100, 40) if labels else (18 + index * 3, 16 + index * 4)
    left, top = 160, 100
    right, bottom = left + columns * px, top + rows * py
    image = np.full((bottom + 130, right + 150, 3), 250, dtype=np.uint8)
    seed = 900913 + 100 * FAMILIES.index(family) + index
    values = np.random.default_rng(seed).uniform(0.05, 0.95, (rows, columns))
    cells = np.repeat(np.repeat(palette(values, family), py, axis=0), px, axis=1)
    if family == "gridlines_below":
        cells[py - 1 : -1 : py] = 255
        cells[:, px - 1 : -1 : px] = 255
        image[top - 1 : bottom + 1, left - 1 : right + 1] = 0
    image[top:bottom, left:right] = cells
    if family == "grayscale_left":
        bar = (90, top, 106, bottom)
    elif family == "sequential_right":
        bar = (right + 50, top, right + 66, bottom)
    elif family == "diverging_top":
        bar = (left, 32, right, 48)
    else:
        bar = (left, bottom + 70, right, bottom + 86)
    vertical = bar[3] - bar[1] > bar[2] - bar[0]
    profile = palette(np.linspace(0, 1, bar[3] - bar[1] if vertical else bar[2] - bar[0]), family)
    image[bar[1] : bar[3], bar[0] : bar[2]] = profile[:, None] if vertical else profile[None]
    figure = Image.fromarray(image)
    draw = ImageDraw.Draw(figure)
    draw.text((15, 5), f"M9 frozen {family} base {index}", fill="black")
    expected_labels = {}
    label_boxes = {}
    if labels:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
        row_text = [f"GENE_{i + 1:02d}" for i in range(rows)]
        col_text = [f"S_{i + 1:02d}" for i in range(columns)]
        row_side = "right" if family == "grayscale_left" else "left"
        col_side = "below" if family == "diverging_top" else "above"
        for axis, texts, side in (("rows", row_text, row_side), ("columns", col_text, col_side)):
            boxes = []
            for i, text in enumerate(texts):
                position = (
                    ((right + 75 if side == "right" else left - 75), top + (i + 0.5) * py)
                    if axis == "rows"
                    else (left + (i + 0.5) * px, bottom + 35 if side == "below" else top - 35)
                )
                draw.text(position, text, font=font, fill="black", anchor="mm")
                boxes.append(draw.textbbox(position, text, font=font, anchor="mm"))
            label_boxes[axis] = [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ]
            expected_labels[axis] = texts
    return figure, {
        "rows": rows,
        "columns": columns,
        "heatmap_bbox": [left, top, right, bottom],
        "colourbar_bbox": list(bar),
        "orientation": "vertical" if vertical else "horizontal",
        "values": values.tolist(),
        "labels": expected_labels,
        "label_bounds": label_boxes,
        "edge_tolerance_pixels": 1,
        "label_edge_tolerance_pixels": 7,
    }


def freeze():
    if (HERE / "manifest.json").exists():
        raise ValueError("M9 corpus is frozen; use a new corpus/version instead of overwriting it")
    entries = []
    for family in FAMILIES:
        for index in range(3):
            figure, annotation = base_figure(family, index)
            base = f"{family}-{index}"
            variants = {"clean": figure}
            if index == 1:
                variants["blur"] = figure.filter(ImageFilter.GaussianBlur(0.8))
                variants["jpeg"] = figure
                width, height = figure.size
                variants["downscaled"] = figure.resize(
                    (width // 2, height // 2), Image.Resampling.BOX
                )
            for variant, image in variants.items():
                name = f"{base}-{variant}"
                suffix = "jpg" if variant == "jpeg" else "png"
                path = HERE / "figures" / f"{name}.{suffix}"
                image.save(path, **({"quality": 65} if variant == "jpeg" else {}))
                record = json.loads(json.dumps(annotation))
                if variant == "downscaled":
                    sx, sy = image.width / figure.width, image.height / figure.height
                    # Ground-truth tolerance covers resize rounding, not inferred pitch.
                    for key in ("heatmap_bbox", "colourbar_bbox"):
                        record[key] = [
                            round(v * (sx if i % 2 == 0 else sy)) for i, v in enumerate(record[key])
                        ]
                entries.append(
                    {
                        "id": name,
                        "base_id": base,
                        "family": family,
                        "split": "held_out",
                        "source": "original synthetic raster; MIT",
                        "image": f"figures/{path.name}",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "variant": variant,
                        "negative": False,
                        **record,
                    }
                )
    # No unique numerical counts can be inferred for uniform/low-contrast cases.
    for kind in (
        "uniform",
        "low_contrast",
        "photographic_noise",
        "gradient",
        "categorical_track",
        "absent_bar",
        "multipanel",
        "annotation_dendrogram",
    ):
        figure, annotation = base_figure("sequential_right", 2)
        pixels = np.asarray(figure).copy()
        left, top, right, bottom = annotation["heatmap_bbox"]
        if kind == "uniform":
            pixels[top:bottom, left:right] = [90, 110, 130]
        elif kind == "low_contrast":
            pixels[top:bottom, left:right] = np.random.default_rng(901).integers(
                126, 130, (bottom - top, right - left, 3), dtype=np.uint8
            )
        elif kind == "photographic_noise":
            pixels[top:bottom, left:right] = np.random.default_rng(902).integers(
                0, 256, (bottom - top, right - left, 3), dtype=np.uint8
            )
        elif kind == "gradient":
            pixels[top:bottom, left:right] = np.linspace(30, 210, right - left, dtype=np.uint8)[
                None, :, None
            ]
        elif kind == "categorical_track":
            pixels[top:bottom, left:right] = 250
            for i in range(annotation["columns"]):
                pixels[top : top + 8, left + i * 24 : left + (i + 1) * 24] = (
                    [220, 40, 40] if i % 2 else [30, 160, 40]
                )
        elif kind == "absent_bar":
            x0, y0, x1, y1 = annotation["colourbar_bbox"]
            pixels[y0:y1, x0:x1] = 250
            annotation["colourbar_bbox"] = None
        elif kind == "multipanel":
            pixels = np.concatenate((pixels, pixels), axis=1)
        else:
            # Separable annotation strips and dendrogram/caption confounders.
            pixels[top - 16 : top - 8, left:right] = [120, 30, 200]
            canvas = Image.fromarray(pixels)
            draw = ImageDraw.Draw(canvas)
            draw.line(
                (left - 30, top, left - 10, top, left - 10, bottom, left - 30, bottom),
                fill="black",
                width=2,
            )
            pixels = np.asarray(canvas)
        image = Image.fromarray(pixels)
        path = HERE / "figures" / f"challenge-{kind}.png"
        image.save(path)
        entries.append(
            {
                "id": f"challenge-{kind}",
                "base_id": "negative-base",
                "family": "confounders",
                "split": "held_out",
                "source": "original synthetic raster; MIT",
                "image": f"figures/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "variant": kind,
                "negative": kind not in {"absent_bar", "multipanel", "annotation_dendrogram"},
                "confounded": True,
                **annotation,
            }
        )
    # Fresh count challenge base: all variants stay together in the held-out split.
    for kind in ("complete", "omitted", "merged", "uniform"):
        figure, annotation = base_figure("gridlines_below", 2)
        pixels = np.asarray(figure).copy()
        left, top, right, bottom = annotation["heatmap_bbox"]
        positions = np.asarray(annotation["values"])
        if kind == "merged":
            positions[1::2] = positions[::2][: len(positions[1::2])]
            positions[:, 1::2] = positions[:, ::2][:, : len(positions[:, 1::2].T)]
        if kind == "uniform":
            positions[:] = 0.5
        cells = np.repeat(np.repeat(palette(positions, "gridlines_below"), 24, axis=0), 24, axis=1)
        if kind in {"complete", "omitted"}:
            for axis, count in ((0, annotation["rows"]), (1, annotation["columns"])):
                for i in range(1, count):
                    if kind == "complete" or i % 2:
                        if axis == 0:
                            cells[i * 24 - 1] = 0
                        else:
                            cells[:, i * 24 - 1] = 0
        pixels[top:bottom, left:right] = cells
        path = HERE / "figures" / f"counts-{kind}.png"
        Image.fromarray(pixels).save(path)
        annotation["values"] = positions.tolist()
        entries.append(
            {
                "id": f"counts-{kind}",
                "base_id": "counts-base",
                "family": "count_challenge",
                "split": "held_out",
                "source": "original synthetic raster; MIT",
                "image": f"figures/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "variant": kind,
                "negative": False,
                "count_only": True,
                **annotation,
            }
        )
    manifest = {
        "schema_version": 1,
        "corpus_version": "m9-held-out.1",
        "baseline_commit": "5a82b8d05dae28f5e9f91e5af175968d189290a0",
        "split_frozen_before_evaluation": True,
        "published_real_figure_count": 0,
        "real_figure_limitation": (
            "No independently annotated published figures; practical coverage is not established"
        ),
        "entries": entries,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (HERE / "manifest.json").write_bytes(data)
    (HERE / "manifest.sha256").write_text(hashlib.sha256(data).hexdigest() + "\n", encoding="utf-8")
    print(f"Frozen {len(entries)} figures in m9-held-out.1 before evaluation")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    freeze()
