"""Independent colour trajectory and position/value fixture variation; no OCR truth to runtime."""

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from heatmap_decoder.extract import extract, extract_labels
from heatmap_decoder.models import (
    ExtractionConfig,
    GridConfig,
    LabelsConfig,
    OCRLabelSource,
    ScaleAnchor,
    ScaleConfig,
)

THRESHOLDS = {
    "mae_fraction": 0.02,
    "max_fraction": 0.05,
    "finite_fraction": 1,
    "maximum_flagged_fraction": 0.01,
}


def fixture(seed, trajectory, nonlinear, vertical, reverse, derivative, challenge=None):
    rng = np.random.default_rng(seed)
    n = 257
    t = np.linspace(0, 1, n)
    if trajectory == "asymmetric":
        colour_knots = [0.0, 0.22, 0.67, 1.0]
        rgb_knots = [[185, 48, 164], [49, 35, 58], [12, 117, 128], [85, 229, 242]]
    else:
        colour_knots = [0, 0.45, 1]
        rgb_knots = [[182, 49, 158], [40, 40, 40], [55, 222, 233]]
    bar = (
        np.array([np.interp(t, colour_knots, np.array(rgb_knots)[:, i]) for i in range(3)])
        .T.round()
        .astype(np.uint8)
    )
    if challenge == "plateau":
        bar[70:150] = bar[90]
    if challenge == "repeated":
        bar[180:220] = bar[20:60]
    positions = [0, 46, 151, 256] if nonlinear else [0, 64, 192, 256]
    values = [-3, -2.4, 0.6, 5] if nonlinear else [-3, -1, 3, 5]
    if reverse:
        values = list(reversed(values))
    cell_positions = rng.integers(8, 248, (7, 8))
    if challenge in ["plateau", "repeated"]:
        cell_positions[0, 0] = 90 if challenge == "plateau" else 30
    if challenge == "edges":
        positions, values = [46, 151, 230], [-2.4, 0.6, 4]
        cell_positions[0, 0] = 10
        cell_positions[0, 1] = 247
    truth = np.interp(cell_positions, positions, values)
    image = np.full((330, 400, 3), 255, dtype=np.uint8)
    for row in range(7):
        for col in range(8):
            image[35 + row * 30 : 65 + row * 30, 35 + col * 30 : 65 + col * 30] = bar[
                cell_positions[row, col]
            ]
    bbox = (330, 30, 346, 287) if vertical else (35, 290, 292, 306)
    if vertical:
        image[30:287, 330:346] = bar[:, None, :]
    else:
        image[290:306, 35:292] = bar[None, :, :]
    im = Image.fromarray(image)
    painter = ImageDraw.Draw(im)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for row in range(7):
        painter.text((280, 42 + row * 30), f"G_{row}", fill="black", font=font)
    for col in range(8):
        painter.text((40 + col * 30, 12), f"S{col}", fill="black", font=font)
    if derivative == "blur":
        im = im.filter(ImageFilter.GaussianBlur(0.65))
    elif derivative == "jpeg":
        stream = io.BytesIO()
        im.save(stream, format="JPEG", quality=85)
        im = Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
    coordinate = 30 if vertical else 35
    anchors = tuple(
        ScaleAnchor(position=coordinate + p, value=v)
        for p, v in zip(positions, values, strict=True)
    )
    config = ExtractionConfig(
        heatmap_bbox=(35, 35, 275, 245),
        rows=7,
        columns=8,
        colourbar_bbox=bbox,
        colourbar_orientation="vertical" if vertical else "horizontal",
        scale=ScaleConfig(source="manual_ticks", model="piecewise_linear", anchors=anchors),
    )
    return np.array(im), config, truth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refuse overwrite of evidence")
    if args.split == "validation":
        frozen = json.loads(Path("benchmarks/m16/freeze-validation.json").read_text())
        for path, digest in frozen["sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError(f"changed frozen source: {path}")
    seed = 166200 if args.split == "validation" else 160100
    records = []
    for trajectory in ["diverging", "asymmetric"]:
        for nonlinear in [False, True]:
            for vertical in [False, True]:
                for reverse in [False, True]:
                    seed += 1
                    for derivative in ["clean", "jpeg", "blur"]:
                        image, c, truth = fixture(
                            seed, trajectory, nonlinear, vertical, reverse, derivative
                        )
                        r = extract(image, c)
                        label_config = GridConfig(
                            heatmap_bbox=c.heatmap_bbox,
                            rows=7,
                            columns=8,
                            labels=LabelsConfig(
                                rows=OCRLabelSource(source="ocr", bbox=(278, 35, 326, 245)),
                                columns=OCRLabelSource(source="ocr", bbox=(35, 5, 275, 30)),
                            ),
                        )
                        labels = extract_labels(image, label_config)
                        raw_exact = accepted = accepted_exact = aligned = 0
                        for axis in labels.axes:
                            for slot in axis.slots:
                                expected = (
                                    f"G_{slot.index}" if axis.axis == "rows" else f"S{slot.index}"
                                )
                                raw_exact += slot.raw_text == expected
                                accepted += slot.status == "accepted"
                                accepted_exact += (
                                    slot.status == "accepted" and slot.final_text == expected
                                )
                                aligned += slot.raw_text is not None and not any(
                                    issue in slot.issues for issue in ["spans_slots", "axis_offset"]
                                )
                        finite = np.isfinite(r.values)
                        delta = np.abs(r.values[finite] - truth[finite]) / 8
                        mae, maximum = (
                            float(delta.mean()) if len(delta) else None,
                            float(delta.max()) if len(delta) else None,
                        )
                        fraction, flagged = float(finite.mean()), float((r.flags != 0).mean())
                        accurate = fraction == 1 and mae < 0.02 and maximum <= 0.05
                        complete = fraction == 1 and flagged <= 0.01
                        records.append(
                            {
                                "base_seed": seed,
                                "label_slots": 15,
                                "raw_exact": raw_exact,
                                "heuristic_accepted": accepted,
                                "accepted_exact": accepted_exact,
                                "false_text_acceptance": accepted - accepted_exact,
                                "slots_spatially_aligned": aligned,
                                "trajectory": trajectory,
                                "nonlinear": nonlinear,
                                "vertical": vertical,
                                "reverse": reverse,
                                "derivative": derivative,
                                "mae_fraction": mae,
                                "max_fraction": maximum,
                                "finite_fraction": fraction,
                                "flagged_fraction": flagged,
                                "complete": complete,
                                "accurate": accurate,
                                "bound_cells": int(np.sum(r.range_limit != 0)),
                            }
                        )
    negatives = []
    for challenge in ["plateau", "repeated", "edges"]:
        for vertical in [False, True]:
            image, c, truth = fixture(
                seed + 1, "diverging", True, vertical, False, "clean", challenge
            )
            r = extract(image, c)
            negatives.append(
                {
                    "challenge": challenge,
                    "vertical": vertical,
                    "complete": bool(np.isfinite(r.values).all() and not np.any(r.flags)),
                    "finite_fraction": float(np.isfinite(r.values).mean()),
                    "first_flags": r.flags[0, :2].tolist(),
                }
            )
    report = {
        "split": args.split,
        "thresholds": THRESHOLDS,
        "ordinary_attempts": len(records),
        "complete": sum(r["complete"] for r in records),
        "accurate_full_coverage": sum(r["accurate"] for r in records),
        "wrong_complete": sum(r["complete"] and not r["accurate"] for r in records),
        "worst_mae_fraction": max(r["mae_fraction"] for r in records),
        "worst_max_fraction": max(r["max_fraction"] for r in records),
        "label_slots": sum(r["label_slots"] for r in records),
        "raw_exact": sum(r["raw_exact"] for r in records),
        "heuristic_accepted": sum(r["heuristic_accepted"] for r in records),
        "accepted_exact": sum(r["accepted_exact"] for r in records),
        "false_text_acceptance": sum(r["false_text_acceptance"] for r in records),
        "identifier_exports": 0,
        "negative_attempts": len(negatives),
        "negative_complete": sum(n["complete"] for n in negatives),
        "records": records,
        "negatives": negatives,
        "limitations": (
            "manual numeric calibration only; correlated derivatives; shared renderer; "
            "no held-out publication generalisation"
        ),
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ["records", "negatives"]}, indent=2)
    )


if __name__ == "__main__":
    main()
