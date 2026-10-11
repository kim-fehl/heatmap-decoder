"""Report every fixed M7 attempt, including degradation and negative cases."""

import json
import runpy
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from heatmap_decoder import DetectionHints, detect

HERE = Path(__file__).resolve().parent
fixtures = runpy.run_path(str(HERE / "generate.py"))


def evaluate(image, box, rows, columns):
    result = detect(image, DetectionHints(rows=rows, columns=columns))
    errors = []
    for candidate in result.candidates[:3]:
        left, top, right, bottom = candidate.bbox
        intersection = max(0, min(right, box[2]) - max(left, box[0])) * max(
            0, min(bottom, box[3]) - max(top, box[1])
        )
        union = (
            (right - left) * (bottom - top) + (box[2] - box[0]) * (box[3] - box[1]) - intersection
        )
        pixels = np.abs(np.asarray(candidate.bbox) - box)
        pitch = np.array([(box[2] - box[0]) / columns, (box[3] - box[1]) / rows] * 2)
        errors.append(
            {
                "id": candidate.id,
                "iou": intersection / union,
                "max_edge_pixels": int(pixels.max()),
                "max_edge_pitch": float((pixels / pitch).max()),
            }
        )
    best = min(errors, key=lambda e: e["max_edge_pitch"]) if errors else None
    return {
        "status": result.status,
        "candidates": len(result.candidates),
        "best_top3": best,
        "recall_at_0.1_pitch": best is not None and best["max_edge_pitch"] <= 0.1,
        "automatically_accepted": False,
    }


def report():
    attempts = {}
    for name, definition in fixtures["FIXTURES"].items():
        image, box, rows, columns = fixtures["make_fixture"](name)
        variants = {
            "clean": image,
            "blur": np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(0.8))),
        }
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="JPEG", quality=65)
        variants["jpeg"] = np.asarray(Image.open(BytesIO(stream.getvalue())).convert("RGB"))
        for variant, pixels in variants.items():
            attempts[f"{name}/{variant}"] = {
                "split": definition[0],
                **evaluate(pixels, box, rows, columns),
            }
    for name in ("blank", "uniform", "noise", "gradient"):
        image = np.full((240, 300, 3), 245, dtype=np.uint8)
        if name == "uniform":
            image[30:210, 30:270] = [55, 90, 110]
        elif name == "noise":
            image[30:210, 30:270] = np.random.default_rng(44).integers(
                0, 256, (180, 240, 3), dtype=np.uint8
            )
        elif name == "gradient":
            image[30:210, 30:270] = np.linspace(20, 220, 240, dtype=np.uint8)[None, :, None]
        result = detect(image, DetectionHints(rows=8, columns=10))
        attempts[f"negative/{name}"] = {
            "status": result.status,
            "candidates": len(result.candidates),
            "automatically_accepted": False,
        }
    image, box, rows, columns = fixtures["make_fixture"]("sequential")
    attempts["confounded/multipanel"] = evaluate(
        np.concatenate((image, image), axis=1), box, rows, columns
    )
    return {
        "detector_version": "m7.1",
        "automatic_coverage": 0,
        "automatic_correct_acceptance_rate": None,
        "count_inference": "not implemented",
        "attempts": attempts,
    }


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, allow_nan=False))
