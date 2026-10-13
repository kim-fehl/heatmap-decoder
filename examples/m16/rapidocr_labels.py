"""Optional RapidOCR review-candidate trial; no automatic label acceptance.

Run in an isolated environment with rapidocr-onnxruntime==1.4.4. This deliberately
writes a separate candidate format: recognizer scores and whole-field boxes must
not be passed off as calibrated confidence or detected glyph boxes.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import rapidocr_onnxruntime
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--white-padding", type=int, choices=range(17), default=0)
    parser.add_argument("--upscale", type=int, choices=range(1, 9), default=1)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    image = Image.open(args.image).convert("L")
    engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    records = []
    for axis, coordinate in [("rows", 1), ("columns", 0)]:
        source = config["labels"].get(axis)
        if source is None:
            continue
        if source["source"] != "ocr" or source.get("recognition") != "slots":
            raise ValueError("This experiment requires explicit OCR slots")
        count = config[axis]
        if not isinstance(count, int) or not 1 <= count <= 128:
            raise ValueError("Each axis requires 1..128 explicit slots")
        start, end = (config["heatmap_bbox"][coordinate + i] for i in [0, 2])
        for index in range(count):
            box = list(source["bbox"])
            box[coordinate] = max(
                box[coordinate], math.floor(start + index * (end - start) / count)
            )
            box[coordinate + 2] = min(
                box[coordinate + 2], math.ceil(start + (index + 1) * (end - start) / count)
            )
            pad = source.get("crop_padding_pixels", 0)
            box = [
                max(0, box[0] - pad),
                max(0, box[1] - pad),
                min(image.width, box[2] + pad + source.get("crop_right_padding_pixels", 0)),
                min(image.height, box[3] + pad),
            ]
            crop = image.crop(box)
            angle = source.get("text_rotation_degrees", 0)
            if angle not in (0, 90, -90):
                raise ValueError("Only explicit 0/90/-90 text rotations are supported")
            if angle:
                crop = crop.transpose(
                    Image.Transpose.ROTATE_270 if angle == 90 else Image.Transpose.ROTATE_90
                )
            if args.upscale != 1:
                crop = crop.resize(
                    (crop.width * args.upscale, crop.height * args.upscale),
                    Image.Resampling.BICUBIC,
                )
            crop = ImageOps.expand(crop, border=args.white_padding * args.upscale, fill=255)
            result, elapsed = engine(np.asarray(crop), use_det=False, use_cls=False)
            records.append(
                {
                    "axis": axis,
                    "index": index,
                    "crop_bbox": box,
                    "raw_text": " ".join(r[0] for r in result or []),
                    "recognizer_score": min((r[1] for r in result or []), default=None),
                    "raw_result": result,
                    "engine_elapsed": elapsed,
                    "status": "needs_review",
                    "final_text": None,
                }
            )
    root = Path(rapidocr_onnxruntime.__file__).parent
    result = {
        "schema_version": 1,
        "operation": "experimental_label_candidates",
        "input": {
            "filename": str(args.image),
            "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        },
        "config": config,
        "preprocessing": {
            "grayscale": "Pillow L",
            "external_upscale": args.upscale,
            "white_padding_before_scale_units": args.white_padding,
            "padding_order": "after external resizing",
            "note": "Overrides configured OCR upscaling/white padding only",
        },
        "engine": {
            "name": "RapidOCR",
            "detector": False,
            "orientation_classifier": False,
            "intra_op_threads": 1,
            "inter_op_threads": 1,
            "packages": {
                n: importlib.metadata.version(n)
                for n in ["rapidocr-onnxruntime", "onnxruntime", "numpy", "Pillow", "opencv-python"]
            },
            "model_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (root / "models").glob("*.onnx")
            },
        },
        "geometry": "crop_bbox is a source field, not a detected glyph box",
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(records)} review candidates to {args.out}")


if __name__ == "__main__":
    main()
