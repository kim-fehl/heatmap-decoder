"""All attempted M8 synthetic bar/count cases; optional fixed OCR reference report."""

import argparse
import json
import runpy
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from heatmap_decoder import DetectionHints, detect, load_raster

HERE = Path(__file__).resolve().parent
FIXTURES = runpy.run_path(str(HERE / "generate.py"))


def region_metrics(candidate, truth):
    box = candidate.bbox
    intersection = max(0, min(box[2], truth[2]) - max(box[0], truth[0])) * max(
        0, min(box[3], truth[3]) - max(box[1], truth[1])
    )

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    return {
        "id": candidate.id,
        "bbox": box,
        "iou": intersection / (area(box) + area(truth) - intersection),
        "max_edge_pixels": int(np.max(np.abs(np.asarray(box) - truth))),
    }


def report(include_ocr=False):
    bars = {}
    for name in FIXTURES["LAYOUTS"]:
        image, annotation = FIXTURES["make_layout"](name)
        variants = {
            "clean": image,
            "blur": np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(0.8))),
        }
        stream = BytesIO()
        Image.fromarray(image).save(stream, format="JPEG", quality=65)
        variants["jpeg"] = np.asarray(Image.open(BytesIO(stream.getvalue())).convert("RGB"))
        for variant, pixels in variants.items():
            result = detect(
                pixels,
                DetectionHints(
                    rows=6,
                    columns=8,
                    heatmap_bbox=annotation["heatmap_bbox"],
                    requested=("colourbar",),
                ),
            )
            metrics = [
                region_metrics(c, annotation["colourbar_bbox"]) for c in result.candidates[:3]
            ]
            best = min(metrics, key=lambda m: m["max_edge_pixels"]) if metrics else None
            bars[f"{name}/{variant}"] = {
                "split": annotation["split"],
                "status": result.status,
                "candidates": len(result.candidates),
                "best_top3": best,
                "recall_within_1px": best is not None and best["max_edge_pixels"] <= 1,
                "correct_bar_association_offered": best is not None
                and any(a.colourbar_id == best["id"] for a in result.associations),
                "automatically_accepted": False,
            }
    counts = {}
    for kind in ("complete", "omitted", "merged", "uniform"):
        image, box, rows, columns = FIXTURES["count_fixture"](kind)
        result = detect(
            image,
            DetectionHints(heatmap_bbox=box, requested=(), infer_rows=True, infer_columns=True),
        )
        for axis, true in (("rows", rows), ("columns", columns)):
            proposals = [p.count for p in result.count_proposals if p.axis == axis]
            primary = proposals[0] if proposals else None
            counts[f"{kind}/{axis}"] = {
                "truth": true,
                "proposals": proposals,
                "status": result.status,
                "candidate_recall": true in proposals,
                "rank1_exact": primary == true,
                "rank1_off_by_one": primary is not None and abs(primary - true) == 1,
                "rank1_integer_multiple_error": primary is not None
                and primary != true
                and (primary % true == 0 or true % primary == 0),
                "abstained_proposals": not proposals,
                "automatically_accepted": False,
            }
    confounders = {}
    for kind in ("absent", "categorical_only", "two_bars"):
        image, annotation = FIXTURES["make_layout"]("vertical_sequential")
        image = image.copy()
        if kind in {"absent", "categorical_only"}:
            image[100:280, 410:424] = 250
        if kind == "absent":
            image[80:88, 130:370] = 250
        if kind == "two_bars":
            image[100:280, 90:104] = image[100:280, 410:424]
        result = detect(
            image,
            DetectionHints(
                rows=6, columns=8, heatmap_bbox=annotation["heatmap_bbox"], requested=("colourbar",)
            ),
        )
        confounders[kind] = {
            "status": result.status,
            "candidates": len(result.candidates),
            "associations": len(result.associations),
            "automatically_accepted": False,
        }
    labels = {}
    if include_ocr:
        fixture = HERE.parent / "labels"
        truth = json.loads((fixture / "truth.json").read_text())
        annotation = json.loads((HERE / "labels_annotation.json").read_text())
        for name in ("reconstruction.png", "reconstruction_unreadable.png"):
            result = detect(
                load_raster(fixture / name),
                DetectionHints(
                    rows=4,
                    columns=3,
                    heatmap_bbox=(220, 140, 1000, 460),
                    requested=("labels_rows", "labels_columns"),
                ),
            )
            entry = {
                "status": result.status,
                "candidates": [c.id for c in result.candidates],
                "warnings": result.warnings,
                "axes": {},
                "automatically_accepted": False,
            }
            for axis, side in (("rows", "left"), ("columns", "above")):
                record = next(
                    (e for e in result.label_evidence if e.axis == axis and e.side == side), None
                )
                if record is not None:
                    candidate = next(
                        (c for c in result.candidates if c.id == f"labels_{axis}-{side}"), None
                    )
                    slots = record.assignment.slots
                    entry["axes"][axis] = {
                        "slots": len(slots),
                        "region_metrics": region_metrics(
                            candidate, annotation["axes"][axis]["text_bounds"]
                        )
                        if candidate is not None
                        else None,
                        "annotated_edge_tolerance_pixels": annotation["edge_tolerance_pixels"],
                        "exact_accepted": sum(s.final_text == truth[axis][s.index] for s in slots),
                        "raw_exact": sum(s.raw_text == truth[axis][s.index] for s in slots),
                        "unresolved": sum(s.status != "accepted" for s in slots),
                        "engine_version": record.engine.engine_version,
                        "wrapper_version": record.engine.wrapper_version,
                        "preprocessing": record.preprocessing.model_dump(mode="json"),
                        "region_proposed": any(
                            c.id == f"labels_{axis}-{side}" for c in result.candidates
                        ),
                    }
            labels[name] = entry
    return {
        "detector_version": "m8.1",
        "bar_attempts": bars,
        "count_attempts": counts,
        "confounded_bar_attempts": confounders,
        "label_attempts": labels,
        "automatic_accepted_coverage": 0,
        "automatic_correct_acceptance_rate": None,
        "automatic_false_acceptances": 0,
        "independent_count_acceptance": "not implemented",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    print(json.dumps(report(args.ocr), indent=2, allow_nan=False))
