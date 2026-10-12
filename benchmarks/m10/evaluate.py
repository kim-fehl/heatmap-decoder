"""Synthetic M10 development evaluation; runtime decides before truth is consulted.

Historical M9 is inspected development evidence. Never modifies its frozen files.
No M12 hold-out, independent real-figure or release validation claims.
"""

import argparse
import json
import platform
import runpy
import time
from collections import Counter
from pathlib import Path

import numpy as np

from heatmap_decoder import DetectionHints, ExtractionConfig, detect, extract, load_raster
from heatmap_decoder.detection_policy import policy_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/m10/generate.py"))


def attempt(identifier, image, truth, group, negative=False):
    started = time.perf_counter()
    settings = DetectionHints(
        rows=truth["rows"],
        columns=truth["columns"],
        requested=("heatmap", "colourbar"),
        colourbar_start_value=2,
        colourbar_end_value=-2,
    )
    result = detect(image, settings)  # Decisions precede truth-based scoring.
    duration = time.perf_counter() - started
    report = result.acceptance
    decisions = {d.requirement: d for d in report.decisions}
    metrics = {}
    for kind in ("heatmap", "colourbar"):
        expected = truth.get(f"{kind}_bbox")
        candidates = [c for c in result.candidates if c.kind == kind]
        if expected is None:
            continue
        tolerance = (
            np.array(
                [
                    (expected[2] - expected[0]) / truth["columns"],
                    (expected[3] - expected[1]) / truth["rows"],
                ]
                * 2
            )
            * 0.1
            if kind == "heatmap"
            else np.ones(4)
        )
        metric = {
            "top_three_within_tolerance": any(
                np.all(np.abs(np.array(c.bbox) - expected) <= tolerance) for c in candidates[:3]
            )
        }
        chosen = decisions[kind]
        if chosen.status == "accepted" and chosen.candidate_ids:
            selected = next((c for c in candidates if c.id == chosen.candidate_ids[0]), None)
            if selected is not None:
                errors = np.abs(np.array(selected.bbox) - expected)
                metric.update(
                    edge_errors_pixels=errors.tolist(),
                    accepted_within_tolerance=bool(np.all(errors <= tolerance)),
                )
        metrics[kind] = metric
    numeric = decisions.get("numeric_quality")
    if numeric is not None and "concrete_config" in numeric.evidence and truth.get("values"):
        reconstructed = extract(
            image, ExtractionConfig.model_validate(numeric.evidence["concrete_config"])
        )
        expected = 2 - 4 * np.array(truth["values"])
        finite = np.isfinite(reconstructed.values)
        errors = np.abs(reconstructed.values[finite] - expected[finite]) / 4
        metrics["numeric"] = {
            "finite_fraction": float(finite.mean()),
            "mae_fraction_of_range": float(errors.mean()) if len(errors) else None,
            "maximum_error_fraction_of_range": float(errors.max()) if len(errors) else None,
            "flagged_fraction": float(np.mean(reconstructed.flags != 0)),
        }
    return {
        "id": identifier,
        "group": group,
        "negative": negative,
        "rows": truth["rows"],
        "columns": truth["columns"],
        "image_shape": list(image.shape),
        "seconds": duration,
        "acceptance": report.model_dump(mode="json"),
        "metrics": metrics,
    }


def challenges():
    for name in (
        "absent_bar",
        "categorical",
        "clipped",
        "partial",
        "competing_bars",
        "uniform",
        "wrong_counts",
        "multipanel",
    ):
        image, truth = FIXTURE["make_layout"](rows=20, columns=6)
        image = image.copy()
        x, y, r, s = truth["colourbar_bbox"]
        if name == "absent_bar":
            image[y:s, x:r] = 255
            truth["colourbar_bbox"] = None
        elif name == "categorical":
            image[y:s, x:r] = np.repeat(
                np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8), (s - y) // 2, axis=0
            )[:, None]
        elif name == "clipped":
            image[y:s, :14] = image[y:s, x:r]
            image[y:s, x:r] = 255
            truth["colourbar_bbox"] = (0, y, 14, s)
        elif name == "partial":
            image[y : (y + s) // 2, x:r] = 255
        elif name == "competing_bars":
            image[y:s, 40:54] = image[y:s, x:r]
        elif name == "uniform":
            left, top, right, bottom = truth["heatmap_bbox"]
            image[top:bottom, left:right] = 100
        elif name == "wrong_counts":
            truth["rows"] += 1
        else:
            width = image.shape[1]
            image = np.pad(image, ((0, 0), (0, width), (0, 0)), constant_values=255)
            image[:, width:] = image[:, :width]
        yield name, image, truth


def summarise(attempts):
    groups = {}
    for group in sorted({a["group"] for a in attempts}):
        subset = [a for a in attempts if a["group"] == group]
        accepted = [a for a in subset if a["acceptance"]["overall"]["status"] == "accepted"]
        groups[group] = {
            "attempts": len(subset),
            "development_complete_accepted": len(accepted),
            "development_acceptance_coverage": len(accepted) / len(subset),
            "negative_development_acceptances": sum(a["negative"] for a in accepted),
            "released_automatic_exports": 0,
            "statuses": dict(Counter(a["acceptance"]["overall"]["status"] for a in subset)),
            "roi_pair_development_accepted": sum(
                all(
                    d["status"] == "accepted"
                    for d in a["acceptance"]["decisions"]
                    if d["requirement"] in {"heatmap", "colourbar", "association"}
                )
                for a in subset
            ),
            "correct_accepted_geometry": sum(
                all(
                    m.get("accepted_within_tolerance", False)
                    for k, m in a["metrics"].items()
                    if k in {"heatmap", "colourbar"}
                )
                for a in accepted
            ),
            "bar_top_three_recall": sum(
                a["metrics"].get("colourbar", {}).get("top_three_within_tolerance", False)
                for a in subset
            ),
            "bar_present_denominator": sum("colourbar" in a["metrics"] for a in subset),
            "median_seconds": float(np.median([a["seconds"] for a in subset])),
            "maximum_seconds": max(a["seconds"] for a in subset),
        }
    return groups


def ocr_attempts():
    from heatmap_decoder import extract_labels
    from heatmap_decoder.models import GridConfig, LabelsConfig

    results = []
    for row_side, column_side in (("left", "top"), ("right", "bottom")):
        for degradation in ("clean", "blur", "jpeg"):
            image, truth = FIXTURE["make_layout"](
                rows=30,
                columns=10,
                row_side=row_side,
                column_side=column_side,
                degradation=degradation,
            )
            config = GridConfig(
                heatmap_bbox=truth["heatmap_bbox"],
                rows=30,
                columns=10,
                labels=LabelsConfig.model_validate(
                    {
                        axis: {"source": "ocr", "bbox": box}
                        for axis, box in truth["label_boxes"].items()
                    }
                ),
            )
            recognised = extract_labels(image, config)
            slots = []
            for axis in recognised.axes:
                expected = truth["genes"] if axis.axis == "rows" else truth["samples"]
                for slot in axis.slots:
                    slots.append(
                        {
                            "axis": axis.axis,
                            "index": slot.index,
                            "expected": expected[slot.index],
                            "raw": slot.raw_text,
                            "suggestion": slot.final_text,
                            "confidence": slot.confidence,
                            "status": slot.status,
                            "exact": slot.final_text == expected[slot.index],
                        }
                    )
            results.append(
                {
                    "row_side": row_side,
                    "column_side": column_side,
                    "degradation": degradation,
                    "slots": slots,
                    "unresolved": recognised.unresolved_count,
                    "exact": sum(s["exact"] for s in slots),
                    "denominator": 40,
                    "identifier_verification": "review_required",
                    "engines": [r.engine_version for _, r in recognised.ocr_runs],
                }
            )
    return results


def count_attempts():
    """Conditional count experiments: manual boxes, no true counts injected."""
    observations = []
    corpus = ROOT / "benchmarks/m9"
    figures = []
    for entry in json.loads((corpus / "manifest.json").read_text())["entries"]:
        if entry["id"].startswith("counts-"):
            figures.append((entry["id"], load_raster(corpus / entry["image"]), entry))
    for rows, columns in ((20, 6), (25, 8), (30, 10)):
        for gridlines in (False, True):
            image, truth = FIXTURE["make_layout"](rows=rows, columns=columns, gridlines=gridlines)
            figures.append((f"omics-count-{rows}-{columns}-lines-{gridlines}", image, truth))
    for identifier, image, truth in figures:
        result = detect(
            image,
            DetectionHints(
                heatmap_bbox=truth["heatmap_bbox"],
                requested=(),
                infer_rows=True,
                infer_columns=True,
            ),
        )
        for axis in ("rows", "columns"):
            d = next(d for d in result.acceptance.decisions if d.requirement == axis)
            proposals = [p for p in result.count_proposals if p.axis == axis]
            observations.append(
                {
                    "id": identifier,
                    "axis": axis,
                    "condition": "manual_bounds",
                    "decision": d.model_dump(mode="json"),
                    "true_count": truth[axis],
                    "first_ranked_count": proposals[0].count if proposals else None,
                    "true_count_offered": any(p.count == truth[axis] for p in proposals),
                    "first_ranked_exact": bool(proposals and proposals[0].count == truth[axis]),
                }
            )
    return observations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("work/m10-report.json"))
    parser.add_argument("--ocr", action="store_true")
    args = parser.parse_args()
    attempts = []
    for index, options in enumerate(FIXTURE["layouts"]()):
        for degradation in ("clean", "blur", "jpeg"):
            image, truth = FIXTURE["make_layout"](**options, degradation=degradation)
            attempts.append(
                attempt(f"layout-{index:02d}-{degradation}", image, truth, f"omics_{degradation}")
            )
        print(f"evaluated omics layout {index + 1}/32", flush=True)
    for name, image, truth in challenges():
        attempts.append(attempt(name, image, truth, "challenge", negative=True))
    corpus = ROOT / "benchmarks/m9"
    for entry in json.loads((corpus / "manifest.json").read_text())["entries"]:
        if entry["id"].startswith("counts-"):
            continue
        attempts.append(
            attempt(
                entry["id"],
                load_raster(corpus / entry["image"]),
                entry,
                "historical_m9",
                entry["negative"],
            )
        )
    report = {
        "purpose": "synthetic development evidence; no M12 hold-out or real figures",
        "runtime_policy": policy_snapshot(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "count_experiments": count_attempts(),
        "attempts": attempts,
        "summary": summarise(attempts),
        "ocr": ocr_attempts() if args.ocr else [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # NumPy boolean metric scalars are converted at the JSON boundary.
    args.out.write_text(
        json.dumps(report, indent=2, allow_nan=False, default=lambda x: x.item()) + "\n"
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
