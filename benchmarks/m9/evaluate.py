"""Evaluate every frozen M9 input; report proposal quality separately from acceptance.

Oracle-assisted coordinate review is an evaluation comparison, never an automatic
selection policy. Count inference is evaluated on annotated boxes as a separate
conditional task, and OCR is conditional on a reviewed heatmap and true counts.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from PIL import __version__ as pillow_version

from heatmap_decoder import (
    DetectionHints,
    ExtractionConfig,
    __version__,
    detect,
    extract,
    load_raster,
)
from heatmap_decoder.detection import DETECTOR_VERSION
from heatmap_decoder.detection_policy import policy_snapshot, threshold_snapshot

HERE = Path(__file__).resolve().parent


def load_frozen(directory=HERE):
    data = (directory / "manifest.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != (directory / "manifest.sha256").read_text().strip():
        raise ValueError("frozen benchmark manifest hash mismatch")
    manifest = json.loads(data)
    ids = [entry["id"] for entry in manifest["entries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate frozen fixture ID")
    splits = {}
    for entry in manifest["entries"]:
        if splits.setdefault(entry["base_id"], entry["split"]) != entry["split"]:
            raise ValueError("transforms of a base figure cross benchmark splits")
        relative = Path(entry["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("benchmark image must be inside the frozen corpus")
        if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"frozen image hash mismatch: {entry['id']}")
    return manifest, digest


def region_metric(candidate, truth, pitch, tolerance):
    box = np.asarray(candidate.bbox)
    edges = np.abs(box - truth)
    intersection = max(0, min(box[2], truth[2]) - max(box[0], truth[0])) * max(
        0, min(box[3], truth[3]) - max(box[1], truth[1])
    )

    def area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    union = area(box) + area(truth) - intersection
    return {
        "id": candidate.id,
        "bbox": candidate.bbox,
        "iou": float(intersection / union),
        "edge_error_pixels": edges.tolist(),
        "edge_error_cell_pitch": (edges / pitch).tolist(),
        "within_tolerance": bool(np.all(edges <= tolerance)),
    }


def roi_metric(candidates, kind, truth, pitch, tolerance):
    proposals = [c for c in candidates if c.kind == kind]
    if truth is None:
        return {
            "truth_present": False,
            "proposal_count": len(proposals),
            "top1": None,
            "best_top3": None,
        }
    metrics = [region_metric(c, truth, pitch, tolerance) for c in proposals[:3]]
    best = min(metrics, key=lambda m: max(m["edge_error_pixels"])) if metrics else None
    return {
        "truth_present": True,
        "proposal_count": len(proposals),
        "top1": metrics[0] if metrics else None,
        "best_top3": best,
        "top3_recall": best is not None and best["within_tolerance"],
    }


def numeric_comparison(image, entry, heatmap, bar):
    if not heatmap or not bar or entry["negative"]:
        return {"status": "not_compared_missing_proposals_or_negative"}
    if not heatmap["within_tolerance"] or not bar["within_tolerance"]:
        return {"status": "not_compared_failed_region_tolerance"}
    settings = {
        "rows": entry["rows"],
        "columns": entry["columns"],
        "colourbar_orientation": entry["orientation"],
        "colourbar_start_value": 0,
        "colourbar_end_value": 1,
    }
    manual = extract(
        image,
        ExtractionConfig(
            **settings, heatmap_bbox=entry["heatmap_bbox"], colourbar_bbox=entry["colourbar_bbox"]
        ),
    )
    reviewed = extract(
        image,
        ExtractionConfig(**settings, heatmap_bbox=heatmap["bbox"], colourbar_bbox=bar["bbox"]),
    )
    overlap = np.isfinite(manual.values) & np.isfinite(reviewed.values)
    truth = np.asarray(entry["values"])
    return {
        "status": "oracle_reviewed_comparison",
        "automatically_accepted": False,
        "true_counts_used": True,
        "finite_overlap_cells": int(overlap.sum()),
        "reviewed_vs_manual_mae": float(
            np.mean(np.abs(reviewed.values[overlap] - manual.values[overlap]))
        )
        if overlap.any()
        else None,
        "reviewed_vs_manual_max_error": float(
            np.max(np.abs(reviewed.values[overlap] - manual.values[overlap]))
        )
        if overlap.any()
        else None,
        "reviewed_vs_truth_mae": float(np.mean(np.abs(reviewed.values[overlap] - truth[overlap])))
        if overlap.any()
        else None,
        "flagged_fraction": float(np.mean(reviewed.flags != 0)),
        "missing_or_ambiguous_fraction": float(np.mean(~np.isfinite(reviewed.values))),
        "scale_range": 1.0,
    }


def label_metrics(image, entry):
    hints = DetectionHints(
        rows=entry["rows"],
        columns=entry["columns"],
        heatmap_bbox=entry["heatmap_bbox"],
        requested=("labels_rows", "labels_columns"),
    )
    result = detect(image, hints)
    left, top, right, bottom = entry["heatmap_bbox"]
    pitch = np.array([(right - left) / entry["columns"], (bottom - top) / entry["rows"]] * 2)
    axes = {}
    for axis, truth in entry["labels"].items():
        records = [r for r in result.label_evidence if r.axis == axis and r.assignment is not None]
        record = (
            max(records, key=lambda r: sum(s.status == "accepted" for s in r.assignment.slots))
            if records
            else None
        )
        metric = roi_metric(
            result.candidates,
            f"labels_{axis}",
            entry["label_bounds"][axis],
            pitch,
            entry["label_edge_tolerance_pixels"],
        )
        slots = record.assignment.slots if record is not None else ()
        slot_records = []
        for slot in slots:
            coordinate = 1 if axis == "rows" else 0
            start = entry["heatmap_bbox"][coordinate]
            cell_pitch = pitch[coordinate]
            band_start, band_end = (
                start + slot.index * cell_pitch,
                start + (slot.index + 1) * cell_pitch,
            )
            bounds = entry["label_bounds"][axis]
            expected = list(bounds)
            expected[coordinate] = max(bounds[coordinate], band_start)
            expected[coordinate + 2] = min(bounds[coordinate + 2], band_end)
            associated = slot.bbox is not None and all(
                (
                    band_start <= slot.bbox[coordinate] < slot.bbox[coordinate + 2] <= band_end,
                    max(slot.bbox[0], expected[0]) < min(slot.bbox[2], expected[2]),
                    max(slot.bbox[1], expected[1]) < min(slot.bbox[3], expected[3]),
                )
            )
            slot_records.append(
                {
                    "index": slot.index,
                    "expected_text": truth[slot.index],
                    "raw_text": slot.raw_text,
                    "final_text": slot.final_text,
                    "confidence": slot.confidence,
                    "status": slot.status,
                    "bbox": slot.bbox,
                    "geometry_association_correct": associated,
                }
            )
        axes[axis] = {
            "region": metric,
            "slot_count": len(slots),
            "true_slots": len(truth),
            "accepted_texts": sum(s.status == "accepted" for s in slots),
            "false_accepted_texts": sum(
                s.status == "accepted" and s.final_text != truth[s.index] for s in slots
            ),
            "slot_evidence": slot_records,
            "exact_accepted": sum(s.final_text == truth[s.index] for s in slots),
            "raw_exact": sum(s.raw_text == truth[s.index] for s in slots),
            "unresolved": sum(s.status != "accepted" for s in slots) if slots else len(truth),
            "association_correct": all(
                s["geometry_association_correct"] for s in slot_records if s["status"] == "accepted"
            )
            if any(s["status"] == "accepted" for s in slot_records)
            else None,
            "engine_version": record.engine.engine_version if record is not None else None,
            "wrapper_version": record.engine.wrapper_version if record is not None else None,
        }
    return {
        "condition": "reviewed heatmap and true supplied counts; OCR not used for ROI gate",
        "status": result.status,
        "axes": axes,
        "warnings": result.warnings,
    }


def evaluate_entry(entry, include_ocr=False):
    image = load_raster(HERE / entry["image"])
    left, top, right, bottom = entry["heatmap_bbox"]
    pitch = np.array([(right - left) / entry["columns"], (bottom - top) / entry["rows"]] * 2)
    record = {
        "id": entry["id"],
        "base_id": entry["base_id"],
        "family": entry["family"],
        "split": entry["split"],
        "variant": entry["variant"],
        "negative": entry["negative"],
        "automatically_accepted_layout": False,
        "automatically_accepted_counts": {"rows": False, "columns": False},
    }
    try:
        if not entry.get("count_only"):
            result = detect(
                image,
                DetectionHints(
                    rows=entry["rows"], columns=entry["columns"], requested=("heatmap", "colourbar")
                ),
            )
            hm = roi_metric(result.candidates, "heatmap", entry["heatmap_bbox"], pitch, 0.1 * pitch)
            bar = roi_metric(
                result.candidates,
                "colourbar",
                entry["colourbar_bbox"],
                pitch,
                entry["edge_tolerance_pixels"],
            )
            best_hm, best_bar = hm.get("best_top3"), bar.get("best_top3")
            correct_pair = (
                any(
                    a.heatmap_id == best_hm["id"] and a.colourbar_id == best_bar["id"]
                    for a in result.associations
                )
                if best_hm and best_bar
                else False
            )
            record["roi"] = {
                "status": result.status,
                "heatmap": hm,
                "colourbar": bar,
                "correct_pair_offered": correct_pair,
                "association_count": len(result.associations),
                "complete_proposal_available": bool(
                    hm.get("top3_recall") and bar.get("top3_recall") and correct_pair
                ),
                "unresolved_requirements": result.unresolved_requirements,
            }
            record["numerical"] = numeric_comparison(image, entry, best_hm, best_bar)
        else:
            record["roi"] = {"status": "not_requested_count_challenge"}
        result = detect(
            image,
            DetectionHints(
                heatmap_bbox=entry["heatmap_bbox"],
                requested=(),
                infer_rows=True,
                infer_columns=True,
            ),
        )
        record["counts"] = {}
        for axis in ("rows", "columns"):
            proposals = [p for p in result.count_proposals if p.axis == axis]
            count = proposals[0].count if proposals else None
            truth = entry[axis]
            record["counts"][axis] = {
                "condition": "annotated heatmap; no ROI inference",
                "truth": truth,
                "proposals": [p.count for p in proposals],
                "candidate_recall": any(p.count == truth for p in proposals),
                "rank1_exact": count == truth,
                "rank1_off_by_one": count is not None and abs(count - truth) == 1,
                "rank1_integer_multiple_error": count is not None
                and count != truth
                and (count % truth == 0 or truth % count == 0),
                "proposal_abstention": not proposals,
                "harmonic_ambiguity": any(
                    "integer_multiple_ambiguity" in p.reasons for p in proposals
                ),
                "rank1_signals": list(proposals[0].signals) if proposals else [],
                "independent_support_validated": False,
            }
        if entry.get("labels"):
            record["labels"] = (
                label_metrics(image, entry) if include_ocr else {"status": "not_run_ocr_disabled"}
            )
    except (ValueError, OSError) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def summarise(records):
    roi = [r for r in records if "heatmap" in r.get("roi", {})]
    positives = [r for r in roi if not r["negative"]]
    negatives = [r for r in roi if r["negative"]]
    numeric = [
        r["numerical"]
        for r in records
        if r.get("numerical", {}).get("status") == "oracle_reviewed_comparison"
    ]
    summary = {
        "attempted_figures": len(records),
        "evaluation_errors": sum("error" in r for r in records),
        "roi_attempts": len(roi),
        "positive_roi_attempts": len(positives),
        "negative_roi_attempts": len(negatives),
        "heatmap_top3_recall_count": sum(
            r["roi"]["heatmap"].get("top3_recall", False) for r in positives
        ),
        "bar_present_attempts": sum(
            r["roi"]["colourbar"].get("truth_present", False) for r in positives
        ),
        "bar_top3_recall_count": sum(
            r["roi"]["colourbar"].get("top3_recall", False) for r in positives
        ),
        "complete_proposals_available": sum(
            r["roi"].get("complete_proposal_available", False) for r in positives
        ),
        "roi_no_proposal_abstentions": sum(
            r["roi"]["heatmap"]["proposal_count"] == 0
            or r["roi"]["colourbar"]["proposal_count"] == 0
            for r in roi
        ),
        "negative_heatmap_proposals": sum(
            r["roi"]["heatmap"]["proposal_count"] > 0 for r in negatives
        ),
        "automatic_layout_acceptances": 0,
        "automatic_layout_coverage": 0.0,
        "automatic_acceptance_abstention_rate": 1.0,
        "automatic_correct_acceptance_rate": None,
        "automatic_false_acceptances": 0,
        "oracle_reviewed_numeric_comparisons": len(numeric),
        "maximum_reviewed_vs_manual_mae": max(
            (
                n["reviewed_vs_manual_mae"]
                for n in numeric
                if n["reviewed_vs_manual_mae"] is not None
            ),
            default=None,
        ),
        "counts": {},
    }
    for axis in ("rows", "columns"):
        attempts = [r["counts"][axis] for r in records if "counts" in r and not r["negative"]]
        summary["counts"][axis] = {
            "attempts": len(attempts),
            "candidate_recall_count": sum(a["candidate_recall"] for a in attempts),
            "proposal_coverage_count": sum(not a["proposal_abstention"] for a in attempts),
            "proposal_abstention_count": sum(a["proposal_abstention"] for a in attempts),
            "rank1_exact_count": sum(a["rank1_exact"] for a in attempts),
            "rank1_off_by_one_errors": sum(a["rank1_off_by_one"] for a in attempts),
            "rank1_integer_multiple_errors": sum(
                a["rank1_integer_multiple_error"] for a in attempts
            ),
            "automatic_acceptances": 0,
            "automatic_exact_count_accuracy": None,
            "automatic_coverage": 0.0,
        }
    labels = [r["labels"] for r in records if "axes" in r.get("labels", {})]
    axes = [a for r in labels for a in r["axes"].values()]
    summary["labels"] = {
        "conditional_figure_attempts": len(labels),
        "slots": sum(a["true_slots"] for a in axes),
        "accepted_texts": sum(a["accepted_texts"] for a in axes),
        "false_accepted_texts": sum(a["false_accepted_texts"] for a in axes),
        "exact_accepted": sum(a["exact_accepted"] for a in axes),
        "raw_exact": sum(a["raw_exact"] for a in axes),
        "unresolved": sum(a["unresolved"] for a in axes),
        "regions_within_tolerance": sum(a["region"].get("top3_recall", False) for a in axes),
        "axis_attempts": len(axes),
    }
    return summary


def gate_report(summary):
    # No acceptance classifier was established; zero accepted cases cannot pass.
    policy = policy_snapshot()
    return {
        "automatic_roi_enabled": False,
        "automatic_count_axes": {"rows": False, "columns": False},
        "complete_layout_gate_passed": False,
        "independent_count_gates_passed": {"rows": False, "columns": False},
        "accepted_complete_layouts": summary["automatic_layout_acceptances"],
        "accepted_layout_families": 0,
        "required_complete_layouts": policy["required_complete_layouts"],
        "required_layout_families": policy["required_layout_families"],
        "accepted_exact_count_examples": {"rows": 0, "columns": 0},
        "required_exact_count_examples_per_axis": policy["required_exact_count_examples_per_axis"],
        "decision": "ship_review_only",
        "reason": (
            "No validated complete-layout acceptance policy or independent exact-count "
            "acceptance gates; zero coverage cannot establish automation"
        ),
    }


def evaluate(include_ocr=False):
    manifest, digest = load_frozen()
    records = []
    for entry in manifest["entries"]:
        records.append(evaluate_entry(entry, include_ocr))
        print(f"Evaluated {entry['id']}", flush=True)
    summary = summarise(records)
    return {
        "schema_version": 1,
        "corpus_version": manifest["corpus_version"],
        "manifest_sha256": digest,
        "software_version": __version__,
        "detector_version": DETECTOR_VERSION,
        "thresholds": threshold_snapshot(),
        "automation_policy": policy_snapshot(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": pillow_version,
        },
        "evaluation_mode": (
            "whole-image ROI; conditional counts/OCR; oracle-reviewed numerical comparison"
        ),
        "published_real_figure_count": manifest["published_real_figure_count"],
        "practical_coverage_limitation": manifest["real_figure_limitation"],
        "summary": summary,
        "automation_gates": gate_report(summary),
        "attempts": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.ocr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2))
