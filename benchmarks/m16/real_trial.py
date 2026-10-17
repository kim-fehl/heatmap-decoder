"""Run all five M16 crop workflows; reference table enters evaluation only."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from heatmap_decoder.automation import AutomationRequest, automate
from heatmap_decoder.io import (
    detect_file,
    extract_file,
    extract_labels_file,
    inspect_legend_file,
    load_raster,
)
from heatmap_decoder.models import DetectionHints, ExtractionConfig, GridConfig, LegendConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    if out.exists():
        raise ValueError("refuse to overwrite trial evidence")
    out.mkdir(parents=True)
    manifest = json.loads(Path("examples/m16/manifest.json").read_text())
    path = Path(manifest["image"])
    image = load_raster(path)
    config = ExtractionConfig.model_validate_json(Path("examples/m16/config.json").read_bytes())
    records = {}
    t = time.monotonic()
    manual = extract_file(path, config, out / "manual", config_path="examples/m16/config.json")
    records["supplied_geometry_reviewed_calibration"] = {
        "seconds": time.monotonic() - t,
        "finite": int(np.isfinite(manual.values).sum()),
        "cells": manual.values.size,
        "flag_counts": {str(f): int(np.sum(manual.flags == f)) for f in np.unique(manual.flags)},
        "bound_cells": int(np.sum(manual.range_limit != 0)),
        "complete": bool(np.isfinite(manual.values).all() and not np.any(manual.flags)),
    }
    hints = DetectionHints(rows=33, columns=8, requested=("heatmap", "colourbar"))
    t = time.monotonic()
    detected = detect_file(path, hints, out / "detection")
    auto = automate(
        image,
        AutomationRequest(hints=hints, numeric_only=True, scale=config.scale),
        experimental_export=True,
    )
    records["automatic_roi_supplied_dimensions"] = {
        "seconds": time.monotonic() - t,
        "result": auto.model_dump(mode="json"),
        "detection": detected.result.model_dump(mode="json"),
    }
    t = time.monotonic()
    legend = inspect_legend_file(
        path,
        LegendConfig(
            colourbar_bbox=config.colourbar_bbox,
            colourbar_orientation="horizontal",
            scale_model="piecewise_linear",
        ),
        out / "legend",
    )
    ticks = manifest["tick_anchors"]
    recognised = []
    for anchor in ticks:
        nearby = [tick for tick in legend.ticks if abs(tick.position - anchor["position"]) <= 2]
        correct = [tick for tick in nearby if tick.parsed_value == anchor["value"]]
        recognised.append(
            {
                "position": anchor["position"],
                "value": anchor["value"],
                "localised": bool(nearby),
                "exact": bool(correct),
                "sign_error": any(
                    tick.parsed_value == -anchor["value"] for tick in nearby if anchor["value"] != 0
                ),
            }
        )
    records["legend_localisation_numeric_ocr"] = {
        "seconds": time.monotonic() - t,
        "reference_ticks": recognised,
        "localised": sum(t["localised"] for t in recognised),
        "exact": sum(t["exact"] for t in recognised),
        "denominator": len(ticks),
        "result": legend.model_dump(mode="json"),
    }
    for name in ["labels", "labels-retries"]:
        t = time.monotonic()
        label_config = GridConfig.model_validate_json(
            Path(f"examples/m16/{name}.json").read_bytes()
        )
        labels = extract_labels_file(path, label_config, out / name)
        raw_exact = accepted = accepted_exact = slots_correct = 0
        for axis in labels.axes:
            truth = manifest["displayed_rows" if axis.axis == "rows" else "displayed_columns"]
            for slot, text in zip(axis.slots, truth, strict=True):
                raw_exact += slot.raw_text == text
                accepted += slot.status == "accepted"
                accepted_exact += slot.status == "accepted" and slot.final_text == text
                slots_correct += not any(i in slot.issues for i in ["spans_slots", "axis_offset"])
        records[name] = {
            "seconds": time.monotonic() - t,
            "slots": 41,
            "raw_exact": raw_exact,
            "heuristic_accepted": accepted,
            "accepted_exact": accepted_exact,
            "false_heuristic_acceptance": accepted - accepted_exact,
            "slots_without_spatial_issue": slots_correct,
            "unresolved": labels.unresolved_count,
            "calls": len(labels.slot_ocr_runs),
            "skipped": labels.skipped_ocr_attempts,
            "axes": [asdict(a) for a in labels.axes],
        }
    t = time.monotonic()
    composed = automate(
        image,
        AutomationRequest(
            hints=DetectionHints(
                rows=33,
                columns=8,
                requested=("heatmap", "colourbar", "labels_rows", "labels_columns"),
            ),
            numeric_only=False,
            scale=config.scale,
        ),
        experimental_export=True,
    )
    records["composed_requested_workflow"] = {
        "seconds": time.monotonic() - t,
        "result": composed.model_dump(mode="json"),
    }
    # Evaluation-only reference access occurs after every production workflow.
    rows = Path(manifest["evaluation_tsv"]).read_text().splitlines()
    reference = np.array([[float(v) for v in row.split("\t")[1:]] for row in rows[1:]])
    if reference.shape != manual.values.shape:
        raise ValueError("reference dimensions differ; do not change geometry from TSV")
    finite = np.isfinite(manual.values)
    difference = np.abs(manual.values[finite] - reference[finite])
    records["reconstruction_agreement"] = {
        "recipe_established": False,
        "denominator": int(reference.size),
        "finite_compared": int(finite.sum()),
        "normalising_range": 4,
        "range_note": "labelled tick span -2..2, physical edge values unresolved",
        "mae": float(np.mean(difference)),
        "maximum": float(np.max(difference)),
        "mae_fraction_labelled_range": float(np.mean(difference) / 4),
        "maximum_fraction_labelled_range": float(np.max(difference) / 4),
        "reference_values_outside_reviewed_ticks": int(np.sum(np.abs(reference) > 2)),
        "reference_mean_by_row_range": [
            float(reference.mean(1).min()),
            float(reference.mean(1).max()),
        ],
        "reference_sd_ddof0_by_row_range": [
            float(reference.std(1).min()),
            float(reference.std(1).max()),
        ],
    }
    records["manifest_sha256"] = hashlib.sha256(
        Path("examples/m16/manifest.json").read_bytes()
    ).hexdigest()
    (out / "report.json").write_text(
        json.dumps(records, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: {i: v for i, v in r.items() if i not in ["axes", "result", "detection"]}
                for k, r in records.items()
                if isinstance(r, dict)
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
