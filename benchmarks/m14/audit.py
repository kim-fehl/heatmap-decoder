"""Individually audit all inspected correction-validation numeric flag failures."""

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np

from heatmap_decoder import ExtractionConfig, extract
from heatmap_decoder.calibration import to_lab
from heatmap_decoder.sampling import sample_colourbar

ROOT = Path(__file__).resolve().parents[2]
RENDER = runpy.run_path(str(ROOT / "examples/corrections/generate.py"))


def intervals(mask):
    indices = np.flatnonzero(mask)
    return [
        (int(g[0]), int(g[-1])) for g in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    ]


def audit():
    historical = json.loads((ROOT / "benchmarks/corrections/validation.json").read_text())
    rows = []
    for record in historical["attempts"]:
        if record["mode"] != 1 or record.get("negative"):
            continue
        reasons = next(
            d["reasons"] for d in record["decisions"] if d["requirement"] == "numeric_quality"
        )
        if not set(reasons) & {
            "excessive_colour_matching_flags",
            "missing_or_ambiguous_numeric_cells",
        }:
            continue
        image, truth = RENDER["make_fixture"](record)
        config = ExtractionConfig(
            heatmap_bbox=record["geometry"]["heatmap"]["bbox"],
            colourbar_bbox=record["geometry"]["colourbar"]["bbox"],
            colourbar_orientation=truth["orientation"],
            rows=truth["rows"],
            columns=truth["columns"],
            colourbar_start_value=truth["start_value"],
            colourbar_end_value=truth["end_value"],
        )
        result = extract(image, config)
        manual = extract(
            image,
            config.model_copy(
                update={
                    "heatmap_bbox": truth["heatmap_bbox"],
                    "colourbar_bbox": truth["colourbar_bbox"],
                }
            ),
        )
        bar = sample_colourbar(image, config)
        target = np.array(truth["expected_matrix"])
        t = (target - truth["start_value"]) / (truth["end_value"] - truth["start_value"])
        kind = truth["family"].split("-")[-1]
        ideal_cells = RENDER["palette"](t, kind).astype(float)
        ideal_bar = RENDER["palette"](np.linspace(0, 1, len(bar)), kind).astype(float)
        colours, bar_lab = to_lab(result.sampled_rgb.reshape(-1, 3)), to_lab(bar)
        failures = []
        for index in np.flatnonzero(result.flags.ravel()):
            distances = np.linalg.norm(bar_lab - colours[index], axis=1)
            matches = distances <= distances.min() + config.ambiguity_colour_tolerance
            parts = intervals(matches)
            row, col = divmod(int(index), config.columns)
            span = (parts[-1][1] - parts[0][0]) / (len(bar) - 1)
            failures.append(
                {
                    "row": row,
                    "column": col,
                    "flags": int(result.flags[row, col]),
                    "best_delta_e76": float(distances.min()),
                    "span_fraction": float(span),
                    "kind": "broad_contiguous" if len(parts) == 1 else "disconnected_near_matches",
                    "components": parts,
                    "numeric_components": [
                        [
                            truth["start_value"]
                            + (truth["end_value"] - truth["start_value"]) * a / (len(bar) - 1),
                            truth["start_value"]
                            + (truth["end_value"] - truth["start_value"]) * b / (len(bar) - 1),
                        ]
                        for a, b in parts
                    ],
                    "sample_vs_plotted_rgb_max": float(
                        np.max(np.abs(result.sampled_rgb[row, col] - ideal_cells[row, col]))
                    ),
                    "nearest_error_fraction": float(
                        abs(result.inferred_values[row, col] - target[row, col])
                        / abs(truth["end_value"] - truth["start_value"])
                    ),
                    "endpoint_match": bool(result.range_limit[row, col]),
                }
            )
        rows.append(
            {
                "id": record["id"],
                "family": truth["family"],
                "degradation": record["degradation"],
                "historical_reasons": reasons,
                "historical_numeric": record["numeric"],
                "heatmap_bounds_error": (
                    np.array(config.heatmap_bbox) - truth["heatmap_bbox"]
                ).tolist(),
                "bar_bounds_error": (
                    np.array(config.colourbar_bbox) - truth["colourbar_bbox"]
                ).tolist(),
                "bar_rgb_mismatch_max": float(np.max(np.abs(bar - ideal_bar))),
                "bar_rgb_mismatch_median": float(
                    np.median(np.max(np.abs(bar - ideal_bar), axis=1))
                ),
                "missing_cells": int(((result.flags & 1) != 0).sum()),
                "poor_match_cells": int(((result.flags & 2) != 0).sum()),
                "ambiguous_cells": int(((result.flags & 4) != 0).sum()),
                "manual_ambiguous_cells": int(((manual.flags & 4) != 0).sum()),
                "cells": failures,
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            }
        )
    return {
        "scope": "inspected correction-validation regression; thresholds unchanged",
        "historical_report_sha256": hashlib.sha256(
            (ROOT / "benchmarks/corrections/validation.json").read_bytes()
        ).hexdigest(),
        "summary": {
            "images": len(rows),
            "excessive_flags": sum(
                "excessive_colour_matching_flags" in r["historical_reasons"] for r in rows
            ),
            "missing_only": sum(
                r["historical_reasons"] == ["missing_or_ambiguous_numeric_cells"] for r in rows
            ),
            "cell_kinds": dict(Counter(c["kind"] for r in rows for c in r["cells"])),
            "ambiguous_cells": sum(r["ambiguous_cells"] for r in rows),
            "poor_match_cells": sum(r["poor_match_cells"] for r in rows),
            "missing_cells": sum(r["missing_cells"] for r in rows),
        },
        "records": rows,
    }


if __name__ == "__main__":
    report = audit()
    (ROOT / "benchmarks/m14/corrections-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report["summary"])
