"""Evaluate deployed M14 runtime first, then score against independent plotted truth."""

import argparse
import hashlib
import json
import platform
import runpy
import time
from collections import Counter
from pathlib import Path

import numpy as np

from heatmap_decoder import AutomationRequest, DetectionHints, ExtractionConfig, automate, extract
from heatmap_decoder.detection import DETECTOR_VERSION
from heatmap_decoder.detection_policy import policy_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "benchmarks/m14/fixtures.py"))


def metrics(result, truth):
    target = np.array(truth["values"])
    finite = np.isfinite(result.values)
    errors = np.abs(result.values[finite] - target[finite]) / 8
    all_errors = np.abs(result.inferred_values - target) / 8
    return {
        "mae": float(errors.mean()) if errors.size else None,
        "maximum": float(errors.max()) if errors.size else None,
        "nearest_maximum": float(np.nanmax(all_errors)),
        "finite_fraction": float(finite.mean()),
        "flagged_fraction": float((result.flags != 0).mean()),
        "poor_match_fraction": float(((result.flags & 2) != 0).mean()),
        "ambiguous_fraction": float(((result.flags & 4) != 0).mean()),
        "missing_fraction": float(((result.flags & 1) != 0).mean()),
        "maximum_delta_e76": float(np.nanmax(result.colour_distance)),
    }


def attempt(case):
    image, truth = FIXTURE["fixture"](case)
    config = ExtractionConfig.model_validate(truth["config"])
    manual = extract(image, config)
    manual_metrics = metrics(manual, truth)
    hints = DetectionHints(
        rows=config.rows,
        columns=config.columns,
        colourbar_start_value=config.colourbar_start_value,
        colourbar_end_value=config.colourbar_end_value,
        requested=("heatmap", "colourbar"),
    )
    start = time.monotonic()
    result = automate(
        image, AutomationRequest(hints=hints, numeric_only=True), experimental_export=True
    )
    decisions = {d.requirement: d for d in result.decisions}
    candidates = {c.id: c for c in result.detection.candidates}
    geometry = {}
    for kind in ("heatmap", "colourbar"):
        decision = decisions[kind]
        box = candidates[decision.candidate_ids[0]].bbox if decision.status == "accepted" else None
        expected = truth["config"][kind + "_bbox"]
        tolerance = (
            [
                ((expected[2] - expected[0]) / config.columns) * 0.1,
                ((expected[3] - expected[1]) / config.rows) * 0.1,
            ]
            * 2
            if kind == "heatmap"
            else [1] * 4
        )
        geometry[kind] = {
            "accepted": decision.status == "accepted",
            "bbox": box,
            "correct": box is not None
            and bool(np.all(np.abs(np.array(box) - expected) <= tolerance)),
            "reasons": decision.reasons,
        }
    recovered = extract(image, result.diagnostic_config) if result.diagnostic_config else None
    numeric = metrics(recovered, truth) if recovered else None
    complete = result.complete.status == "accepted"
    correct = bool(
        all(g["correct"] for g in geometry.values())
        and numeric
        and numeric["mae"] is not None
        and numeric["mae"] < 0.02
        and numeric["maximum"] <= 0.05
        and numeric["finite_fraction"] == 1
        and numeric["flagged_fraction"] <= 0.01
    )
    components = getattr(manual, "colour_evidence", None)
    return {
        **case,
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "truth": truth,
        "manual": manual_metrics,
        "geometry": geometry,
        "association": decisions["association"].model_dump(mode="json"),
        "numeric_quality": decisions["numeric_quality"].model_dump(mode="json"),
        "complete": complete,
        "correct_complete": correct,
        "export_allowed": result.automatic_export_allowed,
        "numeric": numeric,
        "seconds": round(time.monotonic() - start, 3),
        "candidate_boxes": [
            {"id": c.id, "kind": c.kind, "bbox": c.bbox, "orientation": c.orientation}
            for c in result.detection.candidates
        ],
        "manual_evidence_summary": dict(Counter(e.kind for e in components if e))
        if components
        else None,
    }


def evaluate(split):
    records = []
    for case in [*FIXTURE["cases"](split), *FIXTURE["challenges"](split)]:
        record = attempt(case)
        records.append(record)
        print(
            case["id"],
            "complete",
            record["complete"],
            "correct",
            record["correct_complete"],
            "manual",
            record["manual"],
            flush=True,
        )
    summary = {}
    for palette in FIXTURE["PALETTES"]:
        rows = [r for r in records if r["palette"] == palette and r["negative"] is None]
        summary[palette] = {
            "attempts": len(rows),
            "heatmap": sum(r["geometry"]["heatmap"]["accepted"] for r in rows),
            "bar": sum(r["geometry"]["colourbar"]["accepted"] for r in rows),
            "association": sum(r["association"]["status"] == "accepted" for r in rows),
            "numeric_qc": sum(r["numeric_quality"]["status"] == "accepted" for r in rows),
            "complete": sum(r["complete"] for r in rows),
            "wrong_complete": sum(r["complete"] and not r["correct_complete"] for r in rows),
            "clean_manual": [r["manual"] for r in rows if r["variant"] == "clean"],
        }
    sources = [
        "src/heatmap_decoder/" + p
        for p in (
            "calibration.py",
            "sampling.py",
            "extract.py",
            "detection.py",
            "detection_policy.py",
            "acceptance.py",
            "automation.py",
            "bar_cores.py",
            "lattice.py",
        )
    ]
    import matplotlib

    return {
        "scope": "manual numerical truth plus selective supplied-count/scale automation",
        "split": split,
        "policy": policy_snapshot(),
        "detector": DETECTOR_VERSION,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "summary": summary,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Refusing to overwrite retained evidence")
    args.out.write_text(json.dumps(evaluate(args.split), indent=2) + "\n")
