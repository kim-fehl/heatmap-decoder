"""Post-validation diagnosis; does not change policy or retained validation evidence."""

import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

import numpy as np

from heatmap_decoder import AutomationRequest, DetectionHints, automate
from heatmap_decoder.acceptance import grid_decision
from heatmap_decoder.detection import foreground
from heatmap_decoder.detection_policy import ACCEPTANCE_THRESHOLDS as A

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "benchmarks/m14/fixtures.py"))


def investigate():
    source = ROOT / "benchmarks/m14/validation.json"
    records = json.loads(source.read_text())["records"]
    output = []
    for old in records:
        if old["negative"] is not None:
            continue
        image, truth = FIXTURE["fixture"](old)
        assert hashlib.sha256(image.tobytes()).hexdigest() == old["image_sha256"]
        config = truth["config"]
        hints = DetectionHints(
            rows=config["rows"],
            columns=config["columns"],
            colourbar_start_value=config["colourbar_start_value"],
            colourbar_end_value=config["colourbar_end_value"],
            requested=("heatmap", "colourbar"),
        )
        result = automate(image, AutomationRequest(hints=hints, numeric_only=True))
        decisions = {d.requirement: d.model_dump(mode="json") for d in result.decisions}
        assert (result.complete.status == "accepted") == old["complete"]
        candidate_errors = {}
        for kind in ("heatmap", "colourbar"):
            candidate_errors[kind] = [
                {
                    "bbox": c.bbox,
                    "edge_error_pixels": (np.array(c.bbox) - config[kind + "_bbox"]).tolist(),
                }
                for c in result.detection.candidates
                if c.kind == kind
            ]
        first = next(
            (
                k
                for k in ("heatmap", "colourbar", "association", "numeric_quality")
                if decisions[k]["status"] != "accepted"
            ),
            "accepted",
        )
        counterfactual = None
        if old["layout"] == 1 and old["variant"] == "clean":
            counterfactual = grid_decision(
                np.asarray(image),
                hints,
                "diagnostic-truth",
                tuple(config["heatmap_bbox"]),
                tuple(
                    foreground(np.asarray(image), hints.detector.background_distance * factor)
                    for factor in A.perturbation_threshold_factors
                ),
            ).model_dump(mode="json")
        m = old["manual"]
        output.append(
            {
                "id": old["id"],
                "palette": old["palette"],
                "layout": old["layout"],
                "variant": old["variant"],
                "first_blocker": first,
                "manual_full_qc": m["finite_fraction"] == 1 and m["flagged_fraction"] <= 0.01,
                "manual_accuracy_pass": m["maximum"] <= 0.05 and m["mae"] < 0.02,
                "manual": m,
                "candidate_errors": candidate_errors,
                "truth_bounds_grid_counterfactual": counterfactual,
                "decisions": {k: decisions[k] for k in ("heatmap", "colourbar", "numeric_quality")},
            }
        )
    summary = {
        "first_blocker": dict(Counter(r["first_blocker"] for r in output)),
        "manual_full_qc": sum(r["manual_full_qc"] for r in output),
        "rejected_but_manual_full_qc": sum(
            r["first_blocker"] != "accepted" and r["manual_full_qc"] for r in output
        ),
        "by_variant": {
            v: dict(Counter(r["first_blocker"] for r in output if r["variant"] == v))
            for v in ("clean", "jpeg", "blur")
        },
        "heatmap_reasons_overlapping": dict(
            Counter(
                reason
                for r in output
                for reason in r["decisions"]["heatmap"]["reasons"]
                if reason != "development_evidence_passed"
            )
        ),
    }
    return {
        "scope": "Post-validation diagnostic data; no policy changes or new held-out claims",
        "validation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/heatmap_decoder").glob("*.py"))
        },
        "summary": summary,
        "records": output,
    }


if __name__ == "__main__":
    path = ROOT / "benchmarks/m14/investigation.json"
    if path.exists():
        raise SystemExit("Refusing to overwrite retained evidence")
    report = investigate()
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
