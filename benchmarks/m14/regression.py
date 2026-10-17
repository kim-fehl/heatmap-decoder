"""Retest inspected corrections mode 1; retained validation is now development evidence."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = runpy.run_path(str(ROOT / "benchmarks/corrections/evaluate.py"))
PREREG = EVALUATOR["PREREG"]
FIXTURE = EVALUATOR["FIXTURE"]
historical = json.loads((ROOT / "benchmarks/corrections/validation.json").read_text())
old = {r["id"]: r for r in historical["attempts"] if r["mode"] == 1}


def compact(record):
    return {
        k: record[k]
        for k in (
            "id",
            "base_id",
            "mode",
            "complete",
            "correct_complete",
            "negative_applies",
            "experimental_export_allowed",
            "geometry",
            "numeric",
            "decisions",
        )
    }


def evaluate():
    records = []
    for case in FIXTURE["cases"]("validation"):
        image, truth = FIXTURE["make_fixture"](case)
        record, _ = EVALUATOR["attempt"](case, image, truth, 1)
        record = compact(record)
        record["before_complete"] = old[case["id"]]["complete"]
        records.append(record)
        print(case["id"], record["before_complete"], "->", record["complete"], flush=True)
    for case, image, truth in EVALUATOR["negatives"]("validation"):
        if 1 not in PREREG["negative_models"].get(case["negative"], []):
            continue
        record, _ = EVALUATOR["attempt"](case, image, truth, 1)
        record = compact(record)
        record["negative"] = case["negative"]
        records.append(record)
        print(case["id"], record["complete"], flush=True)
    ordinary = [r for r in records if not r.get("negative")]
    negatives = [r for r in records if r.get("negative")]
    return {
        "scope": "inspected correction-validation supplied-count/scale development regression",
        "summary": {
            "ordinary": len(ordinary),
            "before_complete": sum(r["before_complete"] for r in ordinary),
            "complete": sum(r["complete"] for r in ordinary),
            "newly_complete": sum(r["complete"] and not r["before_complete"] for r in ordinary),
            "lost_complete": sum(not r["complete"] and r["before_complete"] for r in ordinary),
            "wrong_complete": sum(r["complete"] and not r["correct_complete"] for r in ordinary),
            "negative_attempts": len(negatives),
            "negative_complete": sum(r["complete"] for r in negatives),
            "negative_exports": sum(r["experimental_export_allowed"] for r in negatives),
        },
        "records": records,
    }


if __name__ == "__main__":
    output = ROOT / "benchmarks/m14/corrections-regression.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite retained evidence")
    report = evaluate()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["summary"])
