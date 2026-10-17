"""Inspected M12 is regression evidence, never a fresh validation split."""

import argparse
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT = runpy.run_path(str(ROOT / "benchmarks/corrections/evaluate.py"))
OLD = runpy.run_path(str(ROOT / "benchmarks/m12/evaluate.py"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    path = parser.parse_args().out
    if path.exists():
        parser.error("report exists; regression evidence must not be overwritten")
    attempts = []
    for case in OLD["FIXTURE"]["cases"]("test"):
        image, truth = OLD["FIXTURE"]["make_fixture"](case)
        record, _ = CURRENT["attempt"](case, image, truth, 1)
        attempts.append(record)
        print(case["id"], record["complete"], record["correct_complete"], flush=True)
    for case, image, truth in OLD["challenges"]("test"):
        case["negative"] = case["challenge"]
        record, _ = CURRENT["attempt"](case, image, truth, 1)
        record["negative_applies"] = 1 in OLD["SCOPE"][case["challenge"]]
        attempts.append(record)
        print(case["id"], record["complete"], flush=True)
    ordinary = [a for a in attempts if not a.get("negative")]
    accepted = [a for a in ordinary if a["complete"]]
    negatives = [a for a in attempts if a["negative_applies"]]
    report = dict(
        purpose=(
            "Inspected historical M12, mode 1 only; development regression, not fresh validation"
        ),
        summary=dict(
            ordinary=len(ordinary),
            accepted=len(accepted),
            coverage=len(accepted) / len(ordinary),
            wrong_complete=sum(not a["correct_complete"] for a in accepted),
            negative_attempts=len(negatives),
            negative_complete=sum(a["complete"] for a in negatives),
            negative_exports=sum(a["experimental_export_allowed"] for a in negatives),
        ),
        attempts=attempts,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
