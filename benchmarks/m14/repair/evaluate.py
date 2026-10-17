"""Freeze grid repairs before a new synthetic seed partition; preserve old evidence."""

import argparse
import hashlib
import json
import runpy
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = runpy.run_path(str(ROOT / "benchmarks/m14/evaluate.py"))
FIXTURE = BASE["FIXTURE"]


def manifest():
    cases = []
    for case in [*FIXTURE["cases"]("validation"), *FIXTURE["challenges"]("validation")]:
        case = dict(case)
        case["id"] = case["id"].replace("validation", "repair-check")
        case["base_id"] = case["base_id"].replace("validation", "repair-check")
        case["split"] = "repair-check"
        case["seed"] = 714000 + case["layout"] if case["negative"] is None else 814000
        cases.append(case)
    return cases


def sources():
    files = [
        *sorted((ROOT / "src/heatmap_decoder").glob("*.py")),
        ROOT / "benchmarks/m14/fixtures.py",
        ROOT / "benchmarks/m14/evaluate.py",
        Path(__file__),
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def evaluate(partition):
    frozen = json.loads((Path(__file__).parent / "freeze.json").read_text())
    assert sources() == frozen["source_sha256"], "Runtime or evaluator changed after freeze"
    if partition == "new-seeds":
        cases = frozen["new_cases"]
    else:
        cases = [*FIXTURE["cases"](partition), *FIXTURE["challenges"](partition)]
    rows = [BASE["attempt"](c) for c in cases]
    ordinary = [r for r in rows if r["negative"] is None]
    negatives = [r for r in rows if r["negative"] not in (None, "dark_band", "black_border")]
    summary = {
        "attempts": len(ordinary),
        "complete": sum(r["complete"] for r in ordinary),
        "wrong_complete": sum(r["complete"] and not r["correct_complete"] for r in ordinary),
        "negative_exports": sum(r["export_allowed"] for r in negatives),
        "negative_attempts": len(negatives),
        "by_palette": {
            p: sum(r["complete"] for r in ordinary if r["palette"] == p)
            for p in FIXTURE["PALETTES"]
        },
        "by_variant": {
            v: sum(r["complete"] for r in ordinary if r["variant"] == v)
            for v in ("clean", "jpeg", "blur")
        },
        "heatmap_accepted": sum(r["geometry"]["heatmap"]["accepted"] for r in ordinary),
        "bar_accepted": sum(r["geometry"]["colourbar"]["accepted"] for r in ordinary),
        "heatmap_reasons": dict(
            Counter(
                reason
                for r in ordinary
                for reason in r["geometry"]["heatmap"]["reasons"]
                if reason != "development_evidence_passed"
            )
        ),
    }
    if partition != "new-seeds":
        report_name = "development" if partition == "development" else "validation"
        old = {
            r["id"]: r
            for r in json.loads((ROOT / f"benchmarks/m14/{report_name}.json").read_text())[
                "records"
            ]
        }
        assert all(r["image_sha256"] == old[r["id"]]["image_sha256"] for r in rows)
        summary["newly_complete"] = sum(
            r["complete"] and not old[r["id"]]["complete"] for r in ordinary
        )
        summary["lost_complete"] = sum(
            not r["complete"] and old[r["id"]]["complete"] for r in ordinary
        )
    return {
        "scope": "Grid repair development regression"
        if partition != "new-seeds"
        else "Frozen new seeds, same synthetic generator/layouts; no publication claims",
        "partition": partition,
        "source_sha256": sources(),
        "summary": summary,
        "records": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("partition", choices=("freeze", "development", "validation", "new-seeds"))
    args = parser.parse_args()
    path = Path(__file__).parent / (
        "freeze.json" if args.partition == "freeze" else args.partition + ".json"
    )
    if path.exists():
        raise SystemExit("Refusing to overwrite retained evidence")
    if args.partition == "freeze":
        result = {
            "scope": "Policy freeze before new-seed evaluation",
            "source_sha256": sources(),
            "new_cases": manifest(),
        }
    else:
        result = evaluate(args.partition)
        print(json.dumps(result["summary"], indent=2))
    path.write_text(json.dumps(result, indent=2) + "\n")
