"""Combine complete, disjoint frozen validation shards without re-deciding images."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

EVALUATOR = runpy.run_path(str(Path(__file__).with_name("evaluate.py")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; evidence must not be overwritten")
    reports = [json.loads(p.read_text()) for p in args.reports]
    total = reports[0]["shard"]["total"]
    assert len(reports) == total
    assert {r["shard"]["index"] for r in reports} == set(range(total))
    fixed = ("purpose", "split", "python", "detector", "pipeline", "policy", "prereg_sha256")
    for r in reports:
        assert r["shard"]["total"] == total
        assert all(r[k] == reports[0][k] for k in fixed)
    attempts = sorted(
        (a for r in reports for a in r["attempts"]), key=lambda a: (a["id"], a["mode"])
    )
    identities = {(a["id"], a["mode"]) for a in attempts}
    assert len(identities) == len(attempts)
    bases = EVALUATOR["PREREG"]["base_splits"][reports[0]["split"]]["bases"]
    assert sum(not a.get("negative") for a in attempts) == bases * 3 * 4
    assert (
        sum(bool(a.get("negative")) for a in attempts)
        == len(EVALUATOR["PREREG"]["negative_models"]) * 4
    )
    conditional = sorted((c for r in reports for c in r["conditional_ocr"]), key=lambda c: c["id"])
    assert len({c["id"] for c in conditional}) == len(conditional)
    if conditional:
        assert len(conditional) == bases * 3
    result = {k: reports[0][k] for k in fixed}
    result.update(
        shard=dict(index=None, total=total, combined=True),
        shard_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reports],
        summary=EVALUATOR["summary"](attempts),
        attempts=attempts,
        conditional_ocr=conditional,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
