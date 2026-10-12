"""Compute release recommendations from frozen test choices; cannot enable runtime flags."""

import argparse
import hashlib
import json
from pathlib import Path

from heatmap_decoder.automation import AUTOMATION_POLICY_VERSION

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; frozen release evidence cannot be overwritten")
    data = args.test.read_bytes()
    report = json.loads(data)
    if report["split"] != "test":
        parser.error("release evaluation requires the untouched test partition")
    gates = json.loads((ROOT / "benchmarks/m12/preregister.json").read_text())["gates"]
    modes = {}
    for mode, s in report["summary"].items():
        checks = dict(
            distinct_bases=s["distinct_accepted_bases"] >= gates["distinct_accepted_bases"],
            layout_families=len(s["accepted_families"]) >= gates["accepted_families"],
            ordinary_coverage=s["coverage"] >= gates["complete_coverage"],
            correct_complete=s["wrong_accepted"] == gates["wrong_complete_acceptances"],
            challenge_abstention=s["challenge_accepted"] == gates["challenge_acceptances"],
        )
        if mode in ("2", "4"):
            for axis, g in s["count_gates"].items():
                checks[f"exact_inferred_{axis}"] = (
                    g["exact_accepted_bases"] >= gates["exact_inferred_counts_per_axis"]
                    and len(g["families"]) >= gates["accepted_families"]
                )
        modes[mode] = dict(
            checks=checks,
            failed=[k for k, v in checks.items() if not v],
            measured_release_eligible=all(checks.values()),
            recommendation="release_candidate"
            if all(checks.values())
            else "experimental_review_only",
            observed=s,
        )
    args.out.write_text(
        json.dumps(
            dict(
                test_sha256=hashlib.sha256(data).hexdigest(),
                selective_policy_version=AUTOMATION_POLICY_VERSION,
                policy=report["policy"],
                gate_source="preregister.json; complete correctness includes QC",
                scope="Synthetic base figures only; shared renderer/layouts limit independence",
                runtime_enablement="Recommendations cannot enable runtime export",
                modes=modes,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
