"""Additional truth scoring of FROZEN decisions; never reruns or selects a candidate."""

import argparse
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/m12/generate.py"))


def analyse(path):
    report = json.loads(path.read_text())
    summary = {}
    for mode in (1, 2, 3, 4):
        ordinary = [a for a in report["attempts"] if a["mode"] == mode and not a.get("challenge")]
        accepted = [a for a in ordinary if a["experimental_accepted"]]
        numeric = [a["numeric"] for a in accepted if a["numeric"] is not None]
        expected_ticks = sum(len(FIXTURE["make_fixture"](a)[1]["ticks"]) for a in ordinary)
        ticks = [t for a in ordinary for t in a["ticks"]]
        exact = sum(t["exact"] for t in ticks)
        # Includes partial-capability scale successes even when counts/ROI abstain.
        scale_passed = [
            a
            for a in ordinary
            if next(d for d in a["decisions"] if d["requirement"] == "scale")["status"]
            == "accepted"
        ]
        complete_tick_cases = [
            a
            for a in scale_passed
            if a["ticks"]
            and len(a["ticks"]) == len(FIXTURE["make_fixture"](a)[1]["ticks"])
            and all(t["exact"] for t in a["ticks"])
        ]
        summary[str(mode)] = dict(
            expected_ticks=expected_ticks if mode in (3, 4) else None,
            observed_tick_records=len(ticks),
            correctly_associated=sum(t["associated"] for t in ticks),
            exact_tick_values=exact,
            exact_fraction_of_expected=exact / expected_ticks if mode in (3, 4) else None,
            sign_errors=sum(t["sign_error"] for t in ticks),
            scale_capability_accepted=len(scale_passed),
            OCR_scale_accepted_exact_tick_cases=len(complete_tick_cases),
            accepted_numeric_attempts=len(numeric),
            maximum_MAE_fraction_of_range=max((n["mae"] for n in numeric), default=None),
            maximum_cell_error_fraction_of_range=max((n["maximum"] for n in numeric), default=None),
            minimum_finite_fraction=min((n["finite"] for n in numeric), default=None),
            maximum_flagged_fraction=max((n["flagged"] for n in numeric), default=None),
            maximum_value_residual=max(
                (
                    max(abs(v) for v in a["legend_fit"]["value_residuals"])
                    for a in complete_tick_cases
                ),
                default=None,
            ),
            maximum_position_residual_pixels=max(
                (
                    max(abs(v) for v in a["legend_fit"]["position_residuals_pixels"])
                    for a in complete_tick_cases
                ),
                default=None,
            ),
            minimum_recognised_span=min(
                (np.ptp(a["legend_fit"]["recognised_span"]) for a in complete_tick_cases),
                default=None,
            ),
        )
    return dict(
        split=report["split"],
        report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note="Supplemental metrics only; original deployed choices and release gates unchanged",
        summary=summary,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists; frozen evidence cannot be overwritten")
    args.out.write_text(json.dumps(analyse(args.report), indent=2) + "\n")
