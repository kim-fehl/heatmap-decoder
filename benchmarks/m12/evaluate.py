"""M12 selective pipeline evaluation. Truth enters only AFTER runtime decisions."""

import argparse
import hashlib
import json
import platform
import runpy
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from heatmap_decoder import DetectionHints, extract
from heatmap_decoder.automation import AutomationRequest, automate
from heatmap_decoder.detection_policy import policy_snapshot
from heatmap_decoder.legend import legend_policy_snapshot
from heatmap_decoder.models import ScaleConfig

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/m12/generate.py"))
PREREG = json.loads((ROOT / "benchmarks/m12/preregister.json").read_text())
SCOPE = json.loads((ROOT / "benchmarks/m12/challenge_scope.json").read_text())


def challenges(split):
    for index, name in enumerate(PREREG["challenges"]):
        case = dict(
            id=f"{split}-challenge-{name}",
            base_id=f"{split}-challenge-{name}",
            split=split,
            seed={"development": 123000, "validation": 124000, "test": 125000}[split] + index,
            family="vertical_flank",
            index=index * 3,
            degradation="clean",
            challenge=name,
        )
        image, truth = FIXTURE["make_fixture"](case)
        image = image.copy()
        x, y, r, s = truth["colourbar_bbox"]
        hx, hy, hr, hs = truth["heatmap_bbox"]
        if name == "absent_bar":
            image[y:s, x:r] = 255
        elif name == "categorical":
            image[y : (y + s) // 2, x:r] = [255, 0, 0]
            image[(y + s) // 2 : s, x:r] = [0, 0, 255]
        elif name == "hidden_boundaries":
            image[hy:hs, hx:hr] = [200, 40, 25]
        elif name == "multipanel":
            image = np.concatenate([image, image], axis=1)
        elif name == "competing_bars":
            image[y:s, 150 : 150 + r - x] = image[y:s, x:r]
        elif name in ("missing_tick", "two_ticks", "nonlinear", "touching_text"):
            # Erase the independently rendered tick's text and measured stroke.
            ticks = truth["ticks"]
            targets = ticks[1:-1] if name in ("missing_tick", "two_ticks") else ticks[1:2]
            figure = Image.fromarray(image)
            draw = ImageDraw.Draw(figure)
            from PIL import ImageFont

            font = ImageFont.truetype("DejaVuSans.ttf", 20)
            for tick in targets:
                a, b, c, d = tick["bbox"]
                draw.rectangle((a - 1, b - 1, c + 1, d + 1), fill="white")
                if name in ("missing_tick", "two_ticks"):
                    p = tick["position"]
                    draw.rectangle((x - 10, p - 1, x - 1, p + 1), fill="white")
                    draw.rectangle((r + 1, p - 1, r + 10, p + 1), fill="white")
                else:
                    text = "999" if name == "nonlinear" else "0 units"
                    draw.text((a, b - 4), text, font=font, fill="black")
            image = np.asarray(figure)
        yield case, image, truth


def attempt(case, image, truth, mode):
    supplied = mode in (1, 3)
    ticks = mode in (3, 4)
    # Only mode-authorised inputs are constructed here; no ROI truth is passed.
    hints = dict(
        requested=["heatmap", "colourbar"],
        rows=truth["rows"] + (case.get("challenge") == "wrong_counts") if supplied else None,
        columns=truth["columns"] if supplied else None,
        infer_rows=not supplied,
        infer_columns=not supplied,
    )
    if not ticks:
        hints.update(
            colourbar_start_value=truth["start_value"], colourbar_end_value=truth["end_value"]
        )
    request = AutomationRequest(
        hints=DetectionHints(**hints),
        numeric_only=True,
        scale=ScaleConfig(source="ocr_ticks") if ticks else None,
    )
    started = time.perf_counter()
    result = automate(image, request)
    duration = time.perf_counter() - started
    # Every choice is fixed by deployed policy. The scorer never chooses a proposal.
    accepted = result.complete.status == "accepted"
    geometry_correct = True
    edge_metrics = {}
    candidates = {c.id: c for c in result.detection.candidates}
    decisions = {d.requirement: d for d in result.decisions}
    for kind in ("heatmap", "colourbar"):
        check = decisions[kind]
        if check.status != "accepted":
            continue
        chosen = candidates[check.candidate_ids[0]]
        expected = np.asarray(truth[f"{kind}_bbox"])
        errors = np.abs(np.asarray(chosen.bbox) - expected)
        tolerance = np.ones(4)
        if kind == "heatmap":
            tolerance = 0.1 * np.array(
                [
                    (expected[2] - expected[0]) / truth["columns"],
                    (expected[3] - expected[1]) / truth["rows"],
                ]
                * 2
            )
        correct = bool(np.all(errors <= tolerance))
        geometry_correct &= correct
        edge_metrics[kind] = dict(errors_pixels=errors.tolist(), correct=correct)
    counts = {}
    for axis in ("rows", "columns"):
        check = decisions[axis]
        count = getattr(request.hints, axis)
        if check.status == "accepted" and count is None and len(check.candidate_ids) == 1:
            proposal = next(
                p for p in result.detection.count_proposals if p.id == check.candidate_ids[0]
            )
            count = proposal.count
        counts[axis] = dict(
            status=check.status,
            value=count,
            exact=check.status == "accepted" and count == truth[axis],
        )
    observations = []
    if result.legend is not None:
        for observed in result.legend.ticks:
            if observed.status == "excluded":
                continue
            nearest = min(truth["ticks"], key=lambda t: abs(t["position"] - observed.position))
            association = (
                observed.position_source == "stroke"
                and abs(nearest["position"] - observed.position) <= 1
            )
            observations.append(
                dict(
                    text=observed.raw_text,
                    status=observed.status,
                    value=observed.parsed_value,
                    associated=association,
                    exact=association and observed.parsed_value == nearest["value"],
                    sign_error=bool(
                        association
                        and observed.parsed_value is not None
                        and np.sign(observed.parsed_value) != np.sign(nearest["value"])
                    ),
                )
            )
    tick_correct = not ticks or (
        len(observations) == len(truth["ticks"]) and all(o["exact"] for o in observations)
    )
    numeric = None
    if result.diagnostic_config is not None:
        reconstructed = extract(
            image, result.diagnostic_config, ocr_anchors=result.ocr_anchors or None
        )
        expected = np.asarray(truth["expected_matrix"])
        if reconstructed.values.shape == expected.shape:
            finite = np.isfinite(reconstructed.values)
            error = np.abs(reconstructed.values[finite] - expected[finite]) / abs(
                truth["end_value"] - truth["start_value"]
            )
            numeric = dict(
                mae=float(error.mean()) if error.size else None,
                maximum=float(error.max()) if error.size else None,
                finite=float(finite.mean()),
                flagged=float(np.mean(reconstructed.flags != 0)),
            )
    numeric_correct = (
        numeric is not None
        and numeric["mae"] is not None
        and (
            numeric["mae"] < 0.02
            and numeric["maximum"] <= 0.05
            and numeric["finite"] == 1
            and numeric["flagged"] <= 0.01
        )
    )
    challenge_applies = mode in SCOPE.get(case.get("challenge"), [])
    correct = geometry_correct and tick_correct and numeric_correct and not challenge_applies
    return dict(
        **case,
        mode=mode,
        experimental_accepted=accepted,
        automatic_export_allowed=result.automatic_export_allowed,
        correct_complete=bool(correct),
        challenge_applies=challenge_applies,
        seconds=duration,
        edges=edge_metrics,
        counts=counts,
        ticks=observations,
        numeric=numeric,
        decisions=[d.model_dump(mode="json") for d in result.decisions],
        complete=result.complete.model_dump(mode="json"),
        release_reasons=result.release_reasons,
        legend_fit=result.legend.fit.model_dump(mode="json")
        if result.legend and result.legend.fit
        else None,
    )


def summarise(attempts):
    summary = {}
    for mode in (1, 2, 3, 4):
        ordinary = [a for a in attempts if a["mode"] == mode and not a.get("challenge")]
        accepted = [a for a in ordinary if a["experimental_accepted"]]
        negative = [a for a in attempts if a["mode"] == mode and a["challenge_applies"]]
        bases = {a["base_id"] for a in accepted}
        families = {a["family"] for a in accepted}
        wrong = sum(not a["correct_complete"] for a in accepted)
        negative_accepted = sum(a["experimental_accepted"] for a in negative)
        coverage = len(accepted) / len(ordinary)
        count_gates = {}
        for axis in ("rows", "columns"):
            exact = [a for a in accepted if a["counts"][axis]["exact"]]
            count_gates[axis] = dict(
                exact_accepted_bases=len({a["base_id"] for a in exact}),
                families=sorted({a["family"] for a in exact}),
            )
        gate = (
            len(bases) >= 30
            and len(families) >= 3
            and coverage >= 0.7
            and wrong == 0
            and negative_accepted == 0
        )
        ticks = [t for a in ordinary for t in a["ticks"]]
        summary[str(mode)] = dict(
            ordinary_attempts=len(ordinary),
            accepted=len(accepted),
            distinct_accepted_bases=len(bases),
            accepted_families=sorted(families),
            coverage=coverage,
            wrong_accepted=wrong,
            challenge_attempts=len(negative),
            challenge_accepted=negative_accepted,
            measured_gate_passed=gate
            and (
                mode not in (2, 4)
                or all(
                    g["exact_accepted_bases"] >= 10 and len(g["families"]) >= 3
                    for g in count_gates.values()
                )
            ),
            count_gates=count_gates,
            automatic_exports=sum(
                a["automatic_export_allowed"] for a in attempts if a["mode"] == mode
            ),
            ticks_observed=len(ticks),
            exact_tick_values=sum(t["exact"] for t in ticks),
            tick_sign_errors=sum(t["sign_error"] for t in ticks),
            median_seconds=float(np.median([a["seconds"] for a in ordinary])),
            maximum_seconds=max(a["seconds"] for a in ordinary),
            zero_error_95pct_upper_bound=1 - 0.05 ** (1 / len(bases))
            if bases and not wrong
            else None,
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation", "test"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("output exists: use a fresh path; frozen evidence cannot be overwritten")
    attempts = []
    ordinary = ((case, *FIXTURE["make_fixture"](case)) for case in FIXTURE["cases"](args.split))
    import itertools

    for case, image, truth in itertools.chain(ordinary, challenges(args.split)):
        for mode in (1, 2, 3, 4):
            attempts.append(attempt(case, image, truth, mode))
        print(case["id"], [a["experimental_accepted"] for a in attempts[-4:]], flush=True)
    report = dict(
        scope="Synthetic selective pipeline; no published/real figures",
        split=args.split,
        python=platform.python_version(),
        prereg_sha256=hashlib.sha256(
            (ROOT / "benchmarks/m12/preregister.json").read_bytes()
        ).hexdigest(),
        policy=policy_snapshot(),
        legend_policy=legend_policy_snapshot(),
        summary=summarise(attempts),
        attempts=attempts,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
