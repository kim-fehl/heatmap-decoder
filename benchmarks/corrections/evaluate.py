"""Deployed correction policy first; synthetic truth scoring only afterwards."""

import argparse
import hashlib
import json
import platform
import runpy
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from heatmap_decoder import (
    AutomationRequest,
    DetectionHints,
    LegendConfig,
    ScaleConfig,
    automate,
    extract,
    inspect_legend,
)
from heatmap_decoder.automation import AUTOMATION_POLICY_VERSION
from heatmap_decoder.detection import DETECTOR_VERSION
from heatmap_decoder.detection_policy import policy_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/corrections/generate.py"))
PREREG = json.loads((ROOT / "benchmarks/corrections/preregister.json").read_text())


def request(truth, mode, wrong=False):
    supplied = mode in (1, 3)
    ticks = mode in (3, 4)
    hints = dict(
        rows=truth["rows"] + int(wrong) if supplied else None,
        columns=truth["columns"] if supplied else None,
        infer_rows=not supplied,
        infer_columns=not supplied,
        requested=("heatmap", "colourbar"),
    )
    if not supplied:
        hints["detector"] = {"count_label_geometry": True}
    if not ticks:
        hints.update(
            colourbar_start_value=truth["start_value"], colourbar_end_value=truth["end_value"]
        )
    return AutomationRequest(
        hints=DetectionHints(**hints),
        numeric_only=True,
        scale=ScaleConfig(source="ocr_ticks") if ticks else None,
    )


def tick_metrics(result, truth):
    expected = truth["ticks"]
    matches = []
    for tick in expected:
        observed = [
            t
            for t in result.ticks
            if t.position_source == "stroke" and abs(t.position - tick["position"]) <= 1
        ]
        unique = len(observed) == 1
        t = observed[0] if unique else None
        matches.append(
            dict(
                expected=tick["value"],
                stroke=unique,
                text=bool(t and t.text_bbox is not None and t.raw_text.strip()),
                exact=bool(t and t.parsed_value == tick["value"]),
                sign_error=bool(
                    t
                    and t.parsed_value is not None
                    and np.sign(t.parsed_value) != np.sign(tick["value"])
                ),
                raw=t.raw_text if t else None,
            )
        )
    return dict(
        expected=len(expected),
        strokes=sum(t["stroke"] for t in matches),
        localised=sum(t["text"] for t in matches),
        exact=sum(t["exact"] for t in matches),
        sign_errors=sum(t["sign_error"] for t in matches),
        matches=matches,
        complete_exact=all(t["exact"] for t in matches),
        eligible=result.development_eligible,
        result=result.model_dump(mode="json"),
    )


def attempt(case, image, truth, mode):
    req = request(truth, mode, case.get("negative") == "wrong_counts")
    start = time.perf_counter()
    result = automate(image, req, experimental_export=True)
    seconds = time.perf_counter() - start
    accepted = result.complete.status == "accepted"
    by_kind = {d.requirement: d for d in result.decisions}
    candidates = {c.id: c for c in result.detection.candidates}
    geometry = {}
    for kind in ("heatmap", "colourbar"):
        d = by_kind[kind]
        if d.status != "accepted":
            continue
        box = candidates[d.candidate_ids[0]].bbox
        expected = np.array(truth[kind + "_bbox"])
        tolerances = np.ones(4)
        if kind == "heatmap":
            tolerances = 0.1 * np.array(
                [
                    (expected[2] - expected[0]) / truth["columns"],
                    (expected[3] - expected[1]) / truth["rows"],
                ]
                * 2
            )
        errors = np.abs(np.array(box) - expected)
        geometry[kind] = dict(
            bbox=box, errors=errors.tolist(), correct=bool(np.all(errors <= tolerances))
        )
    counts = {
        axis: dict(
            status=by_kind[axis].status,
            selected=result.detection.selected_counts[axis].count
            if axis in result.detection.selected_counts
            else None,
        )
        for axis in ("rows", "columns")
    }
    count_exact = all(counts[axis]["selected"] == truth[axis] for axis in counts)
    numeric = None
    if result.diagnostic_config is not None:
        recovered = extract(image, result.diagnostic_config)
        expected = np.array(truth["expected_matrix"])
        if recovered.values.shape == expected.shape:
            finite = np.isfinite(recovered.values)
            errors = np.abs(recovered.values[finite] - expected[finite]) / abs(
                truth["end_value"] - truth["start_value"]
            )
            numeric = dict(
                mae=float(errors.mean()) if len(errors) else None,
                maximum=float(errors.max()) if len(errors) else None,
                finite=float(finite.mean()),
                flagged=float(np.mean(recovered.flags != 0)),
            )
    ticks = tick_metrics(result.legend, truth) if result.legend else None
    correct = (
        len(geometry) == 2
        and all(g["correct"] for g in geometry.values())
        and count_exact
        and numeric is not None
        and numeric["mae"] is not None
        and numeric["mae"] < 0.02
        and numeric["maximum"] <= 0.05
        and numeric["finite"] == 1
        and numeric["flagged"] <= 0.01
        and (mode not in (3, 4) or ticks is not None and ticks["complete_exact"])
    )
    negative_applies = mode in PREREG["negative_models"].get(case.get("negative"), [])
    return dict(
        **case,
        mode=mode,
        seconds=seconds,
        complete=accepted,
        experimental_export_allowed=result.automatic_export_allowed,
        correct_complete=bool(correct),
        negative_applies=negative_applies,
        geometry=geometry,
        counts=counts,
        numeric=numeric,
        ticks=ticks,
        decisions=[d.model_dump(mode="json") for d in result.decisions],
        release_reasons=result.release_reasons,
        diagnostic_tick_runs=len(result.legend_diagnostics),
        bar_search_evidence=result.detection.bar_search_evidence,
    ), result


def negatives(split):
    for i, name in enumerate(PREREG["negative_models"]):
        case = dict(
            id=f"{split}-negative-{name}",
            base_id=f"{split}-negative-{name}",
            index=0,
            seed=(239000 if split == "development" else 249000) + i,
            renderer="pillow",
            degradation="clean",
            split=split,
            negative=name,
        )
        image, truth = FIXTURE["make_fixture"](case)
        x, y, r, s = truth["colourbar_bbox"]
        hx, hy, hr, hs = truth["heatmap_bbox"]
        if name == "absent_bar":
            image[y:s, x:r] = 255
        elif name == "categorical":
            image[y : (y + s) // 2, x:r] = [255, 0, 0]
            image[(y + s) // 2 : s, x:r] = [0, 0, 255]
        elif name == "competing_bars":
            image[y:s, 40 : 40 + r - x] = image[y:s, x:r]
        elif name == "multipanel":
            image = np.concatenate([image, image], axis=1)
        elif name == "hidden_boundaries":
            image[hy:hs, hx:hr] = [235, 235, 235]
        elif name == "partial_crop":
            image = image[hy + 20 :].copy()
            truth["heatmap_bbox"] = (hx, 0, hr, hs - hy - 20)
            truth["colourbar_bbox"] = (x, 0, r, s - hy - 20)
        elif name in ("missing_tick_text", "two_ticks", "nonlinear_ticks", "touching_core_text"):
            figure = Image.fromarray(image)
            draw = ImageDraw.Draw(figure)
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
            middle = next(
                t
                for t in truth["text_boxes"]
                if t["kind"] == "tick" and t["text"] == truth["ticks"][1]["text"]
            )
            if name == "touching_core_text":
                draw.text((x, (y + s) // 2), "0 units", font=font, fill="black", anchor="lm")
            else:
                a, b, c, d = middle["bbox"]
                draw.rectangle((a - 1, b - 1, c + 1, d + 1), fill="white")
                p = truth["ticks"][1]["position"]
                if name == "two_ticks":
                    draw.rectangle((x - 9, p - 1, x - 1, p + 1), fill="white")
                    draw.rectangle((r + 1, p - 1, r + 9, p + 1), fill="white")
                elif name == "nonlinear_ticks":
                    draw.text(
                        (middle["x"], middle["y"]), "99", font=font, anchor="lm", fill="black"
                    )
            image = np.array(figure)
        yield case, image, truth


def summary(attempts):
    report = {}
    for mode in (1, 2, 3, 4):
        ordinary = [a for a in attempts if a["mode"] == mode and not a.get("negative")]
        accepted = [a for a in ordinary if a["complete"]]
        negative = [a for a in attempts if a["mode"] == mode and a["negative_applies"]]
        numeric = [a["numeric"] for a in accepted if a["numeric"]]
        report[str(mode)] = dict(
            ordinary=len(ordinary),
            complete=len(accepted),
            coverage=len(accepted) / len(ordinary),
            distinct_accepted_bases=len({a["base_id"] for a in accepted}),
            wrong_complete=sum(not a["correct_complete"] for a in accepted),
            negative_attempts=len(negative),
            negative_complete=sum(a["complete"] for a in negative),
            experimental_exports=sum(a["experimental_export_allowed"] for a in ordinary),
            negative_experimental_exports=sum(a["experimental_export_allowed"] for a in negative),
            heatmap_passes=sum(
                next(d for d in a["decisions"] if d["requirement"] == "heatmap")["status"]
                == "accepted"
                for a in ordinary
            ),
            bar_passes=sum(
                next(d for d in a["decisions"] if d["requirement"] == "colourbar")["status"]
                == "accepted"
                for a in ordinary
            ),
            statuses=dict(
                Counter(
                    d["reasons"][0]
                    for a in ordinary
                    for d in a["decisions"]
                    if d["status"] != "accepted"
                )
            ),
            worst_mae=max((n["mae"] for n in numeric), default=None),
            worst_error=max((n["maximum"] for n in numeric), default=None),
            median_seconds=float(np.median([a["seconds"] for a in ordinary])),
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditional-ocr", action="store_true")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.shards <= PREREG["base_splits"][args.split]["bases"]:
        parser.error("shards must be between 1 and the number of bases")
    if not 0 <= args.shard_index < args.shards:
        parser.error("shard index must be in [0, shards)")
    if args.out.exists():
        parser.error("report exists; evidence must not be overwritten")
    attempts, conditional = [], []
    for case in FIXTURE["cases"](args.split):
        if case["index"] % args.shards != args.shard_index:
            continue
        image, truth = FIXTURE["make_fixture"](case)
        runtime = None
        for mode in (1, 2, 3, 4):
            attempt_record, result = attempt(case, image, truth, mode)
            attempts.append(attempt_record)
            if mode == 3:
                runtime = result
        if args.conditional_ocr:
            ideal = inspect_legend(
                image,
                LegendConfig(
                    colourbar_bbox=truth["colourbar_bbox"],
                    colourbar_orientation=truth["orientation"],
                ),
            )
            bars = [c for c in runtime.detection.candidates if c.kind == "colourbar"]
            measured = None
            if len(bars) == 1:
                c = bars[0]
                measured = inspect_legend(
                    image, LegendConfig(colourbar_bbox=c.bbox, colourbar_orientation=c.orientation)
                )
            conditional.append(
                dict(
                    id=case["id"],
                    condition="Diagnostic only; no automation coverage",
                    annotated=tick_metrics(ideal, truth),
                    unique_detected=tick_metrics(measured, truth) if measured else None,
                )
            )
        print(case["id"], [a["complete"] for a in attempts[-4:]], flush=True)
    for i, (case, image, truth) in enumerate(negatives(args.split)):
        if i % args.shards != args.shard_index:
            continue
        for mode in (1, 2, 3, 4):
            record, _ = attempt(case, image, truth, mode)
            attempts.append(record)
        print(case["id"], [a["complete"] for a in attempts[-4:]], flush=True)
    report = dict(
        purpose=PREREG["purpose"],
        split=args.split,
        shard=dict(index=args.shard_index, total=args.shards),
        python=platform.python_version(),
        detector=DETECTOR_VERSION,
        pipeline=AUTOMATION_POLICY_VERSION,
        policy=policy_snapshot(),
        prereg_sha256=hashlib.sha256(
            (ROOT / "benchmarks/corrections/preregister.json").read_bytes()
        ).hexdigest(),
        summary=summary(attempts),
        attempts=attempts,
        conditional_ocr=conditional,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
