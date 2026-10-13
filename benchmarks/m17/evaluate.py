"""Freeze then evaluate fresh bases. Run from repository root with uv run python."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from benchmarks.m17.fixtures import fixture  # noqa: E402
from heatmap_decoder.automation import AutomationRequest, automate  # noqa: E402
from heatmap_decoder.blocks import BLOCK_POLICY  # noqa: E402
from heatmap_decoder.extract import extract  # noqa: E402
from heatmap_decoder.geometry import axis_slots  # noqa: E402
from heatmap_decoder.models import DetectionHints  # noqa: E402

FAMILIES = ("none", "horizontal", "vertical", "both")
TARGETS = dict(
    explicit_clean=1.0,
    auto_clean_none=0.80,
    auto_clean_blocked=0.70,
    auto_derivatives=0.70,
    maximum_wrong_complete=0,
    maximum_negative_complete=0,
    maximum_mae=0.02,
    maximum_error=0.05,
    maximum_flagged_fraction=0.01,
)
PLAN = [
    dict(seed=180001 + i, family=f, derivatives=["clean", "jpeg", "blur"])
    for i, f in enumerate(f for f in FAMILIES for _ in range(10))
]
NEGATIVES = [
    dict(
        seed=181001 + i,
        challenge=c,
        family="vertical" if c == "panels" else "none" if c == "white" else "both",
    )
    for i, c in enumerate(
        c
        for c in ["missing", "white", "annotation", "panels", "cropped", "wrong_count"]
        for _ in range(2)
    )
]


def fingerprint():
    files = [
        *ROOT.joinpath("src").rglob("*.py"),
        *ROOT.joinpath("tests").glob("*.py"),
        *Path(__file__).parent.glob("*.py"),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    return dict(
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)
        },
        runtime=dict(
            python=platform.python_version(),
            packages={
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "Pillow",
                    "scikit-image",
                    "scipy",
                    "pydantic",
                    "rapidocr-onnxruntime",
                    "onnxruntime",
                ]
            },
        ),
    )


def write_new(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def frozen():
    return dict(
        **fingerprint(),
        targets=TARGETS,
        policy=BLOCK_POLICY,
        cases=PLAN,
        negatives=NEGATIVES,
        eligibility="All 40 preregistered bases and all derivatives; no exclusions. "
        "Axis-aligned common-pitch cells, one shared linear reviewed legend. "
        "M18 angles/OCR and M19 OCR scales are not tested or qualified here.",
    )


def metrics(image, config, truth):
    start = time.monotonic()
    r = extract(image, config)
    finite = np.isfinite(r.values)
    error = (
        np.abs(r.values[finite] - truth[finite])
        if truth.shape == r.values.shape
        else np.array([np.inf])
    )
    mae = float(error.mean()) if error.size else None
    maximum = float(error.max()) if error.size else None
    complete = bool(finite.all() and (r.flags != 0).mean() <= 0.01)
    accurate = bool(complete and mae is not None and mae < 0.02 and maximum <= 0.05)
    return dict(
        seconds=time.monotonic() - start,
        shape=list(r.values.shape),
        finite_fraction=float(finite.mean()),
        flagged_fraction=float((r.flags != 0).mean()),
        mae=mae,
        maximum=maximum,
        complete=complete,
        accurate=accurate,
    )


def attempt(case, derivative):
    image, c, truth = fixture(case["seed"], case["family"], derivative)
    explicit = metrics(image, c, truth)
    hints = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    start = time.monotonic()
    r = automate(image, AutomationRequest(hints=hints, numeric_only=True), experimental_export=True)
    automatic = None
    if (
        r.diagnostic_config is not None
        and r.diagnostic_config.rows == c.rows
        and r.diagnostic_config.columns == c.columns
    ):
        automatic = metrics(image, r.diagnostic_config, truth)
        automatic["exact_intervals"] = all(
            axis_slots(c, a) == axis_slots(r.diagnostic_config, a) for a in ("rows", "columns")
        )
    correct = bool(
        r.complete.status == "accepted"
        and automatic
        and automatic["accurate"]
        and automatic["exact_intervals"]
    )
    return dict(
        **case,
        derivative=derivative,
        image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        supplied_hints=dict(geometry=False, counts=False, reviewed_scale=True, identifiers=False),
        explicit=explicit,
        automatic=automatic,
        complete_status=r.complete.status,
        automatic_correct=correct,
        wrong_complete=r.complete.status == "accepted" and not correct,
        export_allowed=r.automatic_export_allowed,
        seconds=time.monotonic() - start,
        decisions=[d.model_dump(mode="json") for d in r.decisions],
        block_proposals=[p.model_dump(mode="json") for p in r.detection.block_proposals],
    )


def evaluate(path, output):
    expected = json.loads(Path(path).read_text())
    if expected != frozen():
        raise ValueError(
            "Frozen source/runtime/plan mismatch; create a new partition "
            "rather than refreshing a freeze"
        )
    records = []
    for case in PLAN:
        for derivative in case["derivatives"]:
            row = attempt(case, derivative)
            records.append(row)
            print(
                case["seed"],
                case["family"],
                derivative,
                row["complete_status"],
                row["automatic_correct"],
                flush=True,
            )
    negatives = []
    for case in NEGATIVES:
        image, c, _ = fixture(
            case["seed"],
            case["family"],
            challenge=None if case["challenge"] == "wrong_count" else case["challenge"],
        )
        hints = DetectionHints(
            rows=c.rows + 1 if case["challenge"] == "wrong_count" else None,
            infer_rows=True,
            infer_columns=True,
            requested=("heatmap", "colourbar"),
            colourbar_start_value=c.colourbar_start_value,
            colourbar_end_value=c.colourbar_end_value,
        )
        r = automate(
            image, AutomationRequest(hints=hints, numeric_only=True), experimental_export=True
        )
        negatives.append(
            dict(
                **case,
                status=r.complete.status,
                export_allowed=r.automatic_export_allowed,
                decisions=[d.model_dump(mode="json") for d in r.decisions],
            )
        )
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for derivative in ("clean", "jpeg", "blur"):
            subset = [r for r in records if r["family"] == family and r["derivative"] == derivative]
            summary[family][derivative] = dict(
                bases=len(subset),
                explicit_accurate=sum(r["explicit"]["accurate"] for r in subset),
                automatic_accurate=sum(r["automatic_correct"] for r in subset),
            )
    clean = [r for r in records if r["derivative"] == "clean"]
    gates = dict(
        explicit_clean=all(r["explicit"]["accurate"] for r in clean),
        auto_clean_none=summary["none"]["clean"]["automatic_accurate"] >= 8,
        **{f"auto_clean_{f}": summary[f]["clean"]["automatic_accurate"] >= 7 for f in FAMILIES[1:]},
        auto_derivatives=sum(r["automatic_correct"] for r in records) >= 84,
        zero_wrong_complete=not any(r["wrong_complete"] for r in records),
        zero_negative_complete=not any(r["status"] == "accepted" for r in negatives),
    )
    report = dict(
        freeze_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        summary=summary,
        gates=gates,
        bases=40,
        correlated_attempts=120,
        negative_bases=12,
        records=records,
        negatives=negatives,
        note="Derivatives are correlated. Ten bases/family give limited base-level evidence. "
        "No publication generalisation or M19 export qualification.",
    )
    write_new(output, report)
    print(json.dumps(dict(summary=summary, gates=gates), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "evaluate"])
    parser.add_argument("--freeze", default="benchmarks/m17/freeze.json")
    parser.add_argument("--out", default="benchmarks/m17/validation.json")
    parser.add_argument("--seed-start", type=int, default=180001)
    parser.add_argument("--negative-seed-start", type=int, default=181001)
    args = parser.parse_args()
    for index, case in enumerate(PLAN):
        case["seed"] = args.seed_start + index
    for index, case in enumerate(NEGATIVES):
        case["seed"] = args.negative_seed_start + index
    if args.mode == "freeze":
        write_new(args.freeze, frozen())
    else:
        evaluate(args.freeze, args.out)
