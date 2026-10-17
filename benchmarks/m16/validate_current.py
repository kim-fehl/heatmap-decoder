"""Pre-register and evaluate a versioned M16 partition against exact source/runtime hashes."""

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rapidocr_onnxruntime  # noqa: E402

from benchmarks.m16.validate import THRESHOLDS, fixture  # noqa: E402
from heatmap_decoder.automation import AutomationRequest, automate  # noqa: E402
from heatmap_decoder.extract import extract, extract_labels  # noqa: E402
from heatmap_decoder.models import DetectionHints, GridConfig, LabelsConfig, OCRLabelSource  # noqa: E402

POLICY = {
    **THRESHOLDS,
    "minimum_complete_fraction": 0.80,
    "maximum_wrong_complete": 0,
    "maximum_negative_complete": 0,
}
DEFAULT_SEED = 176300


def plan(seed_start):
    cases = []
    for index, (trajectory, nonlinear, vertical, reverse) in enumerate(
        itertools.product(["diverging", "asymmetric"], [False, True], [False, True], [False, True]),
        1,
    ):
        cases.append(
            dict(
                seed=seed_start + index,
                trajectory=trajectory,
                nonlinear=nonlinear,
                vertical=vertical,
                reverse=reverse,
                derivatives=["clean", "jpeg", "blur"],
            )
        )
    negatives = [
        dict(seed=seed_start + 17 + i, challenge=challenge, vertical=vertical)
        for i, (challenge, vertical) in enumerate(
            itertools.product(["plateau", "repeated", "edges"], [False, True])
        )
    ]
    return dict(
        seed_start=seed_start, cases=cases, negatives=negatives, label_modes=["sparse", "slots"]
    )


def fingerprint():
    paths = [
        *ROOT.joinpath("src").rglob("*.py"),
        ROOT / "benchmarks/m16/validate.py",
        Path(__file__),
        ROOT / "benchmarks/m16/current_trial.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        *ROOT.joinpath("examples/m16").glob("*.json"),
        ROOT / "examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg",
        ROOT / "examples/real_world/10.1038__s41598-024-53477-3__fig4.tsv",
    ]
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
    }
    model_root = Path(rapidocr_onnxruntime.__file__).parent / "models"
    runtime = dict(
        python=platform.python_version(),
        packages={
            n: importlib.metadata.version(n)
            for n in [
                "numpy",
                "Pillow",
                "scikit-image",
                "scipy",
                "pydantic",
                "rapidocr-onnxruntime",
                "onnxruntime",
                "opencv-python",
            ]
        },
        models={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(model_root.glob("*.onnx"))
        },
        font_sha256=hashlib.sha256(
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf").read_bytes()
        ).hexdigest(),
    )
    return sources, runtime


def save_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")


def freeze(path, seed_start):
    if 160100 <= seed_start <= 166300:
        raise ValueError("choose a partition outside the historical development/validation ranges")
    sources, runtime = fingerprint()
    frozen = dict(
        schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        policy=POLICY,
        plan=plan(seed_start),
        source_sha256=sources,
        runtime=runtime,
        note="Registered before evaluation. Same generator distribution; fresh base seeds. "
        "Derivatives stay with each base. No held-out publication claim.",
    )
    save_new(path, frozen)


def verify(frozen):
    sources, runtime = fingerprint()
    if frozen["source_sha256"] != sources:
        raise ValueError("frozen source set/hash mismatch; do not refresh a completed freeze")
    if frozen["runtime"] != runtime:
        raise ValueError("frozen runtime/model/font mismatch")
    if frozen["policy"] != POLICY or frozen["plan"] != plan(frozen["plan"]["seed_start"]):
        raise ValueError("frozen policy/case plan mismatch")


def numerical_metrics(image, config, truth):
    started = time.monotonic()
    result = extract(image, config)
    extraction_seconds = time.monotonic() - started
    finite = np.isfinite(result.values)
    displayed_range = abs(config.scale.anchors[-1].value - config.scale.anchors[0].value)
    error = np.abs(result.values[finite] - truth[finite]) / displayed_range
    fraction = float(finite.mean())
    flagged = float((result.flags != 0).mean())
    mae = float(error.mean()) if error.size else None
    maximum = float(error.max()) if error.size else None
    complete = (
        fraction == POLICY["finite_fraction"] and flagged <= POLICY["maximum_flagged_fraction"]
    )
    accurate = (
        fraction == 1
        and mae is not None
        and maximum is not None
        and mae < POLICY["mae_fraction"]
        and maximum <= POLICY["max_fraction"]
    )
    hints = DetectionHints(
        heatmap_bbox=config.heatmap_bbox,
        rows=config.rows,
        columns=config.columns,
        colourbar_bbox=config.colourbar_bbox,
        colourbar_orientation=config.colourbar_orientation,
        requested=(),
    )
    started = time.monotonic()
    gate = automate(
        image,
        AutomationRequest(hints=hints, numeric_only=True, scale=config.scale),
        experimental_export=True,
    )
    return dict(
        normalising_range=displayed_range,
        mae_fraction=mae,
        max_fraction=maximum,
        finite_fraction=fraction,
        flagged_fraction=flagged,
        bound_cells=int(np.count_nonzero(result.range_limit)),
        complete=complete,
        accurate=accurate,
        experimental_numeric_export_allowed=gate.automatic_export_allowed,
        gate_status=gate.complete.status,
        gate_reasons=[r for d in gate.decisions for r in d.reasons],
        extraction_seconds=extraction_seconds,
        gate_seconds=time.monotonic() - started,
    )


def label_metrics(image, config, mode):
    label_config = GridConfig(
        heatmap_bbox=config.heatmap_bbox,
        rows=7,
        columns=8,
        labels=LabelsConfig(
            rows=OCRLabelSource(source="ocr", bbox=(278, 35, 326, 245), recognition=mode),
            columns=OCRLabelSource(source="ocr", bbox=(35, 5, 275, 30), recognition=mode),
        ),
    )
    started = time.monotonic()
    result = extract_labels(image, label_config)
    seconds = time.monotonic() - started
    # Expected strings are consulted only after recognition.
    rows = []
    for axis in result.axes:
        for slot in axis.slots:
            expected = f"G_{slot.index}" if axis.axis == "rows" else f"S{slot.index}"
            rows.append(
                {
                    **asdict(slot),
                    "axis": axis.axis,
                    "reference": expected,
                    "exact": slot.raw_text == expected,
                }
            )
    hints = DetectionHints(
        heatmap_bbox=config.heatmap_bbox,
        rows=7,
        columns=8,
        colourbar_bbox=config.colourbar_bbox,
        colourbar_orientation=config.colourbar_orientation,
        labels=label_config.labels,
        requested=(),
    )
    gate = automate(
        image,
        AutomationRequest(hints=hints, numeric_only=False, scale=config.scale),
        experimental_export=True,
    )
    runs = [run for _, run in result.ocr_runs] + [run for _, _, _, run in result.slot_ocr_runs]
    return dict(
        mode=mode,
        identifier_export_allowed=gate.automatic_export_allowed,
        settings=label_config.labels.ocr.model_dump(mode="json"),
        slots=len(rows),
        raw_exact=sum(s["exact"] for s in rows),
        accepted=sum(s["status"] == "accepted" for s in rows),
        accepted_exact=sum(s["status"] == "accepted" and s["exact"] for s in rows),
        false_acceptance=sum(s["status"] == "accepted" and not s["exact"] for s in rows),
        unresolved=result.unresolved_count,
        slots_without_spatial_issue=sum(
            not any(
                i in s["issues"] for i in ["spans_slots", "axis_offset", "outside_owned_slot_field"]
            )
            for s in rows
        ),
        calls=len(runs),
        skipped=result.skipped_ocr_attempts,
        seconds=seconds,
        observations=rows,
        config=label_config.model_dump(mode="json"),
        raw_runs=[
            {
                "mode": r.mode,
                "geometry_kind": r.geometry_kind,
                "score_kind": r.score_kind,
                "tokens": [asdict(t) for t in r.tokens],
            }
            for r in runs
        ],
    )


def run(freeze_path, out):
    if out.exists():
        raise ValueError("refuse to overwrite evaluated evidence")
    frozen = json.loads(freeze_path.read_text())
    verify(frozen)  # Before generating or recognising any fresh validation image.
    started = time.monotonic()
    records = []
    for case in frozen["plan"]["cases"]:
        for derivative in case["derivatives"]:
            image, config, truth = fixture(
                case["seed"],
                case["trajectory"],
                case["nonlinear"],
                case["vertical"],
                case["reverse"],
                derivative,
            )
            record = {
                **case,
                "derivative": derivative,
                "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "numeric": numerical_metrics(image, config, truth),
                "labels": [
                    label_metrics(image, config, mode) for mode in frozen["plan"]["label_modes"]
                ],
            }
            record.pop("derivatives")
            records.append(record)
            print(
                f"{case['seed']} {derivative}: complete={record['numeric']['complete']}", flush=True
            )
    negatives = []
    for case in frozen["plan"]["negatives"]:
        image, config, truth = fixture(
            case["seed"], "diverging", True, case["vertical"], False, "clean", case["challenge"]
        )
        negatives.append({**case, "numeric": numerical_metrics(image, config, truth)})
    summary = dict(
        attempts=len(records),
        bases=len(frozen["plan"]["cases"]),
        complete=sum(r["numeric"]["complete"] for r in records),
        complete_accurate=sum(
            r["numeric"]["complete"] and r["numeric"]["accurate"] for r in records
        ),
        wrong_complete=sum(
            r["numeric"]["complete"] and not r["numeric"]["accurate"] for r in records
        ),
        numeric_export_allowed=sum(
            r["numeric"]["experimental_numeric_export_allowed"] for r in records
        ),
        wrong_numeric_export=sum(
            r["numeric"]["experimental_numeric_export_allowed"] and not r["numeric"]["accurate"]
            for r in records
        ),
        negative_attempts=len(negatives),
        negative_complete=sum(r["numeric"]["complete"] for r in negatives),
        negative_export_allowed=sum(
            r["numeric"]["experimental_numeric_export_allowed"] for r in negatives
        ),
        worst_finite_mae_fraction=max(r["numeric"]["mae_fraction"] for r in records),
        worst_finite_max_fraction=max(r["numeric"]["max_fraction"] for r in records),
        labels={
            mode: {
                key: sum(
                    next(label for label in r["labels"] if label["mode"] == mode)[key]
                    for r in records
                )
                for key in [
                    "slots",
                    "raw_exact",
                    "accepted",
                    "accepted_exact",
                    "false_acceptance",
                    "unresolved",
                    "slots_without_spatial_issue",
                    "calls",
                    "identifier_export_allowed",
                    "seconds",
                ]
            }
            for mode in frozen["plan"]["label_modes"]
        },
        seconds=time.monotonic() - started,
    )
    accepted = (
        summary["complete_accurate"] / summary["attempts"] >= POLICY["minimum_complete_fraction"]
        and summary["wrong_complete"] == 0
        and all(v["identifier_export_allowed"] == 0 for v in summary["labels"].values())
        and summary["wrong_numeric_export"] == 0
        and summary["negative_complete"] == 0
        and summary["negative_export_allowed"] == 0
    )
    verify(frozen)  # Also reject changes while a long evaluation is running.
    report = dict(
        schema_version=1,
        freeze_sha256=hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        acceptance_passed=accepted,
        policy=POLICY,
        summary=summary,
        observations_file=out.with_suffix(".jsonl").name,
        negatives=negatives,
        limitations=[
            "Supplied geometry and reviewed numeric anchors only.",
            "Correlated derivatives and shared synthetic generator.",
            "No identifiers or inferred counts exported.",
            "No held-out publication generalisation claim.",
            "Runtime is wall time, with model warm-up included in the first OCR call.",
        ],
    )
    with out.with_suffix(".jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
    report["observations_sha256"] = hashlib.sha256(
        out.with_suffix(".jsonl").read_bytes()
    ).hexdigest()
    save_new(out, report)
    print(json.dumps({"acceptance_passed": accepted, **summary}, indent=2))
    return accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("freeze")
    register.add_argument("--out", required=True, type=Path)
    register.add_argument("--seed-start", type=int, default=DEFAULT_SEED)
    evaluate = sub.add_parser("run")
    evaluate.add_argument("--freeze", required=True, type=Path)
    evaluate.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.out, args.seed_start)
    elif not run(args.freeze, args.out):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
