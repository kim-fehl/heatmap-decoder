"""M18 frozen geometry and real OCR evidence. Truth is evaluator-only."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import rapidocr_onnxruntime  # noqa: E402

from benchmarks.m18.fixtures import FONT, fixture  # noqa: E402
from heatmap_decoder.angle_search import ANGLE_POLICY, estimate_angle  # noqa: E402
from heatmap_decoder.extract import extract, extract_labels  # noqa: E402
from heatmap_decoder.geometry import resolve_grid  # noqa: E402
from heatmap_decoder.oriented_labels import (  # noqa: E402
    LINE_POLICY,
    LabelBudget,
    associate_lines,
    baseline_direction_evidence,
    segment_lines,
)

ANGLES = [0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90, 37.25]
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
PLAN = [
    dict(
        seed=230001 + i,
        angle=a,
        below=bool((i + i // 14) % 2),
        blocked=bool((i // 2) % 2),
        font_size=[14, 15, 16][i % 3],
        font_path=FONT if i % 2 else MONO,
        header=i % 7 == 0,
        derivatives=["clean", "jpeg", "blur"],
    )
    for i, a in enumerate(ANGLES * 3)
]
TARGETS = dict(
    explicit_clean_correct=1.0,
    automatic_clean_correct=0.8,
    maximum_wrong_accepted_associations=0,
    maximum_angle_error_degrees=1.0,
    identifiers_automatically_authorised=0,
)


def fingerprint():
    paths = [
        *ROOT.joinpath("src").rglob("*.py"),
        *ROOT.joinpath("tests").glob("*.py"),
        *Path(__file__).parent.glob("*.py"),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    model = Path(rapidocr_onnxruntime.__file__).parent / "models"
    return dict(
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)
        },
        runtime=dict(
            python=platform.python_version(),
            packages={
                p: importlib.metadata.version(p)
                for p in [
                    "numpy",
                    "Pillow",
                    "scipy",
                    "scikit-image",
                    "pydantic",
                    "pytesseract",
                    "rapidocr-onnxruntime",
                    "onnxruntime",
                ]
            },
            font_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [FONT, MONO]},
            model_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(model.glob("*.onnx"))
            },
            tesseract=subprocess.check_output(["tesseract", "--version"], text=True).splitlines()[
                0
            ],
        ),
    )


def frozen():
    return dict(
        **fingerprint(),
        policy=dict(angle=ANGLE_POLICY, line=LINE_POLICY),
        targets=TARGETS,
        plan=PLAN,
        eligibility="All 42 clean bases, dark single-line sample labels, "
        "common angle, above/below, "
        "explicit selected label/matrix regions and counts, 14-16px DejaVu Sans/Mono. "
        "Explicit-angle path gets only configured angle, auto gets no truth angle/strings. "
        "Negative all-capital/overlapping/truncated/incorrect-attachment cases "
        "remain in the report.",
        workflows=[
            "explicit geometry",
            "automatic axis and directed-angle/attachment proposals",
            "real Tesseract and RapidOCR raw/unreviewed identifier evidence",
            "reviewed numeric invariance",
        ],
    )


def write_new(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


def geometry_trial(image, c, source, budget=None):
    crop, tr, lines, seg = segment_lines(image, source, budget=budget)
    lines, g = associate_lines(lines, c, source)
    direction = baseline_direction_evidence(crop, lines) if crop else {"status": "unresolved"}
    return dict(segmentation=seg, association=g, reading_direction=direction)


def correct_association(trial, truth):
    lines = [
        line
        for line in trial["association"].get("lines", [])
        if line["slot_index"] is not None and not line["issues"]
    ]
    expected = {a["slot"]: a for a in truth}
    return (
        len(lines) == len(expected)
        and {line["slot_index"] for line in lines} == set(expected)
        and all(
            abs(line["attachment"][0] - expected[line["slot_index"]]["attachment"][0]) < 6
            for line in lines
        )
    )


def attempt(case, derivative):
    started = time.monotonic()
    kw = {k: v for k, v in case.items() if k not in ("derivatives",)}
    image, c, n, values, truth = fixture(**kw, derivative=derivative)
    explicit = geometry_trial(image, c, c.labels.columns)
    budget = LabelBudget()
    estimate = estimate_angle(image, c.labels.columns.bbox, budget)
    alternatives = []
    for angle in estimate.get("angle_alternatives", ()):
        source = c.labels.columns.model_copy(update={"text_rotation_degrees": angle})
        trial = geometry_trial(image, c, source, budget)
        supported = (
            estimate["status"] == "unique_axis"
            and trial["association"]["status"] == "accepted"
            and trial["association"].get("complete")
            and trial["reading_direction"]["status"] != "inverted"
        )
        alternatives.append(dict(angle=angle, supported=supported, **trial))
    supported = [a for a in alternatives if a["supported"]]
    accepted = len(supported) == 1
    auto_correct = (
        accepted
        and abs((supported[0]["angle"] - case["angle"] + 180) % 360 - 180) <= 1
        and correct_association(supported[0], truth)
    )
    numeric = extract(image, n.model_copy(update={"labels": None}))
    finite = np.isfinite(numeric.values)
    error = np.abs(numeric.values[finite] - values[finite])
    return dict(
        **case,
        derivative=derivative,
        source_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        supplied_hints=dict(
            matrix_and_label_regions=True,
            counts=True,
            explicit_angle=True,
            truth_strings_or_polygons=False,
            auto_angle=False,
        ),
        explicit=explicit,
        explicit_correct=correct_association(explicit, truth),
        angle_estimation=estimate,
        automatic_alternatives=alternatives,
        automatic_accepted=accepted,
        automatic_correct=bool(auto_correct),
        wrong_accepted=accepted and not auto_correct,
        numeric=dict(
            shape=list(numeric.values.shape),
            finite_fraction=float(finite.mean()),
            flagged_fraction=float((numeric.flags != 0).mean()),
            mae=float(error.mean()) if error.size else None,
            maximum=float(error.max()) if error.size else None,
            grid_sha256=resolve_grid(c).metadata()["sha256"],
        ),
        seconds=time.monotonic() - started,
    )


def real_ocr():
    records = []
    # Every required angle has a real engine check, plus non-integer and reverse directions.
    for engine in ["tesseract", "rapidocr"]:
        angles = ANGLES if engine == "tesseract" else [15, -45, 75, 37.25]
        for i, angle in enumerate(angles):
            image, c, _, _, truth = fixture(
                angle, below=bool(i % 2), blocked=bool(i % 2), seed=223001 + i
            )
            labels = c.labels.model_copy(
                update={"ocr": c.labels.ocr.model_copy(update={"engine": engine})}
            )
            result = extract_labels(image, c.model_copy(update={"labels": labels}))
            expected = {t["slot"]: t["text"] for t in truth}
            slots = result.axes[1].slots
            raw_correct = sum(s.raw_text == expected[s.index] for s in slots)
            false_heuristic = sum(
                s.raw_text is not None
                and s.confidence is not None
                and s.confidence >= 80
                and s.raw_text != expected[s.index]
                for s in slots
            )
            records.append(
                dict(
                    engine=engine,
                    angle=angle,
                    raw_correct=raw_correct,
                    total=len(slots),
                    false_heuristic_acceptances=false_heuristic,
                    reviewed_identifiers=sum(s.status == "manual" for s in slots),
                    unreviewed=sum(s.status == "unreviewed" for s in slots),
                    geometry=result.axes[1].geometry_evidence,
                    slots=[
                        dict(
                            index=s.index,
                            raw=s.raw_text,
                            final=s.final_text,
                            status=s.status,
                            confidence=s.confidence,
                        )
                        for s in slots
                    ],
                    engine_records=result.oriented_ocr_records,
                    skipped=result.skipped_ocr_attempts,
                )
            )
            print("OCR", engine, angle, raw_correct, "/", len(slots), flush=True)
    return records


def negatives():
    records = []
    for kind in ["all_caps", "tight_overlap", "truncated", "wrong_endpoint", "missing_with_header"]:
        kwargs = dict(angle=30, seed=224001)
        if kind == "all_caps":
            kwargs["names"] = ["SAMPLE1234"] * 4
        if kind == "tight_overlap":
            kwargs.update(angle=15, pitch=24)
        if kind == "missing_with_header":
            kwargs.update(missing=(1,), header=True)
        image, c, *rest = fixture(**kwargs)
        source = c.labels.columns
        if kind == "truncated":
            source = source.model_copy(update={"bbox": (300, *source.bbox[1:])})
        if kind == "wrong_endpoint":
            source = source.model_copy(update={"attachment_endpoint": "end"})
        explicit = geometry_trial(image, c, source)
        estimate = estimate_angle(image, source.bbox, LabelBudget())
        trials = []
        for angle in estimate.get("angle_alternatives", ()):
            trial = geometry_trial(
                image, c, source.model_copy(update={"text_rotation_degrees": angle})
            )
            trials.append(dict(angle=angle, **trial))
        supports = [
            t
            for t in trials
            if estimate["status"] == "unique_axis"
            and t["association"]["status"] == "accepted"
            and t["association"].get("complete")
            and t["reading_direction"]["status"] != "inverted"
        ]
        records.append(
            dict(
                kind=kind,
                explicit=explicit,
                estimate=estimate,
                trials=trials,
                directed_accepted=len(supports) == 1,
            )
        )
    return records


def evaluate(freeze_path, out):
    if json.loads(Path(freeze_path).read_text()) != frozen():
        raise ValueError("Frozen source/runtime/policy/inputs mismatch")
    records = []
    for case in PLAN:
        for derivative in case["derivatives"]:
            r = attempt(case, derivative)
            records.append(r)
            print(
                case["seed"],
                case["angle"],
                derivative,
                r["explicit_correct"],
                r["automatic_correct"],
                flush=True,
            )
    ocr = real_ocr()
    negative = negatives()
    clean = [r for r in records if r["derivative"] == "clean"]
    counts = {
        d: dict(
            attempts=42,
            explicit_correct=sum(r["explicit_correct"] for r in records if r["derivative"] == d),
            automatic_correct=sum(r["automatic_correct"] for r in records if r["derivative"] == d),
        )
        for d in ["clean", "jpeg", "blur"]
    }
    gates = dict(
        explicit_clean=all(r["explicit_correct"] for r in clean),
        auto_clean=sum(r["automatic_correct"] for r in clean) >= 34,
        no_wrong_accepted=not any(r["wrong_accepted"] for r in records),
        no_wrong_negative_acceptance=not any(r["directed_accepted"] for r in negative),
    )
    write_new(
        out,
        dict(
            freeze_sha256=hashlib.sha256(Path(freeze_path).read_bytes()).hexdigest(),
            summary=counts,
            gates=gates,
            records=records,
            ocr=ocr,
            negatives=negative,
            note="42 independent bases, 126 correlated derivatives. "
            "Spelling metrics do not authorise identifiers. "
            "No new publication generalisation claim; M17 degradation gate unchanged.",
        ),
    )
    print(json.dumps(dict(summary=counts, gates=gates), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["freeze", "evaluate"])
    p.add_argument("--freeze", default="benchmarks/m18/freeze.json")
    p.add_argument("--out", default="benchmarks/m18/validation.json")
    args = p.parse_args()
    if args.mode == "freeze":
        write_new(args.freeze, frozen())
    else:
        evaluate(args.freeze, args.out)
