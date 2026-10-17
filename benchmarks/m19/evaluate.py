"""Registered M19 workflows: no truth enters detection or tick recognition."""

import argparse
import hashlib
import json
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from benchmarks.m18.evaluate import attempt as label_attempt  # noqa: E402
from benchmarks.m18.evaluate import fingerprint as previous_fingerprint  # noqa: E402
from benchmarks.m18.fixtures import fixture as labels_fixture  # noqa: E402
from benchmarks.m19.fixtures import FONT, MONO, fixture  # noqa: E402
from heatmap_decoder import AutomationRequest, DetectionHints, ScaleConfig, automate, extract  # noqa: E402
from heatmap_decoder.geometry import axis_slots  # noqa: E402
from heatmap_decoder.models import LegendOptions, OCRConfig  # noqa: E402

FAMILIES = ["none", "horizontal", "vertical", "both"]
ANGLES = [15, -15, 30, -30, 45, -45, 60, -60, 75, -75]
PLAN = [
    dict(
        seed=296001 + i,
        family=FAMILIES[i // 10],
        angle=ANGLES[i % 10],
        derivatives=["clean", "jpeg", "blur"],
    )
    for i in range(40)
]
NEGATIVES = [
    dict(seed=297001 + i, family="both", challenge=c)
    for i, c in enumerate(
        [
            "missing",
            "white",
            "annotation",
            "panels",
            "wrong_count",
            "truncated",
            "competing_legends",
            "inconsistent_ticks",
            "unsupported_edges",
            "inconsistent_ticks",
            "competing_legends",
            "unsupported_edges",
        ]
    )
]
OPTIONS = LegendOptions(
    ocr=OCRConfig(engine="tesseract"),
    retry_unconstrained_numeric=True,
    retry_white_padding_pixels=(1, 2),
)
TARGETS = dict(
    explicit_clean=1.0,
    auto_clean_none=0.8,
    auto_clean_blocked=0.7,
    auto_all_derivatives=0.7,
    full_auto_clean=0.7,
    explicit_ocr_clean=0.7,
    explicit_angle_clean=1.0,
    auto_angle_clean=0.8,
    wrong_complete=0,
    mae=0.02,
    maximum=0.05,
    flagged_fraction=0.01,
)


def frozen():
    fingerprint = previous_fingerprint()
    for p in [
        *Path(__file__).parent.glob("*.py"),
        ROOT / "benchmarks/m17/fixtures.py",
        ROOT / "benchmarks/m18/fixtures.py",
    ]:
        fingerprint["source_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    return dict(
        **fingerprint,
        plan=PLAN,
        negatives=NEGATIVES,
        targets=TARGETS,
        legend_options=OPTIONS.model_dump(mode="json"),
        eligibility="All 40 bases, all three derivatives, no retrospective exclusions. "
        "Dark Sans/Mono 18-20px three physical signed integer ticks, common-pitch "
        "regular/blocked matrices, three palettes, both bar directions/orientations. "
        "Label workflows use paired independent wide-pitch M18 renderings; "
        "their ROI/counts are supplied and do not qualify full-figure label localisation.",
        qualification="No export gate changes during validation. "
        "Per-image capability decisions measured separately.",
    )


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def measured(image, config, truth, anchors=None):
    r = extract(image, config, ocr_anchors=anchors)
    same = r.values.shape == truth.shape
    finite = np.isfinite(r.values)
    error = np.abs(r.values[finite] - truth[finite]) / 6 if same else np.array([1.0])
    mae = float(error.mean()) if len(error) else None
    maximum = float(error.max()) if len(error) else None
    accurate = bool(
        same
        and finite.all()
        and np.mean(r.flags != 0) <= 0.01
        and mae is not None
        and mae < 0.02
        and maximum <= 0.05
    )
    return dict(
        shape=list(r.values.shape),
        finite_fraction=float(finite.mean()),
        flagged_fraction=float(np.mean(r.flags != 0)),
        mae=mae,
        maximum=maximum,
        accurate=accurate,
    )


def workflow(image, c, truth, explicit=False, ocr=False):
    settings = dict(requested=("heatmap", "colourbar"), infer_rows=True, infer_columns=True)
    if explicit:
        settings.update(
            requested=(),
            infer_rows=False,
            infer_columns=False,
            heatmap_bbox=c.heatmap_bbox,
            colourbar_bbox=c.colourbar_bbox,
            colourbar_orientation=c.colourbar_orientation,
            rows=c.rows,
            columns=c.columns,
            rows_segments=c.rows_segments,
            columns_segments=c.columns_segments,
        )
    if not ocr:
        settings.update(
            colourbar_start_value=c.colourbar_start_value, colourbar_end_value=c.colourbar_end_value
        )
    request = AutomationRequest(
        numeric_only=True,
        hints=DetectionHints(**settings),
        scale=ScaleConfig(source="ocr_ticks", legend=OPTIONS) if ocr else None,
    )
    started = time.monotonic()
    r = automate(image, request, experimental_export=True)
    metrics = None
    exact = False
    if r.diagnostic_config is not None:
        metrics = measured(image, r.diagnostic_config, truth, r.ocr_anchors or None)
        exact = all(
            axis_slots(c, a) == axis_slots(r.diagnostic_config, a) for a in ("rows", "columns")
        )
    correct = bool(
        r.numeric_complete.status == "accepted" and metrics and metrics["accurate"] and exact
    )
    blockers = [d for d in r.decisions if d.status != "accepted"]
    return dict(
        supplied_hints=request.model_dump(mode="json"),
        numeric_state=r.numeric_state,
        correct=correct,
        wrong_complete=r.numeric_complete.status == "accepted" and not correct,
        exact_intervals=exact,
        metrics=metrics,
        export_allowed=r.automatic_export_allowed,
        earliest_blocker=blockers[0].requirement + ":" + blockers[0].reasons[0]
        if blockers
        else None,
        all_blockers=[dict(requirement=d.requirement, reasons=d.reasons) for d in blockers],
        decisions=[d.model_dump(mode="json") for d in r.decisions],
        ticks=[t.model_dump(mode="json") for t in r.legend.ticks] if r.legend else [],
        seconds=time.monotonic() - started,
    )


def composed(case):
    image, grid, c, _, truth = labels_fixture(
        case["angle"],
        blocked=case["family"] != "none",
        seed=case["seed"],
        below=bool(case["seed"] % 2),
    )
    hints = DetectionHints(
        requested=(),
        rows=c.rows,
        columns=c.columns,
        heatmap_bbox=c.heatmap_bbox,
        colourbar_bbox=c.colourbar_bbox,
        colourbar_orientation=c.colourbar_orientation,
        rows_segments=c.rows_segments,
        columns_segments=c.columns_segments,
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
        labels=grid.labels.model_copy(
            update={"ocr": grid.labels.ocr.model_copy(update={"engine": "tesseract"})}
        ),
    )
    r = automate(
        image, AutomationRequest(numeric_only=False, hints=hints), experimental_export=True
    )
    expected = {t["slot"]: t["text"] for t in truth}
    return dict(
        numeric_state=r.numeric_state,
        numeric_retained=r.numeric_retention_allowed,
        labelled_complete=r.complete.status == "accepted",
        identifier_state=r.identifier_state,
        label_geometry=r.label_geometry_complete.status,
        raw_exact=sum(e["raw_text"] == expected.get(e["index"]) for e in r.label_evidence),
        total=len(expected),
        evidence=r.label_evidence,
    )


def evaluate(freeze, output):
    if json.loads(Path(freeze).read_text()) != frozen():
        raise ValueError("Source/runtime/plan changed after freeze")
    records = []
    for case in PLAN:
        for derivative in case["derivatives"]:
            image, c, truth, _ = fixture(case["seed"], case["family"], derivative)
            row = dict(
                seed=case["seed"],
                family=case["family"],
                derivative=derivative,
                image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                explicit=measured(image, c, truth),
                geometry=workflow(image, c, truth),
                full_ocr=workflow(image, c, truth, ocr=True),
                explicit_ocr=workflow(image, c, truth, explicit=True, ocr=True),
            )
            if derivative == "clean":
                label_case = dict(
                    seed=case["seed"],
                    angle=case["angle"],
                    below=bool(case["seed"] % 2),
                    blocked=case["family"] != "none",
                    font_path=FONT if case["seed"] % 2 else MONO,
                    font_size=14,
                    header=False,
                    derivatives=["clean"],
                )
                row["label_angles"] = label_attempt(label_case, "clean")
                row["composed"] = composed(case)
            records.append(row)
            print(
                case["seed"],
                case["family"],
                derivative,
                row["geometry"]["correct"],
                row["full_ocr"]["correct"],
                row["explicit_ocr"]["correct"],
                flush=True,
            )
    negatives = []
    for case in NEGATIVES:
        image, c, truth, _ = fixture(case["seed"], case["family"], challenge=case["challenge"])
        if case["challenge"] == "wrong_count":
            c = c.model_copy(update={"rows": c.rows + 1, "rows_segments": None})
        explicit = case["challenge"] in {"wrong_count", "inconsistent_ticks", "unsupported_edges"}
        row = workflow(image, c, truth, explicit=explicit, ocr=True)
        negatives.append(dict(**case, **row))
    families = {
        f: {
            d: {
                w: sum(
                    bool(r[w]["correct"] if w != "explicit" else r[w]["accurate"])
                    for r in records
                    if r["family"] == f and r["derivative"] == d
                )
                for w in ["explicit", "geometry", "full_ocr", "explicit_ocr"]
            }
            for d in ["clean", "jpeg", "blur"]
        }
        for f in FAMILIES
    }
    clean = [r for r in records if r["derivative"] == "clean"]
    gates = dict(
        explicit_clean=all(r["explicit"]["accurate"] for r in clean),
        geometry_clean_none=families["none"]["clean"]["geometry"] >= 8,
        geometry_clean_blocked=all(families[f]["clean"]["geometry"] >= 7 for f in FAMILIES[1:]),
        geometry_all_derivatives=sum(r["geometry"]["correct"] for r in records) >= 84,
        full_ocr_clean=sum(r["full_ocr"]["correct"] for r in clean) >= 28,
        explicit_ocr_clean=sum(r["explicit_ocr"]["correct"] for r in clean) >= 28,
        explicit_angle_clean=all(r["label_angles"]["explicit_correct"] for r in clean),
        auto_angle_clean=sum(r["label_angles"]["automatic_correct"] for r in clean) >= 32,
        zero_wrong_complete=not any(
            r[w]["wrong_complete"]
            for r in records
            for w in ["geometry", "full_ocr", "explicit_ocr"]
        ),
        zero_negative_complete=not any(r["numeric_state"] == "complete" for r in negatives),
        identifiers_never_automatic=not any(r["composed"]["labelled_complete"] for r in clean),
    )
    summary = dict(
        bases=40,
        correlated_attempts=120,
        negative_bases=12,
        families=families,
        gates=gates,
        counts={
            w: sum(r[w]["correct"] for r in records)
            for w in ["geometry", "full_ocr", "explicit_ocr"]
        },
        blockers={
            w: dict(Counter(r[w]["earliest_blocker"] or "accepted" for r in records))
            for w in ["geometry", "full_ocr", "explicit_ocr"]
        },
        manual_recovery="Supplied geometry/counts are measured by explicit_ocr. "
        "One-field recovery and blind six-publication coverage are not established.",
    )
    if frozen() != json.loads(Path(freeze).read_text()):
        raise ValueError("Source/runtime changed during evaluation")
    write_new(
        output,
        dict(
            freeze_sha256=hashlib.sha256(Path(freeze).read_bytes()).hexdigest(),
            records=records,
            negatives=negatives,
            summary=summary,
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--freeze", default="benchmarks/m19/freeze.json")
    parser.add_argument("--output", default="benchmarks/m19/validation.json")
    args = parser.parse_args()
    if args.action == "freeze":
        write_new(args.freeze, frozen())
        with tarfile.open(Path(args.freeze).with_suffix(".tar.gz"), "w:gz") as archive:
            for path in frozen()["source_sha256"]:
                archive.add(ROOT / path, arcname=path)
    else:
        evaluate(args.freeze, args.output)
