"""M15 per-capability evidence: truth is consulted only after deployed decisions."""

import argparse
import hashlib
import json
import runpy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from heatmap_decoder import (
    AutomationRequest,
    DetectionHints,
    ExtractionConfig,
    automate,
    detect,
    extract,
)
from heatmap_decoder.extract import extract_labels
from heatmap_decoder.legend import inspect_legend
from heatmap_decoder.models import LegendConfig

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "benchmarks/m15/fixtures.py"))


def numeric_metrics(result, truth):
    finite = np.isfinite(result.values)
    err = np.abs(result.values[finite] - np.array(truth["values"])[finite]) / 8
    return {
        "finite_fraction": float(finite.mean()),
        "flagged_fraction": float((result.flags != 0).mean()),
        "mae": float(err.mean()) if err.size else None,
        "maximum": float(err.max()) if err.size else None,
    }


def label_metrics(slots, truth):
    return {
        "raw_exact": sum(
            s.raw_text == truth["labels"][s.index] and s.raw_text is not None for s in slots
        ),
        "accepted_exact": sum(
            s.final_text == truth["labels"][s.index] and s.final_text is not None for s in slots
        ),
        "false_text_acceptance": sum(
            s.final_text is not None and s.final_text != truth["labels"][s.index] for s in slots
        ),
        "unresolved": sum(s.final_text is None for s in slots),
        "total": sum(t is not None for t in truth["labels"]),
        "slots": [asdict(s) for s in slots],
    }


def attempt(case):
    image, truth = FIXTURE["fixture"](case)
    payload = truth["config"]
    manual = extract(image, ExtractionConfig.model_validate(payload))
    base = dict(
        rows=6,
        columns=8,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=-3,
        colourbar_end_value=5,
    )
    numbers = automate(
        image,
        AutomationRequest(hints=DetectionHints(**base), numeric_only=True),
        experimental_export=True,
    )
    inferred = automate(
        image,
        AutomationRequest(
            hints=DetectionHints(
                rows=6,
                infer_columns=True,
                requested=("heatmap", "colourbar"),
                colourbar_start_value=-3,
                colourbar_end_value=5,
            ),
            numeric_only=True,
        ),
        experimental_export=True,
    )
    ocrscale = None
    if case["variant"] == "clean":
        ocrscale = automate(
            image,
            AutomationRequest(
                hints=DetectionHints(rows=6, columns=8, requested=("heatmap", "colourbar")),
                numeric_only=True,
                scale={"source": "ocr_ticks"},
            ),
            experimental_export=True,
        )
    # Selected/manual grid isolates label orientation from ROI localisation.
    local = detect(
        image,
        DetectionHints(
            heatmap_bbox=payload["heatmap_bbox"],
            rows=6,
            columns=8,
            requested=("labels_columns",),
            labels={"ocr": {"upscale_factor": 2}},
            detector={"propose_column_rotations": True, "label_search_pixels": 240},
        ),
    )
    count_text = None
    if case["variant"] == "clean":
        count_text = detect(
            image,
            DetectionHints(
                heatmap_bbox=payload["heatmap_bbox"],
                rows=6,
                infer_columns=True,
                requested=(),
                labels={"ocr": {"upscale_factor": 2}},
                detector={
                    "propose_column_rotations": True,
                    "count_label_geometry": True,
                    "label_search_pixels": 240,
                },
            ),
        )
    explicit = {
        **payload,
        "labels": {
            "columns": {
                "source": "ocr",
                "bbox": truth["label_bbox"],
                "text_rotation_degrees": case["angle"],
                "recognition": "slots",
            },
            "ocr": {"upscale_factor": 2},
        },
    }
    labels = extract_labels(image, ExtractionConfig.model_validate(explicit)).axes[1]
    legend = None
    composed = None
    if case["variant"] == "clean":
        legend = inspect_legend(
            image,
            LegendConfig(
                colourbar_bbox=payload["colourbar_bbox"], colourbar_orientation="vertical"
            ),
        )
        # Compose once through deployed API; identifiers remain outside export scope.
        composed = automate(
            image,
            AutomationRequest(
                hints=DetectionHints(
                    **{
                        **base,
                        "requested": ("heatmap", "colourbar", "labels_columns"),
                        "labels": {"ocr": {"upscale_factor": 2}},
                        "detector": {"propose_column_rotations": True, "label_search_pixels": 240},
                    }
                ),
                numeric_only=False,
            ),
            experimental_export=True,
        )

    def compact(result):
        if result is None:
            return None
        recovered = extract(image, result.diagnostic_config) if result.diagnostic_config else None
        return {
            "complete": result.complete.status == "accepted",
            "export_allowed": result.automatic_export_allowed,
            "release_reasons": result.release_reasons,
            "decisions": [d.model_dump(mode="json") for d in result.decisions],
            "numeric": numeric_metrics(recovered, truth) if recovered else None,
            "count_exact": result.diagnostic_config.columns == 8
            if result.diagnostic_config
            else None,
        }

    return {
        **case,
        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "truth": truth,
        "manual_numeric": numeric_metrics(manual, truth),
        "explicit_labels": label_metrics(labels.slots, truth),
        "orientation_trials": [e.model_dump(mode="json") for e in local.label_evidence],
        "orientation_candidates": [c.model_dump(mode="json") for c in local.candidates],
        "unique_correct_orientation": len(local.candidates) == 1
        and local.candidates[0].text_rotation_degrees == case["angle"],
        "numeric_supplied": compact(numbers),
        "inferred_columns": compact(inferred),
        "inferred_with_unassigned_text": count_text.model_dump(mode="json") if count_text else None,
        "ocr_scale": compact(ocrscale),
        "conditional_legend": legend.model_dump(mode="json") if legend else None,
        "composed": compact(composed),
    }


def sources():
    files = [
        *sorted((ROOT / "src/heatmap_decoder").glob("*.py")),
        Path(__file__),
        ROOT / "benchmarks/m15/fixtures.py",
        ROOT / "benchmarks/m15/preregister.json",
    ]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", choices=("development", "validation", "freeze"))
    args = parser.parse_args()
    path = (
        ROOT
        / "benchmarks/m15"
        / ("freeze.json" if args.split == "freeze" else args.split + ".json")
    )
    if path.exists():
        raise SystemExit("Refusing to overwrite retained evidence")
    if args.split == "freeze":
        report = {
            "scope": "freeze before recognition of validation partition",
            "sources": sources(),
        }
    else:
        if args.split == "validation":
            assert (
                json.loads((ROOT / "benchmarks/m15/freeze.json").read_text())["sources"]
                == sources()
            )
        prereg = json.loads((ROOT / "benchmarks/m15/preregister.json").read_text())
        assert (
            hashlib.sha256((ROOT / "benchmarks/m15/fixtures.py").read_bytes()).hexdigest()
            == prereg["fixture_sha256"]
        )
        records = []
        for case in prereg["cases"][args.split]:
            r = attempt(case)
            records.append(r)
            print(
                r["id"],
                "orientation",
                r["unique_correct_orientation"],
                "numeric",
                r["numeric_supplied"]["complete"],
                flush=True,
            )
        ordinary = [r for r in records if r["challenge"] is None]
        report = {
            "scope": "synthetic per-capability evidence; no publication generalisation claim",
            "split": args.split,
            "sources": sources(),
            "summary": {
                "ordinary_attempts": len(ordinary),
                "ordinary_ocr_scale_attempts": sum(r["ocr_scale"] is not None for r in ordinary),
                "ordinary_composed_attempts": sum(r["composed"] is not None for r in ordinary),
                "unique_correct_orientation": sum(
                    r["unique_correct_orientation"] for r in ordinary
                ),
                "explicit_raw_exact": sum(r["explicit_labels"]["raw_exact"] for r in ordinary),
                "explicit_accepted_exact": sum(
                    r["explicit_labels"]["accepted_exact"] for r in ordinary
                ),
                "explicit_false_text_acceptance": sum(
                    r["explicit_labels"]["false_text_acceptance"] for r in ordinary
                ),
                "numeric_complete": sum(r["numeric_supplied"]["complete"] for r in ordinary),
                "inferred_columns_complete": sum(
                    r["inferred_columns"]["complete"] for r in ordinary
                ),
                "ocr_scale_complete": sum(
                    r["ocr_scale"]["complete"] for r in ordinary if r["ocr_scale"] is not None
                ),
                "composed_exports": sum(
                    r["composed"]["export_allowed"] for r in records if r["composed"] is not None
                ),
                "inferred_exports": sum(r["inferred_columns"]["export_allowed"] for r in records),
                "ocr_scale_exports": sum(
                    r["ocr_scale"]["export_allowed"] for r in records if r["ocr_scale"] is not None
                ),
            },
            "records": records,
        }
    path.write_text(json.dumps(report, indent=2) + "\n")
