"""Evaluation-only retrieval and audited publication trials; no biological truth claims."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from heatmap_decoder import (
    AutomationRequest,
    DetectionHints,
    ExtractionConfig,
    automate,
    detect,
    extract,
    load_raster,
)
from heatmap_decoder.extract import extract_labels
from heatmap_decoder.io import load_file_labels

HERE = Path(__file__).resolve().parent


def retrieve(directory):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "publications.json").read_text())
    records = []
    for source in manifest["sources"]:
        record = dict(source)
        if source["image_url"] is None:
            record["retrieval"] = "pending panel selection"
        else:
            try:
                with urllib.request.urlopen(source["image_url"], timeout=30) as response:
                    data = response.read(32 * 1024 * 1024 + 1)
                if len(data) > 32 * 1024 * 1024:
                    raise ValueError("image exceeds 32 MiB retrieval cap")
                path = directory / (
                    source["doi"].split("/")[-1] + "-" + source["figure_id"] + ".png"
                )
                if path.exists():
                    raise ValueError("refusing to overwrite asset")
                path.write_bytes(data)
                with Image.open(path) as image:
                    record["source_dimensions"] = list(image.size)
                record.update(
                    retrieval="downloaded for evaluation; panel reuse credits still require audit",
                    local_path=str(path.resolve()),
                    image_sha256=hashlib.sha256(data).hexdigest(),
                )
            except (OSError, ValueError) as error:
                record["retrieval"] = "failed: " + str(error)
        records.append(record)
    output = directory / "retrieval.json"
    if output.exists():
        raise ValueError("refusing to overwrite retrieval evidence")
    output.write_text(
        json.dumps({"scope": "evaluation-only full figures, no crop", "sources": records}, indent=2)
        + "\n"
    )


def evaluate(annotation_path):
    manifest = json.loads(annotation_path.read_text())
    records = []
    for annotation in manifest["trials"]:
        # An unavailable or unsupported scale cannot silently become a linear scale.
        if not annotation.get("second_pass_audited"):
            raise ValueError("publication annotations require a recorded second-pass audit")
        if not annotation.get("publication_url") or not annotation.get("licence"):
            raise ValueError("publication source and reuse licence must be recorded")
        path = Path(annotation["image_path"])
        if not path.is_absolute():
            path = annotation_path.parent / path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == annotation["image_sha256"]
        image = load_raster(path)
        if annotation.get("scale_model") != "linear":
            records.append(
                {
                    "id": annotation["id"],
                    "status": "review_required",
                    "reason": "unsupported_or_unverifiable_scale_model",
                    "annotation": annotation,
                }
            )
            continue
        config = ExtractionConfig.model_validate(annotation["config"])
        numeric_config = config.model_copy(update={"labels": None})
        manual = extract(image, numeric_config)
        # Full image is passed to the runtime; manual ROIs are not injected.
        basic = dict(
            rows=config.rows,
            columns=config.columns,
            requested=("heatmap", "colourbar"),
            colourbar_start_value=config.colourbar_start_value,
            colourbar_end_value=config.colourbar_end_value,
        )
        if config.scale is not None and config.scale.source != "manual_endpoints":
            basic["colourbar_start_value"] = basic["colourbar_end_value"] = None
        supplied = automate(
            image,
            AutomationRequest(hints=DetectionHints(**basic), numeric_only=True, scale=config.scale),
            experimental_export=True,
        )
        inferred = automate(
            image,
            AutomationRequest(
                hints=DetectionHints(**{**basic, "columns": None, "infer_columns": True}),
                numeric_only=True,
                scale=config.scale,
            ),
            experimental_export=True,
        )
        ocrscale = automate(
            image,
            AutomationRequest(
                hints=DetectionHints(
                    **{**basic, "colourbar_start_value": None, "colourbar_end_value": None}
                ),
                numeric_only=True,
                scale={"source": "ocr_ticks"},
            ),
            experimental_export=True,
        )
        labels = None
        orientation = None
        if config.labels:
            file_labels, _ = load_file_labels(config, annotation_path.parent)
            labels = extract_labels(image, config, file_labels=file_labels)
            if config.labels.columns is not None:
                orientation = detect(
                    image,
                    DetectionHints(
                        heatmap_bbox=config.heatmap_bbox,
                        rows=config.rows,
                        columns=config.columns,
                        requested=("labels_columns",),
                        labels={"ocr": config.labels.ocr.model_dump()},
                        detector={"propose_column_rotations": True},
                    ),
                )
        recovered = (
            extract(image, supplied.diagnostic_config) if supplied.diagnostic_config else None
        )
        agreement = None
        if recovered is not None and recovered.values.shape == manual.values.shape:
            finite = np.isfinite(recovered.values) & np.isfinite(manual.values)
            agreement = (
                float(np.mean(np.abs(recovered.values[finite] - manual.values[finite])))
                if finite.any()
                else None
            )
        records.append(
            {
                "id": annotation["id"],
                "annotation": annotation,
                "source_dimensions": [image.shape[1], image.shape[0]],
                "manual_finite_fraction": float(np.isfinite(manual.values).mean()),
                "manual_flagged_fraction": float((manual.flags != 0).mean()),
                "numeric_truth": "unavailable unless separately documented source data is aligned",
                "manual_auto_agreement_mae": agreement,
                "agreement_is_biological_accuracy": False,
                "supplied": supplied.model_dump(mode="json"),
                "inferred": inferred.model_dump(mode="json"),
                "ocr_scale": ocrscale.model_dump(mode="json"),
                "orientation_trials": orientation.model_dump(mode="json") if orientation else None,
                "labels": [
                    [
                        {
                            "index": s.index,
                            "raw_text": s.raw_text,
                            "status": s.status,
                            "bbox": s.bbox,
                        }
                        for s in axis.slots
                    ]
                    for axis in labels.axes
                ]
                if labels
                else None,
            }
        )
    return {
        "scope": "formative full-image workflow agreement; no biological accuracy claim",
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieve", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.retrieve:
        retrieve(args.retrieve)
    elif args.annotations and args.out:
        if args.out.exists():
            raise SystemExit("Refusing to overwrite evidence")
        args.out.write_text(json.dumps(evaluate(args.annotations), indent=2) + "\n")
    else:
        parser.error("use --retrieve DIRECTORY or --annotations FILE --out REPORT")
