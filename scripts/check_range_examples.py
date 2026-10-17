"""Reproduce image-derived evidence for the TPM and independent gene-category examples."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from heatmap_decoder.detection import count_proposals
from heatmap_decoder.extract import extract
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels
from heatmap_decoder.io import load_file_labels, load_raster
from heatmap_decoder.lattice import observe_axis
from heatmap_decoder.models import DetectionHints, ExtractionConfig
from heatmap_decoder.range_scale import RangeExtractionRequest, extract_ranges

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/real_world"


def json_file(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def group_check(folder, filename, output):
    request = GroupLabelsRequest.model_validate_json((folder / filename).read_bytes())
    labels = {} if request.grid is None else load_file_labels(request.grid, folder)[0]
    result = infer_group_labels(load_raster(folder / "figure.jpg"), request, file_labels=labels)
    json_file(output / filename.replace("request", "results"), result)
    target = folder / ("row_groups.tsv" if request.axis == "rows" else "column_groups.tsv")
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["index", "entity_kind", "visible_name", "category", "candidates", "status"]
        )
        for r in result["records"]:
            track = r["tracks"][0]
            writer.writerow(
                [
                    r["index"],
                    result["entity_kind"],
                    r["entity_name"],
                    track["category"],
                    " | ".join(track.get("candidate_categories", [])),
                    track["status"],
                ]
            )
    return {
        "axis": result["axis"],
        "entity_kind": result["entity_kind"],
        "complete": result["complete"],
        "matched": sum(r["tracks"][0]["status"] == "matched" for r in result["records"]),
        "total": len(result["records"]),
        "visible_sample_names_recovered": result["visible_sample_names_recovered"],
        "original_sample_ids_recovered": result["original_sample_ids_recovered"],
    }


def count_evidence(image, box):
    observed = observe_axis(image, box, "columns")
    positions = np.array(observed.transitions)
    candidates = []
    for count in range(2, min(512, (box[2] - box[0]) // 3) + 1):
        pitch = (box[2] - box[0]) / count
        residual = np.abs((positions - box[0]) / pitch - np.rint((positions - box[0]) / pitch))
        candidates.append(
            {
                "count": count,
                "pitch": pitch,
                "mean_phase_residual_in_cells": float(residual.mean()),
                "fraction_within_0.2_cells": float(np.mean(residual <= 0.2)),
            }
        )
    candidates.sort(key=lambda c: (c["mean_phase_residual_in_cells"], c["count"]))
    hints = DetectionHints(rows=5, infer_columns=True, detector={"max_count": 512})
    return {
        "method": "rank regular-grid hypotheses by mean distance of observed transitions to edges",
        "observations": list(observed.transitions),
        "ranked_candidates": candidates[:12],
        "count_is_image_derived": True,
        "sample_identities_recovered": False,
        "production_detector_proposals": [
            p.count for p in count_proposals(image, box, "columns", hints, "reviewed-heatmap")
        ],
        "automatic_count_accepted": False,
        "limitation": "Sparse transitions and harmonics do not independently prove sample count.",
    }


def main():
    output = ROOT / "artifacts/range-examples"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ["elife-87930-response15b", "lsa-e202302332-fig7a"]:
        folder = BASE / name
        target = output / name
        target.mkdir(exist_ok=True)
        image = load_raster(folder / "figure.jpg")
        entry = {
            "image_sha256": hashlib.sha256((folder / "figure.jpg").read_bytes()).hexdigest(),
            "numeric_ground_truth": False,
            "columns": group_check(folder, "groups-request.json", target),
        }
        if (folder / "row-groups-request.json").exists():
            entry["rows"] = group_check(folder, "row-groups-request.json", target)
            config = ExtractionConfig.model_validate_json((folder / "config.json").read_bytes())
            result = extract(image, config, file_labels=load_file_labels(config, folder)[0])
            with (folder / "decoded_estimates.tsv").open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["gene", *[f"position_{i + 1:02d}" for i in range(config.columns)]])
                for gene, row in zip((folder / "rows.txt").read_text().splitlines(), result.values):
                    writer.writerow([gene, *[float(v) if np.isfinite(v) else "NaN" for v in row]])
            entry["numeric"] = {
                "shape": list(result.values.shape),
                "finite_cells": int(np.isfinite(result.values).sum()),
                "total_cells": result.values.size,
                "quantity": config.scale.quantity,
            }
        else:
            request = RangeExtractionRequest.model_validate_json(
                (folder / "ranges-request.json").read_bytes()
            )
            result = extract_ranges(image, request)
            json_file(target / "ranges.json", result)
            with (folder / "decoded_ranges.tsv").open("w", newline="") as stream:
                writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
                writer.writerow(["row", "column", "range", "lower", "upper", "status"])
                for r in result["records"]:
                    band = r["range"] or {}
                    writer.writerow(
                        [
                            r["row"],
                            r["column"],
                            band.get("label"),
                            band.get("lower"),
                            band.get("upper"),
                            r["status"],
                        ]
                    )
            entry["ranges"] = {
                "shape": result["shape"],
                "quantity": result["quantity"],
                "statuses": dict(Counter(r["status"] for r in result["records"])),
                "point_values_recovered": False,
            }
            json_file(
                folder / "count_evidence.json",
                count_evidence(image, request.position_config.heatmap_bbox),
            )
        report[name] = entry
    json_file(BASE / "range_examples_evaluation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
