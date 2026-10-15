"""Exercise existing group-label, label, legend, and extraction APIs on original images."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from heatmap_decoder.extract import extract, extract_labels
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels
from heatmap_decoder.io import load_file_labels, load_raster
from heatmap_decoder.legend import inspect_legend
from heatmap_decoder.models import ExtractionConfig, GridConfig, LegendConfig

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "examples/real_world/publication_headers"


def evaluate(folder: Path, output: Path) -> dict:
    request = GroupLabelsRequest.model_validate_json((folder / "groups-request.json").read_bytes())
    source = next(
        p
        for p in sorted(folder.glob("figure-*.*"))
        if p.suffix in {".png", ".jpg"} and "reduced" not in p.name
    )
    image = load_raster(source)
    groups = infer_group_labels(image, request)
    (output / f"{folder.name}-groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    spans = []
    for track_index, name in enumerate(groups["track_names"]):
        segments = []
        last_key = None
        if groups["complete"]:
            for record in groups["records"]:
                track = record["tracks"][track_index]
                key = (track["category"], track.get("region_index"))
                if not segments or key != last_key:
                    segments.append(
                        {
                            "label": track["category"],
                            "start": record["index"],
                            "end": record["index"] + 1,
                        }
                    )
                else:
                    segments[-1]["end"] = record["index"] + 1
                last_key = key
        spans.append({"track": name, "segments": segments})
    report = {
        "image": source.name,
        "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "image_dimensions": [image.shape[1], image.shape[0]],
        "geometry_source": "explicit_reviewed_image_boxes_and_counts",
        "groups_complete": groups["complete"],
        "groups": spans,
        "sample_names": [r["sample_name"] for r in groups["records"]]
        if groups["visible_sample_names_recovered"]
        else None,
        "numeric_ground_truth": False,
    }
    if (folder / "config.json").exists():
        config = ExtractionConfig.model_validate_json((folder / "config.json").read_bytes())
        names, _ = load_file_labels(config, folder)
        result = extract(image, config, file_labels=names)
        report.update(
            shape=list(result.values.shape),
            finite_cells=int(np.isfinite(result.values).sum()),
            total_cells=result.values.size,
            median_colour_distance=float(np.median(result.colour_distance)),
            scale_source="reviewed_visual_anchors",
        )
    else:
        report.update(
            resolution_limited=True,
            article_inferred_total=request.count,
            individual_columns_resolved=False,
            header_candidates=[r["raw_text"] for r in groups["track_evidence"][0]["regions"]],
        )
    if (folder / "labels-config.json").exists():
        config = GridConfig.model_validate_json((folder / "labels-config.json").read_bytes())
        labels = extract_labels(image, config)
        (output / f"{folder.name}-labels.json").write_text(
            json.dumps(asdict(labels), indent=2, default=lambda v: v.model_dump(mode="json")) + "\n"
        )
        report["ocr"] = {}
        for axis in labels.axes:
            reference_path = folder / ("genes.txt" if axis.axis == "rows" else "samples.txt")
            reference = reference_path.read_text().splitlines()
            raw = [s.raw_text for s in axis.slots]
            report["ocr"][axis.axis] = {
                "reference": reference_path.name,
                "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                "total": len(reference),
                "exact": sum(a == b for a, b in zip(raw, reference, strict=True)),
                "wrong": sum(a is not None and a != b for a, b in zip(raw, reference, strict=True)),
                "missing": sum(a is None for a in raw),
                "accepted": sum(s.status == "accepted" for s in axis.slots),
                "require_review": getattr(config.labels, axis.axis).require_review,
            }
    if (folder / "legend-config.json").exists():
        config = LegendConfig.model_validate_json((folder / "legend-config.json").read_bytes())
        legend = inspect_legend(image, config)
        report["legend"] = {
            "status": legend.status,
            "development_eligible": legend.development_eligible,
            "anchors": [
                {"value": t.parsed_value, "position": t.position, "source": t.position_source}
                for t in legend.ticks
                if t.parsed_value is not None
            ],
        }
        (output / f"{folder.name}-legend.json").write_text(
            json.dumps(legend.model_dump(mode="json"), indent=2) + "\n"
        )
    return report


def main() -> None:
    output = ROOT / "artifacts/headers"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for request in sorted(BASE.glob("*/groups-request.json")):
        folder = request.parent
        report[folder.name] = evaluate(folder, output)
        print(
            folder.name,
            report[folder.name].get("shape"),
            report[folder.name]["groups_complete"],
            flush=True,
        )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
