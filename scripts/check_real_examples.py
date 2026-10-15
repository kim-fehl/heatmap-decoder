"""Run the actual decoder on the original examples; emit reviewable artifacts."""

import json
from pathlib import Path

from heatmap_decoder.publication import decode

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    examples = [
        ("figure-1c", "plos-ppat-1012368", "png", None),
        ("figure-7c", "plos-pone-0244065", "png", None),
        ("figure-4a", "iscience-108897", "jpg", 1237),
        ("figure-4a", "plos-pone-0155215", "png", None),
    ]
    report = {}
    for stem, folder, extension, total in examples:
        result = decode(
            ROOT / "examples/real_world/publication_headers" / folder / f"{stem}.{extension}",
            expected_columns=total,
        )
        artifact = stem + "-0155215" if folder == "plos-pone-0155215" else stem
        (output / f"{artifact}.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        report[artifact] = {
            "rows": result["n_rows"],
            "image_derived_columns": result["n_columns"],
            "expected_columns": result.get("expected_n_columns"),
            "status": result.get("status", "decoded"),
            "groups": [
                [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in t["segments"]]
                for t in result["column_annotations"]
            ],
            "colorbar_values": [a["value"] for a in result["colorbar"]["anchors"]],
            "anchor_sources": sorted({a["anchor_source"] for a in result["colorbar"]["anchors"]}),
            "row_labels_read": sum(label is not None for label in result["row_labels"]),
            "value_quality": result.get("value_quality"),
        }
        print(artifact, report[artifact]["status"], result["n_rows"], result["n_columns"])
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
