"""Run reproducible extraction/OCR comparisons for the attached published crops."""

import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from heatmap_decoder.extract import extract, extract_labels
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels
from heatmap_decoder.io import load_file_labels, load_raster
from heatmap_decoder.models import CellFlag, ExtractionConfig

ROOT = Path(__file__).resolve().parent


def matrix_tsv(path, genes, columns, values):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["gene", *columns])
        for gene, row in zip(genes, values, strict=True):
            writer.writerow(
                [gene, *("NaN" if not np.isfinite(v) else format(v, ".17g") for v in row)]
            )


def main():
    report = {
        "packages": {
            name: version(name)
            for name in [
                "heatmap-decoder",
                "numpy",
                "Pillow",
                "scikit-image",
                "rapidocr-onnxruntime",
            ]
        },
        "panels": {},
    }
    for folder in ["elife-78496-fig5e", "elife-97830-fig4e", "elife-17290-fig2"]:
        path = ROOT / folder
        image = load_raster(path / "figure.jpg")
        genes = (path / "rows.txt").read_text().splitlines()
        columns = (path / "columns.txt").read_text().splitlines()
        truth = None
        if (path / "truth.tsv").exists():
            with (path / "truth.tsv").open(newline="") as stream:
                table = list(csv.reader(stream, delimiter="\t"))
            assert table[0][1:] == columns and [r[0] for r in table[1:]] == genes
            truth = np.array([row[1:] for row in table[1:]], dtype=float)
        panel = {
            "shape": [len(genes), len(columns)],
            "independent_truth_available": truth is not None,
            "image_sha256": hashlib.sha256((path / "figure.jpg").read_bytes()).hexdigest(),
            "numeric": {},
        }
        for config_name in ["config.json", "config-linear.json"]:
            config = ExtractionConfig.model_validate_json((path / config_name).read_bytes())
            supplied, _ = load_file_labels(config, path)
            result = extract(image, config, file_labels=supplied)
            valid = np.isfinite(result.values)
            numeric = {
                "finite_cells": int(valid.sum()),
                "total_cells": int(valid.size),
                "flag_counts": {
                    flag.name: int(np.sum((result.flags & flag) != 0))
                    for flag in CellFlag
                    if flag != CellFlag.OK
                },
                "calibration": "manual endpoint fit (0–6); bar positions estimated"
                if folder == "elife-17290-fig2"
                else "reviewed ticks only; no extrapolation"
                if config_name == "config.json"
                else "affine fit to visible ticks; extrapolates to bar edges",
            }
            if truth is not None:
                differences = truth[valid] - result.values[valid]
                numeric.update(
                    mae=float(np.mean(np.abs(differences))),
                    rmse=float(np.sqrt(np.mean(differences**2))),
                    max_absolute_error=float(np.max(np.abs(differences))),
                    correlation=float(np.corrcoef(truth[valid], result.values[valid])[0, 1]),
                )
            panel["numeric"][config_name] = numeric
            if config_name == "config.json":
                # Clearly marked estimates; these never become the source truth.
                matrix_tsv(path / "decoded_estimates.tsv", genes, columns, result.values)
        config = ExtractionConfig.model_validate_json((path / "labels-ocr.json").read_bytes())
        labels = extract_labels(image, config)
        slots = labels.axes[0].slots
        mismatches = [
            {
                "index": i,
                "expected": gene,
                "raw": slot.raw_text,
                "final": slot.final_text,
                "status": slot.status,
            }
            for i, (gene, slot) in enumerate(zip(genes, slots, strict=True))
            if gene != slot.final_text
        ]
        panel["row_ocr"] = {
            "exact_correct": len(genes) - len(mismatches),
            "raw_exact_correct": sum(
                gene == slot.raw_text for gene, slot in zip(genes, slots, strict=True)
            ),
            "accepted_wrong": sum(
                slot.final_text is not None and gene != slot.final_text
                for gene, slot in zip(genes, slots, strict=True)
            ),
            "total": len(genes),
            "unresolved": sum(slot.final_text is None for slot in slots),
            "mismatches": mismatches,
        }
        if folder == "elife-17290-fig2":
            variants = {}
            for factor in [2, 3]:
                request = json.loads((path / "labels-ocr.json").read_text())
                request["labels"]["ocr"]["upscale_factor"] = factor
                variant = extract_labels(image, ExtractionConfig.model_validate(request))
                observed = variant.axes[0].slots
                variants[str(factor)] = {
                    "exact_correct": sum(
                        g == s.final_text for g, s in zip(genes, observed, strict=True)
                    ),
                    "raw_exact_correct": sum(
                        g == s.raw_text for g, s in zip(genes, observed, strict=True)
                    ),
                    "unresolved": sum(s.final_text is None for s in observed),
                    "accepted_wrong": sum(
                        s.final_text is not None and g != s.final_text
                        for g, s in zip(genes, observed, strict=True)
                    ),
                }
            panel["row_ocr_upscale_comparison"] = variants
        if folder == "elife-97830-fig4e":
            panel["source_data"] = json.loads((path / "source_manifest.json").read_text())
        report["panels"][folder] = panel
    for folder in ["elife-78496-fig5-supp3a", "elife-97830-fig4e"]:
        path = ROOT / folder
        request = GroupLabelsRequest.model_validate_json(
            (path / "groups-request.json").read_bytes()
        )
        groups = infer_group_labels(load_raster(path / "figure.jpg"), request)
        assert groups["complete"]
        assert [r["generated_name"] for r in groups["records"]] == (
            path / "columns.txt"
        ).read_text().splitlines()
        panel = report["panels"].setdefault(folder, {"independent_truth_available": False})
        panel["group_labels"] = {
            "complete": True,
            "count": len(groups["records"]),
            "names": [r["generated_name"] for r in groups["records"]],
            "original_sample_ids_recovered": False,
        }
    (ROOT / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
