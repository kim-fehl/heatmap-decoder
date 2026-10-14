"""Paired inspected-development comparisons of Chinese and English recognizers."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from heatmap_decoder.extract import extract_labels
from heatmap_decoder.models import GridConfig

ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT / "examples/m16/manifest.json").read_text())
cases = [
    (
        "diagnostic",
        "examples/diagnostic/figure.png",
        "examples/diagnostic/labels.json",
        {
            "rows": ["BRCA1", "TP53", "EGFR", "MYC"],
            "columns": [f"Sample_{i} long 51" for i in range(1, 5)],
        },
    ),
    (
        "publication",
        manifest["image"],
        "examples/m16/labels-default.json",
        {"rows": manifest["displayed_rows"], "columns": manifest["displayed_columns"]},
    ),
]
report = {"scope": "Paired inspected development figures; not held-out validation", "cases": []}
for name, filename, configfile, truth in cases:
    pixels = np.asarray(Image.open(ROOT / filename).convert("RGB"))
    case = {"name": name, "input": filename, "runs": []}
    for language in ("chinese", "eng"):
        data = json.loads((ROOT / configfile).read_text())
        data["labels"]["ocr"]["language"] = language
        data["labels"]["character_policy"] = "unrestricted"
        result = extract_labels(pixels, GridConfig.model_validate(data))
        axes = []
        for axis in result.axes:
            expected = truth[axis.axis]
            records = [
                dict(
                    index=s.index,
                    raw=s.raw_text,
                    expected=expected[s.index],
                    score=s.confidence,
                    status=s.status,
                    issues=s.issues,
                )
                for s in axis.slots
            ]
            exact = sum(s.raw_text == expected[s.index] for s in axis.slots)
            axes.append(dict(axis=axis.axis, exact=exact, total=len(axis.slots), records=records))
        runs = [run for _, run in result.ocr_runs]
        runs += [run for _, _, _, run in result.slot_ocr_runs]
        runs += [r["run"] for r in result.oriented_ocr_records]
        # Line records already contain serializable run metadata.
        models = sorted(
            {
                (n, h)
                for run in runs
                for n, h in (run["model_sha256"] if isinstance(run, dict) else run.model_sha256)
            }
        )
        case["runs"].append(dict(language=language, axes=axes, models=models))
        print(name, language, [(a["axis"], a["exact"], a["total"]) for a in axes], flush=True)
    report["cases"].append(case)
(ROOT / "benchmarks/ocr-review/language-comparison.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
