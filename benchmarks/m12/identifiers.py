"""Known-grid identifier OCR comparison on DEVELOPMENT only; no biological verification."""

import json
import runpy
from pathlib import Path

import numpy as np

from heatmap_decoder import extract_labels
from heatmap_decoder.models import GridConfig

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/m12/generate.py"))


def main():
    attempts = []
    for case in FIXTURE["cases"]("development"):
        if case["index"] not in (0, 3, 8):
            continue
        image, truth = FIXTURE["make_fixture"](case)
        for strategy in ("sparse", "slots"):
            labels = {}
            for axis in ("rows", "columns"):
                boxes = np.array([t["bbox"] for t in truth["tokens"][axis]])
                labels[axis] = dict(
                    source="ocr",
                    recognition=strategy,
                    bbox=[
                        int(boxes[:, 0].min()) - 5,
                        int(boxes[:, 1].min()) - 5,
                        int(boxes[:, 2].max()) + 5,
                        int(boxes[:, 3].max()) + 5,
                    ],
                )
            result = extract_labels(
                image,
                GridConfig(
                    heatmap_bbox=truth["heatmap_bbox"],
                    rows=truth["rows"],
                    columns=truth["columns"],
                    labels=labels,
                ),
            )
            axes = {}
            for axis in result.axes:
                expected = truth["genes"] if axis.axis == "rows" else truth["samples"]
                axes[axis.axis] = dict(
                    exact=sum(s.raw_text == t for s, t in zip(axis.slots, expected, strict=True)),
                    total=len(expected),
                    engine_accepted=sum(s.status == "accepted" for s in axis.slots),
                    wrong_engine_accepted=sum(
                        s.status == "accepted" and s.raw_text != t
                        for s, t in zip(axis.slots, expected, strict=True)
                    ),
                    strings=[
                        dict(expected=t, raw=s.raw_text, status=s.status, confidence=s.confidence)
                        for s, t in zip(axis.slots, expected, strict=True)
                    ],
                )
            attempts.append(dict(id=case["id"], strategy=strategy, axes=axes))
            print(case["id"], strategy, {a: v["exact"] for a, v in axes.items()}, flush=True)
    summary = {}
    for strategy in ("sparse", "slots"):
        subset = [a for a in attempts if a["strategy"] == strategy]
        summary[strategy] = {
            axis: {
                key: sum(a["axes"][axis][key] for a in subset)
                for key in ("exact", "total", "engine_accepted", "wrong_engine_accepted")
            }
            for axis in ("rows", "columns")
        }
    (ROOT / "benchmarks/m12/identifiers-development.json").write_text(
        json.dumps(
            dict(
                scope="Known-grid development OCR; identity requires human review",
                summary=summary,
                attempts=attempts,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
