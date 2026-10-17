"""Renderer-only truth audit; all preregistered images remain in evaluation."""

import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/m12/generate.py"))


def overlaps(a, b):
    return bool(min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))


if __name__ == "__main__":
    records = []
    for split in ("development", "validation", "test"):
        for case in FIXTURE["cases"](split):
            if case["degradation"] != "clean":
                continue
            image, truth = FIXTURE["make_fixture"](case)
            boxes = [t["bbox"] for t in truth["ticks"]]
            identities = [t["bbox"] for records in truth["tokens"].values() for t in records]
            touching = sum(overlaps(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
            identity_overlap = sum(overlaps(a, b) for a in boxes for b in identities)
            clipped = sum(
                a < 0 or b < 0 or r > image.shape[1] or s > image.shape[0] for a, b, r, s in boxes
            )
            records.append(
                dict(
                    base_id=case["base_id"],
                    split=split,
                    tick_box_overlaps=int(touching),
                    tick_identity_overlaps=int(identity_overlap),
                    clipped_tick_boxes=int(clipped),
                )
            )
    summary = {
        split: dict(
            bases=sum(r["split"] == split for r in records),
            tick_overlap_bases=sum(
                r["split"] == split and r["tick_box_overlaps"] > 0 for r in records
            ),
            tick_identity_overlap_bases=sum(
                r["split"] == split and r["tick_identity_overlaps"] > 0 for r in records
            ),
            clipped_tick_bases=sum(
                r["split"] == split and r["clipped_tick_boxes"] > 0 for r in records
            ),
        )
        for split in ("development", "validation", "test")
    }
    (ROOT / "benchmarks/m12/renderer-audit.json").write_text(
        json.dumps(
            dict(
                note="Truth audit; no exclusions or denominator changes",
                summary=summary,
                bases=records,
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(summary, indent=2))
