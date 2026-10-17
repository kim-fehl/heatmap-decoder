import json
from pathlib import Path

m = json.loads(Path("examples/m16/manifest.json").read_text())
baseline = json.loads(Path("benchmarks/m16/numeric-signs/labels.json").read_text())
summary = {}
for name in ["native", "pad4"]:
    r = json.loads(Path(f"benchmarks/m16/rapidocr/{name}.json").read_text())["records"]
    stats = {}
    for axis in ["rows", "columns"]:
        records = [x for x in r if x["axis"] == axis]
        reference = m["displayed_" + axis]
        assert len(records) == len(reference)
        old = next(a for a in baseline["axes"] if a["axis"] == axis)["slots"]
        stats[axis] = {
            "total": len(records),
            "exact": sum(x["raw_text"] == reference[x["index"]] for x in records),
            "baseline_exact": sum(x["raw_text"] == reference[x["index"]] for x in old),
            "errors": [
                dict(
                    index=x["index"],
                    raw=x["raw_text"],
                    reference=reference[x["index"]],
                    score=x["recognizer_score"],
                )
                for x in records
                if x["raw_text"] != reference[x["index"]]
            ],
            "new_exact_indices": [
                x["index"]
                for x in records
                if x["raw_text"] == reference[x["index"]]
                and old[x["index"]]["raw_text"] != reference[x["index"]]
            ],
            "lost_exact_indices": [
                x["index"]
                for x in records
                if x["raw_text"] != reference[x["index"]]
                and old[x["index"]]["raw_text"] == reference[x["index"]]
            ],
        }
    for x, old in zip(r, baseline["ocr"]["slot_runs"], strict=True):
        assert (x["axis"], x["index"], x["crop_bbox"]) == (
            old["axis"],
            old["index"],
            old["crop_bbox"],
        )
    summary[name] = stats
Path("benchmarks/m16/rapidocr/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary["native"], indent=2))
