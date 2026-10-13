import hashlib
import json
import time
from pathlib import Path

import numpy as np

from heatmap_decoder.automation import AutomationRequest, automate
from heatmap_decoder.extract import extract, extract_labels
from heatmap_decoder.io import load_raster, write_legend_outputs
from heatmap_decoder.legend import inspect_legend
from heatmap_decoder.models import (
    DetectionHints,
    ExtractionConfig,
    GridConfig,
    LabelsConfig,
    LegendConfig,
    OCRLabelSource,
    ScaleAnchor,
    ScaleConfig,
)

P = Path("examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg")
image = load_raster(P)
out = Path("work/m16/baseline")
out.mkdir(parents=True, exist_ok=True)
anchors = [
    ScaleAnchor(position=x, value=v) for x, v in [(39, -2), (78, -1), (117, 0), (156, 1), (195, 2)]
]
c = ExtractionConfig(
    heatmap_bbox=(335, 189, 854, 861),
    rows=33,
    columns=8,
    colourbar_bbox=(18, 39, 217, 103),
    colourbar_orientation="horizontal",
    scale=ScaleConfig(source="manual_ticks", anchors=anchors),
)
Path("examples/m16/config-linear-baseline.json").write_text(c.model_dump_json(indent=2) + "\n")
records = {}
t = time.monotonic()
r = extract(image, c)
records["manual_numeric"] = {
    "seconds": time.monotonic() - t,
    "finite": int(np.isfinite(r.values).sum()),
    "flagged": int((r.flags != 0).sum()),
}
np.save(out / "values.npy", r.values)
t = time.monotonic()
legend = inspect_legend(
    image, LegendConfig(colourbar_bbox=c.colourbar_bbox, colourbar_orientation="horizontal")
)
records["legend"] = {"seconds": time.monotonic() - t, "result": legend.model_dump(mode="json")}
write_legend_outputs(
    image, legend, out / "legend", hashlib.sha256(P.read_bytes()).hexdigest(), P.name
)
g = GridConfig(
    heatmap_bbox=c.heatmap_bbox,
    rows=33,
    columns=8,
    labels=LabelsConfig(
        rows=OCRLabelSource(source="ocr", bbox=(858, 189, 980, 861), recognition="slots"),
        columns=OCRLabelSource(source="ocr", bbox=(335, 868, 854, 904), recognition="slots"),
    ),
)
Path("examples/m16/labels.json").write_text(g.model_dump_json(indent=2) + "\n")
t = time.monotonic()
labels = extract_labels(image, g)
records["labels"] = {
    "seconds": time.monotonic() - t,
    "unresolved": labels.unresolved_count,
    "slots": {
        a.axis: [
            {"raw": s.raw_text, "status": s.status, "confidence": s.confidence} for s in a.slots
        ]
        for a in labels.axes
    },
}
# Save through the CLI later: label writer has the filesystem adapter signature.
for mode in ["auto_roi", "composed"]:
    hints = DetectionHints(
        rows=33,
        columns=8,
        requested=("heatmap", "colourbar")
        if mode == "auto_roi"
        else ("heatmap", "colourbar", "labels_rows", "labels_columns"),
    )
    t = time.monotonic()
    a = automate(
        image,
        AutomationRequest(hints=hints, numeric_only=mode == "auto_roi", scale=c.scale),
        experimental_export=True,
    )
    records[mode] = {"seconds": time.monotonic() - t, "result": a.model_dump(mode="json")}
(out / "report.json").write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            k: (
                {i: v for i, v in x.items() if i != "result"}
                if k not in ["auto_roi", "composed"]
                else {
                    "seconds": x["seconds"],
                    "complete": x["result"]["complete"],
                    "export": x["result"]["automatic_export_allowed"],
                }
            )
            for k, x in records.items()
        },
        indent=2,
    )
)
