"""Rebuild the README overview from real OCR of a synthetic development fixture."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydantic import TypeAdapter

from heatmap_decoder.diagnostics import recognition_diagnostic_image
from heatmap_decoder.extract import extract, extract_labels
from heatmap_decoder.geometry import axis_slots
from heatmap_decoder.legend import inspect_legend
from heatmap_decoder.models import ExtractionConfig, GridConfig, LegendConfig

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "examples/diagnostic"
image = Image.open(ROOT / "examples/m18/figure.png").convert("RGB")
original = np.asarray(image).copy()
data = json.loads((ROOT / "examples/m18/labels.json").read_text())
# Insert a physical gene-group gap without resampling any original cell pixels.
heatmap = image.crop((240, 360, 697, 480))
ImageDraw.Draw(image).rectangle((240, 360, 696, 496), fill="white")
image.paste(heatmap.crop((0, 0, heatmap.width, 60)), (240, 360))
image.paste(heatmap.crop((0, 60, heatmap.width, 120)), (240, 437))
data["heatmap_bbox"][3] = 497
data["rows_segments"] = [
    dict(start=360, end=420, count=2),
    dict(start=437, end=497, count=2),
]
data["labels"]["rows"] = dict(source="ocr", bbox=[110, 360, 235, 497], recognition="slots")
config = GridConfig.model_validate(data)
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("DejaVuSans.ttf", 16)
for slot, name in zip(axis_slots(config, "rows"), ("BRCA1", "TP53", "EGFR", "MYC"), strict=True):
    draw.text((145, slot.centre), name, anchor="lm", font=font, fill="black")
for position, value in ((360, "-3"), (450, "0"), (540, "3")):
    draw.line((54, position, 60, position), fill="black")
    draw.text((70, position), value, anchor="lm", font=font, fill="black")
legend_config = LegendConfig(
    colourbar_bbox=(40, 360, 54, 541),
    colourbar_orientation="vertical",
    legend={"search_pixels": 64},
)
pixels = np.asarray(image)
labels = extract_labels(pixels, config)
legend = inspect_legend(pixels, legend_config)
numeric_data = json.loads((ROOT / "examples/m18/numeric_config.json").read_text())
baseline = extract(original, ExtractionConfig.model_validate(numeric_data))
numeric_data.update(heatmap_bbox=data["heatmap_bbox"], rows_segments=data["rows_segments"])
numeric_config = ExtractionConfig.model_validate(numeric_data)
numeric = extract(pixels, numeric_config)
assert np.array_equal(baseline.values, numeric.values, equal_nan=True)
assert np.array_equal(baseline.flags, numeric.flags)
(OUT / "numeric_config.json").write_text(numeric_config.model_dump_json(indent=2) + "\n")
image.save(OUT / "figure.png")
(OUT / "labels.json").write_text(config.model_dump_json(indent=2) + "\n")
(OUT / "legend.json").write_text(legend_config.model_dump_json(indent=2) + "\n")
(OUT / "evidence.json").write_text(
    json.dumps(
        dict(
            labels=TypeAdapter(type(labels)).dump_python(labels, mode="json"),
            legend=legend.model_dump(mode="json"),
        ),
        indent=2,
    )
    + "\n"
)
recognition_diagnostic_image(pixels, labels, legend).save(
    ROOT / "docs/images/recognition-latest.png"
)
print([(a.axis, [s.raw_text for s in a.slots]) for a in labels.axes])
print([(t.raw_text, t.parsed_value, t.status) for t in legend.ticks])
