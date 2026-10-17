"""Development-only RapidOCR tick-crop probe; does not change production policy."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from heatmap_decoder.legend import inspect_legend, parse_numeric_tick
from heatmap_decoder.models import ExtractionConfig, LegendConfig, OCRConfig
from heatmap_decoder.rapid_ocr import recognize_rapid

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg"
image = Image.open(source).convert("RGB")
config = ExtractionConfig.model_validate_json(
    (ROOT / "examples/m16/config-numeric.json").read_text()
)
legend = inspect_legend(
    np.asarray(image),
    LegendConfig(
        colourbar_bbox=config.colourbar_bbox,
        colourbar_orientation=config.colourbar_orientation,
        legend=config.scale.legend.model_copy(
            update={"ocr": OCRConfig(engine="tesseract"), "white_padding_pixels": 2}
        ),
    ),
)
records = []
for tick in legend.ticks:
    if tick.position_source != "stroke" or tick.ocr_bbox is None:
        continue
    for factor in (1, 2, 3):
        for padding in (0, 2):
            crop = ImageOps.expand(image.crop(tick.ocr_bbox).convert("L"), padding, fill=255)
            crop = crop.resize(
                (crop.width * factor, crop.height * factor), Image.Resampling.BICUBIC
            )
            run = recognize_rapid(
                crop,
                OCRConfig(engine="rapidocr", language="chinese"),
                mode="text_line",
                timeout_seconds=30,
            )
            raw = " ".join(t.text for t in run.tokens)
            records.append(
                dict(
                    position=tick.position,
                    box=tick.ocr_bbox,
                    factor=factor,
                    padding=padding,
                    raw=raw,
                    parsed=parse_numeric_tick(raw).value,
                    scores=[t.confidence for t in run.tokens],
                    tesseract_raw=tick.raw_text,
                    tesseract_parsed=tick.parsed_value,
                )
            )
report = dict(
    scope="Inspected publication, development-only; no held-out qualification",
    input=str(source.relative_to(ROOT)),
    records=records,
    engine=run.engine_version,
    wrapper=run.wrapper_version,
    models=run.model_sha256,
)
(ROOT / "benchmarks/ocr-review/rapid-ticks.json").write_text(json.dumps(report, indent=2) + "\n")
for factor in (1, 2, 3):
    for padding in (0, 2):
        print(
            factor,
            padding,
            [r["raw"] for r in records if r["factor"] == factor and r["padding"] == padding],
        )
