"""End-to-end English RapidOCR legend probe, with no Tesseract-derived fields."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.m19.fixtures import fixture  # noqa: E402
from heatmap_decoder.legend import inspect_legend  # noqa: E402
from heatmap_decoder.models import ExtractionConfig, LegendConfig, LegendOptions, OCRConfig  # noqa: E402

aligned = "--aligned" in sys.argv
records = []
for family, seed in zip(
    ("none", "horizontal", "vertical", "both"), range(291001, 291005), strict=True
):
    for derivative in ("clean", "jpeg", "blur"):
        image, config, _, truth = fixture(seed=seed, family=family, derivative=derivative)
        settings = LegendOptions(
            ocr=OCRConfig(engine="rapidocr", upscale_factor=1),
            white_padding_pixels=2,
            align_tick_fields=aligned,
            crop_padding_pixels=2,
        )
        result = inspect_legend(
            image,
            LegendConfig(
                colourbar_bbox=config.colourbar_bbox,
                colourbar_orientation=config.colourbar_orientation,
                legend=settings,
            ),
        )
        physical = [t for t in result.ticks if t.position_source == "stroke"]
        expected = [t["value"] for t in truth]
        records.append(
            dict(
                family=family,
                seed=seed,
                derivative=derivative,
                expected=expected,
                raw=[t.raw_text for t in physical],
                parsed=[t.parsed_value for t in physical],
                exact=[t.parsed_value for t in physical] == expected,
                result=result.model_dump(mode="json"),
            )
        )
        print(
            family, derivative, records[-1]["raw"], records[-1]["exact"], result.status, flush=True
        )
config = ExtractionConfig.model_validate_json(
    (ROOT / "examples/m16/config-numeric.json").read_text()
)
settings = config.scale.legend.model_copy(
    update={
        "ocr": OCRConfig(engine="rapidocr", upscale_factor=1),
        "white_padding_pixels": 2,
        "retry_white_padding_pixels": (),
        "retry_unconstrained_numeric": False,
    }
)
image = np.asarray(
    Image.open(ROOT / "examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg").convert("RGB")
)
result = inspect_legend(
    image,
    LegendConfig(
        colourbar_bbox=config.colourbar_bbox,
        colourbar_orientation=config.colourbar_orientation,
        legend=settings,
    ),
)
physical = [t for t in result.ticks if t.position_source == "stroke"]
print(
    "publication",
    [(t.raw_text, t.parsed_value, t.status) for t in physical],
    result.status,
    flush=True,
)
report = dict(
    scope="Inspected development, production RapidOCR numeric pipeline; no Tesseract calls",
    records=records,
    publication=result.model_dump(mode="json"),
)
(
    ROOT
    / (
        "benchmarks/ocr-review/rapid-legend-current-aligned.json"
        if aligned
        else "benchmarks/ocr-review/rapid-legend-current.json"
    )
).write_text(json.dumps(report, indent=2) + "\n")
