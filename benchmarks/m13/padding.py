"""Test actual crop expansion and artificial borders against M13 underscore failures."""

import json
import sys
from pathlib import Path

from PIL import ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.m13.generate import STRINGS, fixture  # noqa: E402
from heatmap_decoder.labels import (  # noqa: E402
    CropTransform,
    assign_candidates,
    group_tokens,
    transform_tokens,
)
from heatmap_decoder.models import ExtractionConfig  # noqa: E402
from heatmap_decoder.ocr import recognize  # noqa: E402


def evaluate():
    records = []
    image, payload, _ = fixture()
    config = ExtractionConfig.model_validate(payload)
    for kind in ("expand_original", "blank_border"):
        for pad in (0, 3, 5, 8):
            raw = []
            assigned = 0
            for i in range(8):
                box = (40 + i * 64, 20, 40 + (i + 1) * 64, 360)
                if kind == "expand_original":
                    box = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
                transform = CropTransform(box, 3, 90)
                crop = transform.prepare(image)
                if kind == "blank_border":
                    crop = ImageOps.expand(crop, border=pad * 3, fill=255)
                run = recognize(crop, config.labels.ocr, mode="text_line")
                raw.append(
                    " ".join(
                        t.text.strip() for t in run.tokens if t.text.strip() and t.confidence >= 0
                    )
                )
                if kind == "expand_original":
                    candidates = group_tokens(run.tokens, config, "columns", transform=transform)
                    result = assign_candidates(
                        candidates,
                        config,
                        "columns",
                        tokens=transform_tokens(run.tokens, transform),
                    )
                    assigned += int(
                        result.slots[i].bbox is not None
                        and len(result.slots[i].candidate_indices) == 1
                    )
            row = {
                "kind": kind,
                "padding_pixels": pad,
                "raw_exact": sum(a == b for a, b in zip(raw, STRINGS, strict=True)),
                "total": 8,
                "raw": raw,
            }
            if kind == "expand_original":
                row["assigned_geometry"] = assigned
            records.append(row)
            print(row, flush=True)
    return records


if __name__ == "__main__":
    (ROOT / "benchmarks/m13/padding-report.json").write_text(
        json.dumps(evaluate(), indent=2) + "\n"
    )
