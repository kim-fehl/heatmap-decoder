"""Conditional explicit-angle OCR evidence; synthetic fixtures, never hold-out validation."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.m13.generate import CLEAN_STRINGS, FONTS, STRINGS, fixture  # noqa: E402
from heatmap_decoder.extract import extract_labels  # noqa: E402
from heatmap_decoder.models import ExtractionConfig  # noqa: E402


def evaluate():
    records = []
    for family in ("supported", "punctuation"):
        fonts = ("sans",) if family == "supported" else tuple(FONTS)
        variants = ("clean",) if family == "supported" else ("clean", "jpeg", "blur")
        for angle in (90, -90):
            for placement in ("above", "below"):
                for font in fonts:
                    for variant in variants:
                        modes = ("slots", "sparse") if family == "supported" else ("slots",)
                        for mode in modes:
                            strings = CLEAN_STRINGS if family == "supported" else STRINGS
                            image, payload, _ = fixture(angle, placement, font, variant, strings)
                            if variant == "jpeg":
                                from io import BytesIO

                                data = BytesIO()
                                image.save(data, format="JPEG", quality=85)
                                data.seek(0)
                                image = Image.open(data).convert("RGB")
                            payload["labels"]["columns"]["recognition"] = mode
                            config = ExtractionConfig.model_validate(payload)
                            start = time.monotonic()
                            result = extract_labels(np.asarray(image), config)
                            columns = result.axes[1]
                            # Assignment checks geometry against renderer's original column bands.
                            # It does not certify spelling or a biological identity.
                            assignment = sum(
                                s.bbox is not None
                                and 40 + s.index * 64
                                <= s.bbox[0]
                                < s.bbox[2]
                                <= 40 + (s.index + 1) * 64
                                and len(s.candidate_indices) == 1
                                for s in columns.slots
                            )
                            record = {
                                "family": family,
                                "angle": angle,
                                "placement": placement,
                                "font": font,
                                "variant": variant,
                                "recognition": mode,
                                "total": len(strings),
                                "raw_exact": sum(
                                    s.raw_text == strings[s.index] for s in columns.slots
                                ),
                                "accepted_exact": sum(
                                    s.final_text == strings[s.index] for s in columns.slots
                                ),
                                "false_text_acceptance": sum(
                                    s.final_text is not None and s.final_text != strings[s.index]
                                    for s in columns.slots
                                ),
                                "assigned_geometry": assignment,
                                "unresolved": sum(s.final_text is None for s in columns.slots),
                                "unmatched_candidates": len(columns.unmatched_candidate_indices),
                                "slots": [
                                    {
                                        "index": s.index,
                                        "expected": strings[s.index],
                                        "raw": s.raw_text,
                                        "final": s.final_text,
                                        "status": s.status,
                                        "bbox": s.bbox,
                                    }
                                    for s in columns.slots
                                ],
                                "seconds": round(time.monotonic() - start, 3),
                            }
                            print({k: v for k, v in record.items() if k != "slots"}, flush=True)
                            records.append(record)
    import pytesseract

    return {
        "scope": "conditional synthetic development evidence; explicit ROI/counts/angle",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
            "tesseract": str(pytesseract.get_tesseract_version()),
            "pytesseract": pytesseract.__version__,
            "fonts_sha256": {
                k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in FONTS.items()
            },
        },
        "generator_version": 1,
        "records": records,
        "summary": {
            family: {
                key: sum(r[key] for r in records if r["family"] == family)
                for key in (
                    "total",
                    "raw_exact",
                    "accepted_exact",
                    "false_text_acceptance",
                    "assigned_geometry",
                    "unresolved",
                    "unmatched_candidates",
                )
            }
            for family in ("supported", "punctuation")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks/m13/report.json")
    args = parser.parse_args()
    report = evaluate()
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(report["summary"])
