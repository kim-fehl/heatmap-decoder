"""Render the observed native-crop candidate; enlargement is display-only."""

import json
from pathlib import Path

from PIL import Image

from heatmap_decoder.diagnostic_style import DiagnosticCanvas

run = json.loads(Path("benchmarks/m16/rapidocr/native.json").read_text())
slot = next(r for r in run["records"] if r["axis"] == "rows" and r["index"] == 20)
crop = Image.open(run["input"]["filename"]).crop(slot["crop_bbox"])
crop = crop.resize((crop.width * 5, crop.height * 5), Image.Resampling.NEAREST)
canvas = DiagnosticCanvas(crop, lines=8)
canvas.note("RapidOCR: same equal-row source crop, no manual tightening")
canvas.note("OCR input: native-size grayscale, no added white border")
canvas.note("Display enlarged 5x; OCR did not use this enlargement")
canvas.note(f"Raw: {slot['raw_text']}; recognizer score {slot['recognizer_score']:.3f}")
canvas.note("26/33 exact genes; 8/8 exact samples on this image")
canvas.note("Scores are engine-specific; candidates still require review")
canvas.image.save("docs/images/recognition-galectin-rapidocr-m16.png")
