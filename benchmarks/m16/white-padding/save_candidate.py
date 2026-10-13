import hashlib
import json
import platform
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

from heatmap_decoder.diagnostic_style import DiagnosticCanvas

out = Path("benchmarks/m16/white-padding")
source = Path("examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg")
im = Image.open(source).convert("L")
box = (859, 597, 976, 617)
crop = ImageOps.autocontrast(im.crop(box))
crop = crop.resize((crop.width * 5, crop.height * 5), Image.Resampling.BICUBIC)
crop = ImageOps.expand(crop, border=40, fill=255)
d = pytesseract.image_to_data(
    crop, lang="eng", config="--oem 1 --psm 8", output_type=pytesseract.Output.DICT
)
idx = [i for i, t in enumerate(d["text"]) if t.strip()]
raw = " ".join(d["text"][i] for i in idx)
assert raw == "EsGalectin1"
confidence = min(float(d["conf"][i]) for i in idx)
result = dict(
    source_image=str(source),
    image_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    crop_bbox=box,
    crop_selection="manual inspected follow-up; not automatic localisation",
    preprocessing=[
        "grayscale",
        "autocontrast",
        "bicubic 5x",
        "white border 40 output pixels on all sides",
    ],
    engine_version=str(pytesseract.get_tesseract_version()),
    wrapper_version=pytesseract.__version__,
    python=platform.python_version(),
    engine_mode="--oem 1 --psm 8",
    raw_text=raw,
    confidence=confidence,
    status="low_confidence",
    final_text=None,
    raw_tsv_data=d,
)
(out / "recovered-candidate.json").write_text(json.dumps(result, indent=2) + "\n")
canvas = DiagnosticCanvas(crop, lines=8)
canvas.note("Manually tightened crop; contrast and 5x enlargement")
canvas.note("40px artificial white border added after enlargement")
canvas.ocr(None, confidence, "Galectin", raw, "low_confidence")
canvas.note("Trailing 1 recovered by OCR; review still required")
canvas.note("Equal-row automatic trial: 7/33 exact; baseline 9/33")
canvas.image.save("docs/images/recognition-galectin-white-padding-m16.png")
print(raw, confidence)
