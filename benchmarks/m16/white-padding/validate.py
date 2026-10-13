import json
import math
from pathlib import Path

from PIL import Image, ImageOps

from heatmap_decoder.models import OCRConfig
from heatmap_decoder.ocr import recognize

m = json.loads(Path("examples/m16/manifest.json").read_text())
im = Image.open(m["image"])
records = []


def read(box, pad, scale=5):
    c = ImageOps.expand(im.crop(box).convert("L"), border=pad, fill=255)
    c = c.resize((c.width * scale, c.height * scale), Image.Resampling.BICUBIC)
    run = recognize(c, OCRConfig(upscale_factor=scale), mode="text_line")
    t = [t for t in run.tokens if t.text.strip()]
    return dict(
        box=box,
        pad=pad,
        scale=scale,
        raw=" ".join(x.text for x in t),
        confidence=min((x.confidence for x in t), default=None),
        engine_mode=run.mode,
    )


for pad in [0, 1, 2, 4, 8, 12, 16]:
    r = read((859, 597, 976, 617), pad)
    r["trial"] = "manual-tight-crop-ablation"
    records.append(r)
for row in range(33):
    top, bottom = 189 + row * 672 / 33, 189 + (row + 1) * 672 / 33
    box = (859, math.floor(top) + 1, 983, math.ceil(bottom) - 1)
    r = read(box, 8)
    r.update(trial="uniform-slot-inset", row=row)
    records.append(r)
# Reference transcriptions are used only after OCR, for scoring.
for r in records:
    if "row" in r:
        r["exact"] = r["raw"] == m["displayed_rows"][r["row"]]
Path("work/m16/white-border-validation.json").write_text(json.dumps(records, indent=2) + "\n")
print(records[:7])
print("uniform exact", sum(r.get("exact", False) for r in records))
print("uniform Galectin", records[27])
