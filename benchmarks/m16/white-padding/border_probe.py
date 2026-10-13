import json
from pathlib import Path

import pytesseract
from PIL import Image, ImageOps

im = Image.open("examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg").convert("L")
records = []
for scale in [3, 4, 6]:
    for pad in [0, 2, 4, 8, 12, 16]:
        for psm in [7, 8, 13]:
            crop = ImageOps.expand(im.crop((857, 595, 985, 618)), border=pad, fill=255)
            crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.BICUBIC)
            d = pytesseract.image_to_data(
                crop, config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT
            )
            words = [i for i, t in enumerate(d["text"]) if t.strip()]
            raw = " ".join(d["text"][i] for i in words)
            r = dict(
                scale=scale,
                pad=pad,
                psm=psm,
                raw=raw,
                confidence=min((float(d["conf"][i]) for i in words), default=None),
            )
            records.append(r)
            if raw == "EsGalectin1":
                print(r, flush=True)
Path("work/m16/white-border-probe.json").write_text(json.dumps(records, indent=2) + "\n")
print(
    "Completed", len(records), "exact", sum(r["raw"] == "EsGalectin1" for r in records), flush=True
)
