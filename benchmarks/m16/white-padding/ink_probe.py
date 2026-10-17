import json
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image, ImageOps

im = Image.open("examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg").convert("L")
crop = im.crop((857, 595, 985, 618))
a = np.asarray(crop)
records = []
for threshold in [80, 100, 120, 160, 200]:
    ys, xs = np.where(a < threshold)
    for margin in [0, 1, 2]:
        box = (
            max(857, 857 + int(xs.min()) - margin),
            max(595, 595 + int(ys.min()) - margin),
            min(985, 858 + int(xs.max()) + margin),
            min(618, 596 + int(ys.max()) + margin),
        )
        for pad in [0, 2, 8]:
            field = ImageOps.expand(im.crop(box), border=pad, fill=255)
            field = field.resize((field.width * 5, field.height * 5), Image.Resampling.BICUBIC)
            d = pytesseract.image_to_data(
                field, config="--oem 1 --psm 7", output_type=pytesseract.Output.DICT
            )
            idx = [i for i, t in enumerate(d["text"]) if t.strip()]
            raw = " ".join(d["text"][i] for i in idx)
            r = dict(
                threshold=threshold,
                margin=margin,
                box=box,
                pad=pad,
                scale=5,
                psm=7,
                raw=raw,
                confidence=min((float(d["conf"][i]) for i in idx), default=None),
            )
            records.append(r)
            if raw == "EsGalectin1":
                print(r, flush=True)
Path("work/m16/galectin-tight-probe.json").write_text(json.dumps(records, indent=2) + "\n")
print(
    "Completed", len(records), "exact", sum(r["raw"] == "EsGalectin1" for r in records), flush=True
)
