import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
m = json.loads(Path("examples/m16/manifest.json").read_text())
im = Image.open(m["image"]).convert("RGB")
records = []
for row in range(33):
    box = (
        857,
        math.floor(189 + row * 672 / 33) - 1,
        985,
        math.ceil(189 + (row + 1) * 672 / 33) + 1,
    )
    crop = ImageOps.expand(im.crop(box).convert("L"), border=1, fill=255)
    crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.BICUBIC)
    result, elapsed = engine(np.asarray(crop), use_det=False, use_cls=False)
    records.append(dict(row=row, box=box, raw=result, elapsed=elapsed))
    print(row, result, flush=True)
Path("work/m16/rapid-probe.json").write_text(json.dumps(records, indent=2) + "\n")
print("EXACT", sum(r["raw"][0][0] == m["displayed_rows"][r["row"]] for r in records), flush=True)
