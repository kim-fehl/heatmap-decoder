import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from rapidocr_onnxruntime import RapidOCR

engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
im = Image.open("examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg").convert("L")
records = []
for box in [(857, 595, 985, 618), (858, 596, 980, 617), (859, 597, 976, 617)]:
    for scale in [1, 3, 5]:
        for pad in [0, 1, 4]:
            for detection in [False, True]:
                crop = im.crop(box)
                crop = crop.resize(
                    (crop.width * scale, crop.height * scale), Image.Resampling.BICUBIC
                )
                crop = ImageOps.expand(crop, border=pad * scale, fill=255)
                result, elapsed = engine(np.asarray(crop), use_det=detection, use_cls=False)
                raw = " ".join(r[1] if detection else r[0] for r in result or [])
                r = dict(box=box, scale=scale, pad=pad, detection=detection, raw=raw, result=result)
                records.append(r)
                if raw == "EsGalectin1":
                    print(r, flush=True)
Path("work/m16/rapid-galectin-probe.json").write_text(json.dumps(records, indent=2) + "\n")
print("Done", len(records), flush=True)
