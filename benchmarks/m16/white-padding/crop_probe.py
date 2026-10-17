import json
from pathlib import Path

import pytesseract
from PIL import Image, ImageFilter, ImageOps

im = Image.open("examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg").convert("L")
records = []
for box in [(857, 595, 985, 618), (859, 597, 976, 617), (859, 597, 976, 615), (859, 598, 976, 616)]:
    for variant in [
        "original",
        "contrast",
        "threshold120",
        "threshold160",
        "threshold200",
        "bold",
        "nearest",
        "wide",
    ]:
        crop = im.crop(box)
        if variant == "contrast":
            crop = ImageOps.autocontrast(crop)
        if variant.startswith("threshold"):
            crop = crop.point(lambda p: 255 if p >= int(variant[9:]) else 0)
        crop = crop.resize(
            (crop.width * (8 if variant == "wide" else 5), crop.height * 5),
            Image.Resampling.NEAREST if variant == "nearest" else Image.Resampling.BICUBIC,
        )
        if variant == "bold":
            crop = crop.filter(ImageFilter.MinFilter(3))
        crop = ImageOps.expand(crop, border=40, fill=255)
        for psm in [7, 8, 13]:
            d = pytesseract.image_to_data(
                crop, config=f"--oem 1 --psm {psm}", output_type=pytesseract.Output.DICT
            )
            words = [i for i, t in enumerate(d["text"]) if t.strip()]
            raw = " ".join(d["text"][i] for i in words)
            r = dict(
                box=box,
                variant=variant,
                psm=psm,
                white_border_output_pixels=40,
                raw=raw,
                confidence=min((float(d["conf"][i]) for i in words), default=None),
            )
            records.append(r)
            if "Galectin1" in raw:
                print(r, flush=True)
                crop.save("work/m16/recovered-galectin.png")
Path("work/m16/galectin-more-probe.json").write_text(json.dumps(records, indent=2) + "\n")
print(
    "Completed", len(records), "exact", sum(r["raw"] == "EsGalectin1" for r in records), flush=True
)
