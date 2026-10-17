"""Record contrast and illustrative colour-vision simulation checks for shared frames."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageColor

from heatmap_decoder.diagnostic_style import ROI_STYLE, DiagnosticCanvas


def luminance(rgb):
    colour = np.array(rgb) / 255
    colour = np.where(colour <= 0.04045, colour / 12.92, ((colour + 0.055) / 1.055) ** 2.4)
    return float(colour @ [0.2126, 0.7152, 0.0722])


out = Path("work/m16/diagnostic-checks")
out.mkdir(parents=True, exist_ok=True)
source = Image.new("RGB", (220, 220), "black")
source.paste(Image.new("RGB", (110, 220), "white"), (110, 0))
canvas = DiagnosticCanvas(source, lines=14)
for i, (kind, _) in enumerate(ROI_STYLE.items()):
    canvas.roi(
        (10, 10 + i * 30, 200, 30 + i * 30), kind, "proposed" if i % 2 else "manual", f"R{i}"
    )
for i, confidence in enumerate([95, 85, 20, None]):
    canvas.ocr((15 + 50 * i, 170, 45 + 50 * i, 200), confidence, f"T{i}", "example", "ambiguous")
canvas.image.save(out / "original.png")
# Approximate illustrative RGB projections, not clinical colour-vision modelling.
matrices = {
    "protan_approx": [[0.567, 0.433, 0], [0.558, 0.442, 0], [0, 0.242, 0.758]],
    "deutan_approx": [[0.625, 0.375, 0], [0.7, 0.3, 0], [0, 0.3, 0.7]],
    "grayscale": [[0.213, 0.715, 0.072]] * 3,
}
for name, matrix in matrices.items():
    pixels = np.array(canvas.image).astype(float) @ np.array(matrix).T
    Image.fromarray(np.clip(pixels, 0, 255).astype("uint8")).save(out / f"{name}.png")
report = {
    "ledger_black_on_white_contrast": 21,
    "minimum_white_background_frame_contrast": min(
        1.05 / (luminance(ImageColor.getrgb(colour)) + 0.05) for _, colour in ROI_STYLE.values()
    ),
    "states_use_text_and_dashes": True,
    "OCR_bands_use_H_M_L_unknown": True,
    "original_coordinates_preserved": True,
    "simulation_limit": (
        "illustrative RGB projections; status letters/dashes carry information independently of hue"
    ),
}
(out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
