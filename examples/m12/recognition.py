"""Regenerate the README's observed proposals and composable decision fragment."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

from heatmap_decoder import AutomationRequest, DetectionHints, automate, detect, load_raster

ROOT = Path(__file__).resolve().parents[2]


def main():
    directory = ROOT / "examples/m12"
    image = load_raster(directory / "figure.png")
    request = AutomationRequest.model_validate_json((directory / "request.json").read_bytes())
    result = automate(image, request)
    labels = detect(
        image,
        DetectionHints(
            rows=20, columns=6, requested=("heatmap", "colourbar", "labels_rows", "labels_columns")
        ),
    )
    evidence = dict(
        automation=result.model_dump(mode="json"), label_localisation=labels.model_dump(mode="json")
    )
    (directory / "recognition_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    overlay = Image.fromarray(image[:, :, :3])
    painter = ImageDraw.Draw(overlay)
    colours = {
        "heatmap": "blue",
        "colourbar": "purple",
        "labels_rows": "green",
        "labels_columns": "orange",
    }
    for candidate in labels.candidates:
        x, y, r, s = candidate.bbox
        colour = colours[candidate.kind]
        painter.rectangle((x, y, r - 1, s - 1), outline=colour, width=3)
        painter.text(
            (x, max(0, y - 18)), candidate.kind, fill=colour, stroke_width=1, stroke_fill="white"
        )
    if result.legend:
        for tick in result.legend.ticks:
            if tick.text_bbox is not None:
                painter.rectangle(tick.text_bbox, outline="purple", width=2)
    overlay.save(ROOT / "docs/images/recognition-m12.png")
    print(result.complete.status, result.automatic_export_allowed)
    if result.legend and result.legend.fit:
        print(result.legend.fit.model_dump_json())


if __name__ == "__main__":
    main()
