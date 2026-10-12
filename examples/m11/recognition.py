"""Build README recognition visuals from actual detector/OCR evidence, without exports."""

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from heatmap_decoder import DetectionHints, LegendConfig, inspect_legend_file
from heatmap_decoder.io import detect_file

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = Path(__file__).resolve().parent


def main():
    figure = EXAMPLE / "figure.png"
    truth = json.loads((EXAMPLE / "truth.json").read_text())
    hints = DetectionHints(
        rows=25,
        columns=8,
        requested=("heatmap", "labels_rows", "labels_columns"),
        colourbar_bbox=truth["colourbar_bbox"],
        colourbar_orientation="vertical",
    )
    (EXAMPLE / "hints.json").write_text(hints.model_dump_json(indent=2) + "\n")
    saved = detect_file(figure, hints, ROOT / "work/m11-readme-roi", overwrite=True)
    legend = inspect_legend_file(
        figure,
        LegendConfig(colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation="vertical"),
        ROOT / "work/m11-readme-legend",
        overwrite=True,
    )
    # Known synthetic reference review: never reuse on another source image.
    digest = hashlib.sha256(figure.read_bytes()).hexdigest()
    with (EXAMPLE / "reviewed_ticks.tsv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["index", "image_sha256", "include", "position", "value"])
        for tick in legend.ticks:
            nearest = min(
                range(len(truth["positions"])),
                key=lambda i: abs(truth["positions"][i] - tick.position),
            )
            recognised = (
                tick.position_source == "stroke"
                and abs(tick.position - truth["positions"][nearest]) <= 1
            )
            value = float(truth["tick_tokens"][nearest]["text"])
            writer.writerow(
                [
                    tick.index,
                    digest,
                    "yes" if recognised else "no",
                    tick.position,
                    value if recognised else "",
                ]
            )
    image = Image.open(figure).convert("RGB")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default(size=18)
    colours = {"heatmap": "#1875cc", "labels_rows": "#008a54", "labels_columns": "#da7900"}
    for kind, title in (
        ("heatmap", "Heatmap: 25 genes x 8 samples"),
        ("labels_rows", "Gene-label proposal"),
        ("labels_columns", "Sample-label proposal"),
    ):
        candidates = [c for c in saved.result.candidates if c.kind == kind]
        if not candidates:
            continue
        # Display the complete first-ranked proposal while retaining every losing
        # hypothesis in the linked machine-readable evidence.
        candidate = max(candidates, key=lambda c: c.score)
        x, y, r, s = candidate.bbox
        draw.rectangle((x, y, r - 1, s - 1), outline=colours[kind], width=3)
        caption = (
            (max(10, x - 210), y + 10)
            if kind == "labels_rows"
            else (x, s + 12)
            if kind == "labels_columns"
            else (x, max(5, y - 32))
        )
        draw.text(
            caption,
            title,
            fill=colours[kind],
            font=font,
            stroke_fill="white",
            stroke_width=2,
        )
    x, y, r, s = truth["colourbar_bbox"]
    draw.rectangle((x, y, r - 1, s - 1), outline="#9a30ae", width=3)
    draw.text(
        (x - 40, y - 55),
        "Selected colourbar",
        fill="#9a30ae",
        font=font,
        stroke_fill="white",
        stroke_width=2,
    )
    for tick in legend.ticks:
        if tick.text_bbox is not None:
            draw.rectangle(tick.text_bbox, outline="#9a30ae", width=2)
        draw.line((r, tick.position, r + 9, tick.position), fill="#9a30ae", width=2)
    draw.text(
        (320, 40),
        "Recognise regions and ticks; verify identities and scale before export",
        fill="black",
        font=font,
    )
    (ROOT / "docs/images").mkdir(parents=True, exist_ok=True)
    image.save(ROOT / "docs/images/recognition.png")
    (EXAMPLE / "recognition_evidence.json").write_text(
        json.dumps(
            {
                "source": "MIT synthetic development example",
                "roi": saved.model_dump(mode="json"),
                "legend": legend.model_dump(mode="json"),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
