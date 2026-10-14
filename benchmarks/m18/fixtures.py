"""Independent text/layout renderer for M18; annotations never enter detection."""

import math
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from heatmap_decoder.models import AxisSegment, ExtractionConfig, GridConfig

FONT = str(Path(__file__).resolve().parents[1] / "fonts/DejaVuSans.ttf")


def fixture(
    angle=30,
    below=False,
    blocked=False,
    seed=1801,
    derivative="clean",
    missing=(),
    header=False,
    names=None,
    font_size=14,
    pitch=110,
    font_path=FONT,
):
    rng = np.random.default_rng(seed)
    n = 4
    left = 240
    top = 360
    bottom = 480
    gap = 17 if blocked else 0
    right = left + n * pitch + gap
    size = (right + 230, 850)
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    columns = (
        (
            AxisSegment(start=left, end=left + 2 * pitch, count=2),
            AxisSegment(start=left + 2 * pitch + gap, end=right, count=2),
        )
        if blocked
        else None
    )
    config = GridConfig(
        heatmap_bbox=(left, top, right, bottom),
        rows=4,
        columns=n,
        columns_segments=columns,
        labels={
            "columns": {
                "source": "ocr",
                "bbox": (100, bottom + 2, size[0] - 20, 820)
                if below
                else (100, 20, size[0] - 20, top - 2),
                "text_rotation_degrees": angle,
                "recognition": "lines",
            },
            "ocr": {"engine": "tesseract", "upscale_factor": 1},
        },
    )
    from heatmap_decoder.geometry import axis_slots

    positions = rng.integers(10, 171, (4, n))
    truth = positions / 180
    profile = (
        np.stack(
            (
                30 + 180 * np.linspace(0, 1, 181),
                np.full(181, 90),
                220 - 170 * np.linspace(0, 1, 181),
            ),
            axis=1,
        )
        .round()
        .astype(np.uint8)
    )
    for row in axis_slots(config, "rows"):
        for col in axis_slots(config, "columns"):
            draw.rectangle(
                (int(col.start), int(row.start), int(col.end) - 1, int(row.end) - 1),
                fill=tuple(profile[positions[row.index, col.index]]),
            )
    barbox = (40, top, 54, top + 181)
    pixels = np.asarray(canvas).copy()
    pixels[top : top + 181, 40:54] = profile[:, None]
    canvas = Image.fromarray(pixels)
    font = ImageFont.truetype(font_path, font_size)
    names = names or [
        f"Sample_{i + 1}" if angle % 180 == 0 else f"Sample_{i + 1} long {10 + seed % 80}"
        for i in range(n)
    ]
    annotations = []
    for slot, text in zip(axis_slots(config, "columns"), names, strict=True):
        if slot.index in missing:
            continue
        box = font.getbbox(text)
        width = box[2] - box[0] + 4
        height = box[3] - box[1] + 4
        tile = Image.new("L", (width, height), 255)
        ImageDraw.Draw(tile).text((2 - box[0], 2 - box[1]), text, font=font, fill=0)
        # Render independently with Pillow rotate; no detector transform receives truth.
        rotated = tile.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC, fillcolor=255)
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        corners = [(c * x + s * y, -s * x + c * y) for x in (0, width) for y in (0, height)]
        tx, ty = -min(x for x, y in corners), -min(y for x, y in corners)
        start = (s * height / 2 + tx, c * height / 2 + ty)
        end = (c * width + s * height / 2 + tx, -s * width + c * height / 2 + ty)
        if abs(s) < 1e-8:
            endpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            role = "centred"
        else:
            start_near = start[1] < end[1] if below else start[1] > end[1]
            endpoint = start if start_near else end
            role = "start" if start_near else "end"
        target = (slot.centre, bottom + 16 if below else top - 16)
        origin = (round(target[0] - endpoint[0]), round(target[1] - endpoint[1]))
        canvas.paste(Image.new("RGB", rotated.size, "black"), origin, ImageOps.invert(rotated))
        annotations.append(
            dict(
                slot=slot.index,
                text=text,
                attachment=target,
                endpoint=role,
                tile_bbox=(
                    origin[0],
                    origin[1],
                    origin[0] + rotated.width,
                    origin[1] + rotated.height,
                ),
            )
        )
    if header:
        ImageDraw.Draw(canvas).text(
            (left + pitch, bottom + 250 if below else top - 250),
            "Group header",
            font=font,
            fill="black",
        )
    if derivative == "blur":
        canvas = canvas.filter(ImageFilter.GaussianBlur(0.55))
    elif derivative == "jpeg":
        buffer = BytesIO()
        canvas.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        canvas = Image.open(buffer).convert("RGB")
    numeric = ExtractionConfig.model_validate(
        {
            **config.model_dump(),
            "colourbar_bbox": barbox,
            "colourbar_orientation": "vertical",
            "colourbar_start_value": 0,
            "colourbar_end_value": 1,
        }
    )
    return np.asarray(canvas), config, numeric, truth, annotations
