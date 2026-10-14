"""Fresh M19 bases. Pillow ticks are rendered independently of OCR/fit code."""

import runpy
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

matrix_fixture = runpy.run_path(str(Path(__file__).parents[1] / "m17/fixtures.py"))["fixture"]

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def fixture(seed=290001, family="both", derivative="clean", challenge=None):
    old_challenge = challenge if challenge in {"missing", "white", "annotation", "panels"} else None
    image, config, truth = matrix_fixture(seed, family, challenge=old_challenge)
    canvas = Image.new("RGB", (image.shape[1] + 70, image.shape[0] + 50), "white")
    canvas.paste(Image.fromarray(image), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT if (seed // 2) % 2 else MONO, 18 + seed % 3)
    x, y, r, b = config.colourbar_bbox
    vertical = config.colourbar_orientation == "vertical"
    reverse = config.colourbar_start_value == 1
    tick_offsets = [20, 90, 160] if challenge == "unsupported_edges" else [0, 90, 180]
    ticks = []
    for offset in tick_offsets:
        value = 3 - offset / 30 if reverse else -3 + offset / 30
        if challenge == "inconsistent_ticks" and offset == 90:
            value = 1
        text = str(int(value)) if value.is_integer() else f"{value:.3g}"
        position = (y if vertical else x) + offset
        if vertical:
            draw.line((r, position, r + 6, position), fill="black", width=1)
            draw.text((r + 18, position), text, fill="black", font=font, anchor="lm")
        else:
            draw.line((position, b, position, b + 6), fill="black", width=1)
            draw.text((position, b + 18), text, fill="black", font=font, anchor="mt")
        ticks.append(dict(position=position, value=value))
    if challenge == "competing_legends":
        # A second gradient with equal geometry/quality is a competing model.
        copy = canvas.crop((x, y, r, b))
        canvas.paste(copy, (x + 35, y) if vertical else (x, b + 30))
    if challenge == "truncated":
        canvas = canvas.crop((32, 0, canvas.width, canvas.height))
    if derivative == "jpeg":
        stream = BytesIO()
        canvas.save(stream, format="JPEG", quality=88)
        stream.seek(0)
        canvas = Image.open(stream).convert("RGB")
    elif derivative == "blur":
        canvas = canvas.filter(ImageFilter.GaussianBlur(0.6))
    config = config.model_copy(
        update={
            "colourbar_start_value": 3 if reverse else -3,
            "colourbar_end_value": -3 if reverse else 3,
        }
    )
    return np.asarray(canvas), config, truth * 6 - 3, ticks
