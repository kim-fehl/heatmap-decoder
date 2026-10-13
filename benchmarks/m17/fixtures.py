"""M17 renderer. Truth and explicit geometry are evaluation-only annotations."""

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from heatmap_decoder.models import AxisSegment, ExtractionConfig


def fixture(seed=170001, family="both", derivative="clean", challenge=None):
    rng = np.random.default_rng(seed)
    nr, nc = int(rng.integers(6, 11)), int(rng.integers(6, 10))
    py, px = int(rng.integers(14, 25)), int(rng.integers(14, 25))
    gy, gx = max(4, round(py * 0.45)), max(4, round(px * 0.65))
    if challenge == "missing":
        gy = py
    if challenge == "panels":
        gx = px * 3

    def segments(start, n, pitch, blocked, gap):
        cut = n // 3
        if not blocked:
            return (AxisSegment(start=start, end=start + n * pitch, count=n),)
        return (
            AxisSegment(start=start, end=start + cut * pitch, count=cut),
            AxisSegment(
                start=start + cut * pitch + gap, end=start + n * pitch + gap, count=n - cut
            ),
        )

    rows = segments(32, nr, py, family in ("horizontal", "both"), gy)
    cols = segments(32, nc, px, family in ("vertical", "both"), gx)
    right, bottom = int(cols[-1].end), int(rows[-1].end)
    vertical = seed % 2 == 0
    reverse = seed % 3 == 0
    palette = seed % 3
    positions = rng.integers(10, 171, size=(nr, nc))
    truth = positions.astype(float) / 180

    def colour(v):
        if palette == 0:
            rgb = np.stack((40 + 180 * v, 60 + 30 * v, 220 - 170 * v), axis=-1)
        elif palette == 1:
            rgb = np.stack((30 + 130 * v, 20 + 190 * v, 160 - 120 * v), axis=-1)
        else:
            rgb = np.stack((35 + 190 * v, 35 + 190 * v, 35 + 190 * v), axis=-1)
        return np.rint(rgb).astype(np.uint8)

    im = np.full((max(bottom + 40, 260), right + 85, 3), 255, dtype=np.uint8)
    ri = 0
    for rs in rows:
        for i in range(rs.count):
            ci = 0
            for cs in cols:
                for j in range(cs.count):
                    y, x = int(rs.start + i * py), int(cs.start + j * px)
                    im[y : y + py, x : x + px] = colour(truth[ri, ci])
                    ci += 1
            ri += 1
    barbox = (right + 30, 32, right + 44, 213) if vertical else (32, bottom + 12, 213, bottom + 24)
    if not vertical:
        # Keep the horizontal legend inside the figure even for narrow matrices.
        expanded = np.full((bottom + 55, max(im.shape[1], 250), 3), 255, dtype=np.uint8)
        expanded[: min(im.shape[0], expanded.shape[0]), : im.shape[1]] = im[: expanded.shape[0]]
        im = expanded
    profile = colour(np.linspace(1 if reverse else 0, 0 if reverse else 1, 181))
    x, y, r, b = barbox
    im[y:b, x:r] = profile[:, None, :] if vertical else profile[None, :, :]
    if challenge == "white":
        # A complete palette-white row is data and must not disappear.
        im[32 + py : 32 + 2 * py, 32:right] = 255
    if challenge == "annotation":
        # Full-width colour annotation inserted where a gap would be.
        a, b = int(rows[0].end), int(rows[1].start)
        im[a:b, 32:right] = colour(0.5)
    if challenge == "cropped":
        im = im[32:]
        # This negative is exercised only through automatic detection.
    pil = Image.fromarray(im)
    if derivative == "jpeg":
        buffer = BytesIO()
        pil.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        pil = Image.open(buffer).convert("RGB")
    elif derivative == "blur":
        pil = pil.filter(ImageFilter.GaussianBlur(0.6))
    config = ExtractionConfig(
        heatmap_bbox=(32, 32, right, bottom),
        rows=nr,
        columns=nc,
        rows_segments=rows,
        columns_segments=cols,
        colourbar_bbox=barbox,
        colourbar_orientation="vertical" if vertical else "horizontal",
        colourbar_start_value=1 if reverse else 0,
        colourbar_end_value=0 if reverse else 1,
    )
    return np.asarray(pil), config, truth
