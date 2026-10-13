"""Inspection of the authoritative dense observed colourbar; no axial smoothing."""

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .calibration import to_lab
from .geometry import interior_slice
from .models import ExtractionConfig, LegendConfig, RGBArray


def inspect_gradient(
    image: RGBArray, config: ExtractionConfig | LegendConfig, tick_positions: tuple[float, ...] = ()
) -> dict[str, Any]:
    x, y, r, s = config.colourbar_bbox
    vertical = config.colourbar_orientation == "vertical"
    transverse = interior_slice(x if vertical else y, r if vertical else s, 0.25)
    start, last = (y, s - 1) if vertical else (x, r - 1)
    ticks = sorted(set(p for p in tick_positions if start <= p <= last))
    positions = sorted(
        set([float(start), float(last), *ticks, *[(a + b) / 2 for a, b in zip(ticks, ticks[1:])]])
    )
    checkpoints = []
    for p in positions:
        pixel = int(np.floor(p + 0.5))
        patch = (
            image[pixel : pixel + 1, transverse, :3]
            if vertical
            else image[transverse, pixel : pixel + 1, :3]
        )
        rgb = np.median(patch.reshape(-1, 3), axis=0)
        dispersion = np.percentile(np.linalg.norm(patch.reshape(-1, 3) - rgb, axis=1), 90)
        footprint = (
            [transverse.start, pixel, transverse.stop, pixel + 1]
            if vertical
            else [pixel, transverse.start, pixel + 1, transverse.stop]
        )
        missing = bool(
            image.shape[2] == 4
            and np.any(image[footprint[1] : footprint[3], footprint[0] : footprint[2], 3] != 255)
        )
        value = (
            next((a.value for a in config.scale.anchors if a.position == p), None)
            if isinstance(config, ExtractionConfig) and config.scale
            else None
        )
        checkpoints.append(
            {
                "position": p,
                "reviewed_value": value,
                "sample_position": pixel,
                "footprint": footprint,
                "rgb": rgb.tolist(),
                "lab": to_lab(rgb[None, :])[0].tolist(),
                "transverse_rgb_dispersion_p90": float(dispersion),
                "missing_alpha": missing,
                "contamination_evidence": "dispersion only; not a classifier",
                "kind": "tick" if p in ticks else "edge" if p in (start, last) else "between_ticks",
            }
        )
    return {
        "method": "single axial pixel; median transverse central half",
        "axial_aggregation": False,
        "orientation": config.colourbar_orientation,
        "bbox": config.colourbar_bbox,
        "checkpoints": checkpoints,
    }


def gradient_diagnostic(image: RGBArray, evidence: dict[str, Any]) -> Image.Image:
    x, y, r, s = evidence["bbox"]
    vertical = evidence["orientation"] == "vertical"
    crop = Image.fromarray(image[:, :, :3]).crop((x, y, r, s))
    if vertical:
        crop = crop.transpose(Image.Transpose.ROTATE_90)
    width = max(800, crop.width * 4)
    canvas = Image.new("RGB", (width + 20, 200 + len(evidence["checkpoints"]) * 25), "white")
    canvas.paste(crop.resize((width, 80), Image.Resampling.NEAREST), (10, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 10), "Observed Colourbar [manual/selected]; dense pixels preserved", fill="black"
    )
    start, last = (y, s - 1) if vertical else (x, r - 1)
    for i, point in enumerate(evidence["checkpoints"]):
        px = 10 + (point["position"] - start) / max(1, last - start) * (width - 1)
        draw.line((px, 122, px, 137), fill="black", width=2)
        at = 170 + i * 25
        draw.rectangle((10, at, 35, at + 18), fill=tuple(int(v) for v in point["rgb"]))
        draw.text(
            (45, at),
            f"C{i}: {point['kind']} pos={point['position']:.2f} "
            f"value={point['reviewed_value']} "
            f"footprint={point['footprint']} "
            f"dispersion={point['transverse_rgb_dispersion_p90']:.2f}",
            fill="black",
        )
    return canvas
