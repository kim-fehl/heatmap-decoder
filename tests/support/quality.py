"""Quality contracts and fixtures."""

import numpy as np

from heatmap_decoder import ExtractionConfig


def reference():
    image = np.full((256, 56, 3), 255, dtype=np.uint8)
    levels = np.arange(16).reshape(4, 4) * 12 + 20
    cells = np.repeat(levels[..., None], 3, axis=-1).astype(np.uint8)
    image[:32, :32] = np.repeat(np.repeat(cells, 8, axis=0), 8, axis=1)
    bar = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 3, axis=1)
    image[:, 40:48] = bar[:, None]
    config = ExtractionConfig(
        heatmap_bbox=(0, 0, 32, 32),
        colourbar_bbox=(40, 0, 48, 256),
        rows=4,
        columns=4,
        colourbar_orientation="vertical",
        colourbar_start_value=0,
        colourbar_end_value=255,
    )
    return image, config
