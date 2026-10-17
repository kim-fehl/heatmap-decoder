"""Legend contracts and fixtures."""

import numpy as np

from heatmap_decoder import ExtractionConfig
from heatmap_decoder.models import (
    LegendTick,
    ScaleAnchor,
)
from tests.support import ROOT as REPO_ROOT

ROOT = REPO_ROOT


def anchors_for(box, orientation, count=5, descending=True, nonuniform=False, interior=False):
    start, last = (box[1], box[3] - 1) if orientation == "vertical" else (box[0], box[2] - 1)
    t = np.linspace(0, 1, count) ** (1.5 if nonuniform else 1)
    if interior:
        t = 0.15 + 0.7 * t
    values = 2 - 4 * t if descending else -2 + 4 * t
    return tuple(
        ScaleAnchor(position=float(start + x * (last - start)), value=float(value))
        for x, value in zip(t, values, strict=True)
    )


def fixture_config(truth, scale):
    return ExtractionConfig(
        heatmap_bbox=truth["heatmap_bbox"],
        colourbar_bbox=truth["colourbar_bbox"],
        colourbar_orientation=truth["orientation"],
        rows=truth["rows"],
        columns=truth["columns"],
        scale=scale,
    )


def ticks_from_anchors(anchors, position_source="stroke"):
    return tuple(
        LegendTick(
            index=i,
            side="right",
            position=a.position,
            position_source=position_source,
            raw_text=str(a.value),
            parsed_value=a.value,
            confidence=99,
            status="suggested",
        )
        for i, a in enumerate(anchors)
    )
