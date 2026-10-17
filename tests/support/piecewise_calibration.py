"""Piecewise calibration contracts and fixtures."""

from heatmap_decoder.models import ScaleAnchor


def anchors(positions=(10, 40, 110), values=(0, 1, 5)):
    return tuple(ScaleAnchor(position=p, value=v) for p, v in zip(positions, values, strict=True))
