"""Rotation fixtures contracts and fixtures."""

from heatmap_decoder.models import GridConfig


def config(angle=90, recognition="sparse"):
    return GridConfig.model_validate(
        {
            "heatmap_bbox": [10, 110, 130, 150],
            "rows": 2,
            "columns": 3,
            "labels": {
                "columns": {
                    "source": "ocr",
                    "bbox": [10, 20, 130, 100],
                    "text_rotation_degrees": angle,
                    "recognition": recognition,
                }
            },
        }
    )
