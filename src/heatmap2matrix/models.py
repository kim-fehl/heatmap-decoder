"""Validated external configuration and simple in-memory results."""

from dataclasses import dataclass
from typing import Annotated, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

Pixel: TypeAlias = Annotated[int, Field(strict=True, ge=0)]
BoundingBox: TypeAlias = tuple[Pixel, Pixel, Pixel, Pixel]
FloatArray: TypeAlias = NDArray[np.float64]
RGBArray: TypeAlias = NDArray[np.uint8]


class ExtractionConfig(BaseModel):
    """Exclusive pixel bounds; bar start is top (vertical) or left (horizontal)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    heatmap_bbox: BoundingBox
    colourbar_bbox: BoundingBox
    rows: Annotated[int, Field(strict=True, gt=0)]
    columns: Annotated[int, Field(strict=True, gt=0)]
    colourbar_orientation: Literal["vertical", "horizontal"]
    colourbar_start_value: float
    colourbar_end_value: float
    sampling_inset_fraction: Annotated[float, Field(ge=0, lt=0.5)] = 0.25

    @model_validator(mode="after")
    def validate_extent(self) -> "ExtractionConfig":
        for name in ("heatmap_bbox", "colourbar_bbox"):
            left, top, right, bottom = getattr(self, name)
            if right <= left or bottom <= top:
                raise ValueError(f"{name} must have positive width and height")
        if self.colourbar_start_value == self.colourbar_end_value:
            raise ValueError("colourbar endpoints must differ")
        return self


@dataclass(frozen=True)
class ExtractionResult:
    """Approximate values in source row/column order; QC is deferred to M2."""

    values: FloatArray
