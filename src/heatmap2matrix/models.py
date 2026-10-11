"""Validated external configuration and simple in-memory results."""

from dataclasses import dataclass
from enum import IntFlag
from typing import Annotated, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

Pixel: TypeAlias = Annotated[int, Field(strict=True, ge=0)]
BoundingBox: TypeAlias = tuple[Pixel, Pixel, Pixel, Pixel]
FloatArray: TypeAlias = NDArray[np.float64]
RGBArray: TypeAlias = NDArray[np.uint8]
BoolArray: TypeAlias = NDArray[np.bool_]
RGBChannel: TypeAlias = Annotated[int, Field(strict=True, ge=0, le=255)]
MissingColour: TypeAlias = tuple[RGBChannel, RGBChannel, RGBChannel]


class CellFlag(IntFlag):
    OK = 0
    MISSING = 1
    POOR_MATCH = 2
    AMBIGUOUS = 4


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

    max_colour_distance: Annotated[float, Field(ge=0)] = 10.0
    ambiguity_colour_tolerance: Annotated[float, Field(ge=0)] = 1.0
    ambiguity_position_fraction: Annotated[float, Field(gt=0, le=1)] = 0.05
    minimum_valid_fraction: Annotated[float, Field(gt=0, le=1)] = 0.5
    missing_colours: tuple[MissingColour, ...] = ()
    missing_colour_tolerance: Annotated[float, Field(ge=0)] = 3.0

    @model_validator(mode="after")
    def validate_extent(self) -> "ExtractionConfig":
        for name in ("heatmap_bbox", "colourbar_bbox"):
            left, top, right, bottom = getattr(self, name)
            if right <= left or bottom <= top:
                raise ValueError(f"{name} must have positive width and height")
        if not np.isfinite(self.colourbar_end_value - self.colourbar_start_value):
            raise ValueError("colourbar numeric range must be finite")
        if self.colourbar_start_value == self.colourbar_end_value:
            raise ValueError("colourbar endpoints must differ")
        return self


@dataclass(frozen=True)
class ExtractionResult:
    """Row-major matrices; ΔE76 is a colour diagnostic, not numeric uncertainty.

    Missing and ambiguous values are NaN. Unambiguous poor matches retain their
    estimate and a flag. inferred_values preserves the nearest candidate for QC.
    """

    values: FloatArray
    inferred_values: FloatArray
    colour_distance: FloatArray
    flags: NDArray[np.uint8]
    valid_fraction: FloatArray
    sampled_rgb: FloatArray
    reconstructed_rgb: FloatArray
    config: ExtractionConfig
    image_shape: tuple[int, ...]
