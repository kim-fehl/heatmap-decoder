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


class OCRConfig(BaseModel):
    """Engineering defaults; engine confidence is a heuristic, not a probability."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    language: Annotated[str, Field(strict=True, min_length=1)] = "eng"
    upscale_factor: Annotated[int, Field(strict=True, ge=1)] = 3
    min_confidence: Annotated[float, Field(strict=True, ge=0, le=100)] = 80
    max_axis_offset_fraction: Annotated[float, Field(strict=True, gt=0, le=0.5)] = 0.4


class OCRLabelSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["ocr"]
    bbox: BoundingBox

    @model_validator(mode="after")
    def validate_extent(self) -> "OCRLabelSource":
        left, top, right, bottom = self.bbox
        if right <= left or bottom <= top:
            raise ValueError("label bbox must have positive width and height")
        return self


class FileLabelSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["file"]
    path: Annotated[str, Field(strict=True, min_length=1)]


LabelSource: TypeAlias = Annotated[OCRLabelSource | FileLabelSource, Field(discriminator="source")]


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rows: LabelSource | None = None
    columns: LabelSource | None = None
    ocr: OCRConfig = Field(default_factory=OCRConfig)


class GridConfig(BaseModel):
    """One shared heatmap grid for numerical sampling and label assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    heatmap_bbox: BoundingBox
    rows: Annotated[int, Field(strict=True, gt=0)]
    columns: Annotated[int, Field(strict=True, gt=0)]
    labels: LabelsConfig | None = None

    @model_validator(mode="after")
    def validate_grid_extent(self) -> "GridConfig":
        left, top, right, bottom = self.heatmap_bbox
        if right <= left or bottom <= top:
            raise ValueError("heatmap_bbox must have positive width and height")
        return self


class ExtractionConfig(GridConfig):
    """Exclusive pixel bounds; bar start is top (vertical) or left (horizontal)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    colourbar_bbox: BoundingBox
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


Axis: TypeAlias = Literal["rows", "columns"]
TextBox: TypeAlias = tuple[float, float, float, float]
LabelStatus: TypeAlias = Literal[
    "accepted", "low_confidence", "missing", "ambiguous", "manual", "generated"
]


@dataclass(frozen=True)
class OCRToken:
    """Raw evidence; adapter boxes are crop pixels until explicitly transformed."""

    text: str
    bbox: TextBox
    confidence: float
    line_id: tuple[int, ...] | None = None


@dataclass(frozen=True)
class LabelCandidate:
    text: str
    bbox: TextBox
    confidence: float
    token_indices: tuple[int, ...]


@dataclass(frozen=True)
class AxisSlot:
    index: int
    centre: float
    start: float
    end: float


@dataclass(frozen=True)
class LabelResult:
    index: int
    raw_text: str | None
    final_text: str | None
    confidence: float | None
    bbox: TextBox | None
    source: Literal["ocr", "file", "manual", "generated"]
    status: LabelStatus
    candidate_indices: tuple[int, ...] = ()
    duplicate: bool = False
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class AxisLabelResults:
    axis: Axis
    slots: tuple[LabelResult, ...]
    tokens: tuple[OCRToken, ...] = ()
    candidates: tuple[LabelCandidate, ...] = ()
    unmatched_candidate_indices: tuple[int, ...] = ()
