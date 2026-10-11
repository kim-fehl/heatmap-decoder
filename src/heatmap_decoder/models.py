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
    labels: "LabelExtractionResult | None" = None


Axis: TypeAlias = Literal["rows", "columns"]
ReviewLabels: TypeAlias = dict[tuple[Axis, int], str]
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


@dataclass(frozen=True)
class OCRRun:
    """Raw engine evidence in resized-crop pixels and observed runtime metadata."""

    tokens: tuple[OCRToken, ...]
    engine_version: str
    wrapper_version: str
    language: str
    available_languages: tuple[str, ...]
    mode: str


@dataclass(frozen=True)
class LabelExtractionResult:
    config: GridConfig
    axes: tuple[AxisLabelResults, ...]
    ocr_runs: tuple[tuple[Axis, OCRRun], ...] = ()
    unreviewed_axes: tuple[AxisLabelResults, ...] | None = None
    review_sha256: str | None = None
    review_filename: str | None = None

    @property
    def unresolved_count(self) -> int:
        return sum(
            slot.status
            not in (
                {"accepted", "manual"}
                if self.config.labels is not None
                and getattr(self.config.labels, axis.axis) is not None
                else {"generated"}
            )
            or slot.final_text is None
            or not slot.final_text.strip()
            for axis in self.axes
            for slot in axis.slots
        )


ROIKind: TypeAlias = Literal["heatmap", "colourbar", "labels_rows", "labels_columns"]


class ROIReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    regions: dict[ROIKind, str] = Field(default_factory=dict)
    counts: dict[Axis, str] = Field(default_factory=dict)


class DetectorOptions(BaseModel):
    """Bounded heuristic settings, versioned separately from scientific QC."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    max_analysis_size: Annotated[int, Field(strict=True, ge=64, le=2048)] = 512
    max_candidates: Annotated[int, Field(strict=True, ge=1, le=100)] = 12
    max_components: Annotated[int, Field(strict=True, ge=1, le=256)] = 64
    background_distance: Annotated[float, Field(gt=0, le=100)] = 12.0
    gridline_gap: Annotated[int, Field(strict=True, ge=1, le=7)] = 3
    min_score: Annotated[float, Field(ge=0, le=1)] = 0.65
    max_count: Annotated[int, Field(strict=True, ge=2, le=512)] = 128
    max_count_candidates: Annotated[int, Field(strict=True, ge=2, le=32)] = 8
    label_search_pixels: Annotated[int, Field(strict=True, ge=16, le=1024)] = 256
    label_padding_pixels: Annotated[int, Field(strict=True, ge=0, le=32)] = 4
    max_ocr_pixels: Annotated[int, Field(strict=True, ge=1000, le=16000000)] = 4000000


class DetectionHints(BaseModel):
    """Supplied counts are authoritative; M8 inference requires an explicit axis request."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    rows: Annotated[int, Field(strict=True, gt=0)] | None = None
    columns: Annotated[int, Field(strict=True, gt=0)] | None = None
    infer_rows: Annotated[bool, Field(strict=True)] = False
    infer_columns: Annotated[bool, Field(strict=True)] = False
    requested: tuple[Literal["heatmap", "colourbar", "labels_rows", "labels_columns"], ...] = (
        "heatmap",
    )
    heatmap_bbox: BoundingBox | None = None
    colourbar_bbox: BoundingBox | None = None
    colourbar_orientation: Literal["vertical", "horizontal"] | None = None
    colourbar_start_value: float | None = None
    colourbar_end_value: float | None = None
    labels: LabelsConfig | None = None
    detector: DetectorOptions = Field(default_factory=DetectorOptions)

    @model_validator(mode="after")
    def validate_requests(self) -> "DetectionHints":
        for axis in ("rows", "columns"):
            if getattr(self, axis) is None and not getattr(self, f"infer_{axis}"):
                raise ValueError(f"supply {axis} or explicitly request infer_{axis}")
        if self.colourbar_bbox is not None and "colourbar" in self.requested:
            raise ValueError("manual colourbar_bbox conflicts with requested colourbar")
        if self.labels is not None:
            for axis in ("rows", "columns"):
                if getattr(self.labels, axis) is not None and f"labels_{axis}" in self.requested:
                    raise ValueError(f"fixed labels.{axis} conflicts with requested labels_{axis}")
        if len(set(self.requested)) != len(self.requested):
            raise ValueError("requested region kinds must be unique")
        if self.heatmap_bbox is not None and "heatmap" in self.requested:
            raise ValueError(
                "manual heatmap_bbox conflicts with requested heatmap; set requested=[]"
            )
        if self.heatmap_bbox is None and "heatmap" not in self.requested:
            raise ValueError("supply heatmap_bbox or request heatmap detection")
        for box in (self.heatmap_bbox, self.colourbar_bbox):
            if box is not None and (box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError("manual boxes must have positive width and height")
        return self


class ROICandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    id: str
    kind: Literal["heatmap", "colourbar", "labels_rows", "labels_columns"] = "heatmap"
    bbox: BoundingBox
    orientation: Literal["vertical", "horizontal"] | None = None
    score: Annotated[float, Field(ge=0, le=1)]
    score_components: dict[str, float]
    reasons: tuple[str, ...] = ("review_required",)
    associations: tuple[str, ...] = ()


class SelectedRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    bbox: BoundingBox
    origin: Literal["manual", "reviewed_detection"]
    candidate_id: str | None = None


class ROIAssociation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    id: str
    heatmap_id: str
    colourbar_id: str
    score: Annotated[float, Field(ge=0, le=1)]
    score_components: dict[str, float]
    reasons: tuple[str, ...] = ("review_required",)


class CountSignal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    positions: tuple[float, ...]
    alignment: Annotated[float, Field(ge=0, le=1)]
    coverage: Annotated[float, Field(ge=0, le=1)]
    kind: Literal["boundaries", "centres"]


class CountProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    id: str
    axis: Axis
    count: Annotated[int, Field(strict=True, gt=0)]
    heatmap_id: str
    bbox: BoundingBox
    pitch: Annotated[float, Field(gt=0)]
    phase: float
    score: Annotated[float, Field(ge=0, le=1)]
    signals: dict[str, CountSignal]
    reasons: tuple[str, ...]
    conflicts: tuple[str, ...] = ()
    origin: None = None


class SelectedCount(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    count: Annotated[int, Field(strict=True, gt=0)]
    origin: Literal["manual", "reviewed_detection"] = "manual"
    proposal_id: str | None = None


class LabelLocalisationEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    axis: Axis
    side: Literal["left", "right", "above", "below"]
    search_bbox: BoundingBox
    preprocessing: OCRConfig = Field(default_factory=OCRConfig)
    tokens: tuple[OCRToken, ...]
    unassigned_candidates: tuple[LabelCandidate, ...]
    assignment: AxisLabelResults | None = None
    engine: OCRRun


class DetectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    candidates: tuple[ROICandidate, ...]
    selected: dict[ROIKind, SelectedRegion]
    status: Literal["manual", "review_required", "ambiguous", "unresolved"]
    unresolved_requirements: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    associations: tuple[ROIAssociation, ...] = ()
    count_proposals: tuple[CountProposal, ...] = ()
    selected_counts: dict[Axis, SelectedCount] = Field(default_factory=dict)
    label_evidence: tuple[LabelLocalisationEvidence, ...] = ()


class SavedDetection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1, 2] = 1
    image_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    image_dimensions: tuple[
        Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]
    ]
    detector_version: str
    analysis_transforms: tuple[dict[str, int], ...] = ()
    hints: DetectionHints
    result: DetectionResult
    associations: tuple[str, ...] = ()
    proposed_config_complete: bool
    missing_extraction_fields: tuple[str, ...]
