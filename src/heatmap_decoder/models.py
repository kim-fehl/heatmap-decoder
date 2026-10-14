"""Validated external configuration and simple in-memory results."""

import math
from dataclasses import dataclass
from enum import IntFlag
from typing import Annotated, Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    UNCALIBRATED = 8


class OCRConfig(BaseModel):
    """Engineering defaults; engine confidence is a heuristic, not a probability."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    engine: Literal["tesseract", "rapidocr"] = "tesseract"
    language: Annotated[str, Field(strict=True, min_length=1)] = "eng"
    upscale_factor: Annotated[int, Field(strict=True, ge=1)] = 3
    min_confidence: Annotated[float, Field(strict=True, ge=0, le=100)] = 80
    high_confidence: Annotated[float, Field(strict=True, ge=0, le=100)] = 90
    retry_variants: tuple[Literal["contrast", "threshold"], ...] = ()

    @field_validator("retry_variants")
    @classmethod
    def validate_retries(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 2 or len(set(value)) != len(value):
            raise ValueError("at most two distinct OCR retry variants")
        return value

    @model_validator(mode="after")
    def validate_confidence_bands(self) -> "OCRConfig":
        if self.high_confidence < self.min_confidence:
            raise ValueError("high_confidence must be >= min_confidence")
        return self

    max_axis_offset_fraction: Annotated[float, Field(strict=True, gt=0, le=0.5)] = 0.4


def normalise_text_angle(value: Any) -> float:
    """Finite numeric degrees in [-180,180), preserving visual CCW direction."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("text_rotation_degrees must be a finite numeric angle")
    return float((value + 180) % 360 - 180)


class OCRLabelSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["ocr"]
    bbox: BoundingBox
    recognition: Literal["sparse", "slots", "lines"] = "sparse"
    crop_padding_pixels: Annotated[int, Field(strict=True, ge=0, le=2)] = 0
    crop_right_padding_pixels: Annotated[int, Field(strict=True, ge=0, le=16)] = 0
    white_padding_pixels: Annotated[int, Field(strict=True, ge=0, le=2)] = 0
    text_rotation_degrees: Annotated[float, Field(strict=True, allow_inf_nan=False)] = 0
    attachment_endpoint: Literal["nearest", "start", "end"] = "nearest"
    max_attachment_distance_pixels: Annotated[float, Field(gt=0, le=256)] = 40.0

    @field_validator("text_rotation_degrees", mode="before")
    @classmethod
    def validate_rotation(cls, value: Any) -> float:
        return normalise_text_angle(value)

    @model_validator(mode="after")
    def validate_extent(self) -> "OCRLabelSource":
        if self.recognition not in ("slots", "lines") and (
            self.crop_padding_pixels or self.crop_right_padding_pixels or self.white_padding_pixels
        ):
            raise ValueError("label padding requires explicit slot or line recognition")
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
    character_policy: Literal["latin_greek", "unrestricted"] = "latin_greek"
    ocr: OCRConfig = Field(default_factory=lambda: OCRConfig(engine="rapidocr", upscale_factor=1))

    @field_validator("ocr", mode="before")
    @classmethod
    def label_engine_defaults(cls, value: Any) -> Any:
        if isinstance(value, dict):
            factor = 3 if value.get("engine") == "tesseract" else 1
            return {"engine": "rapidocr", "upscale_factor": factor, **value}
        return value

    @model_validator(mode="after")
    def validate_row_rotation(self) -> "LabelsConfig":
        if isinstance(self.rows, OCRLabelSource) and self.rows.recognition == "lines":
            raise ValueError("oriented line recognition is supported for columns only")
        if isinstance(self.columns, OCRLabelSource) and self.columns.crop_right_padding_pixels:
            raise ValueError("right-only crop padding is supported for gene/row labels only")
        if isinstance(self.rows, OCRLabelSource) and self.rows.text_rotation_degrees != 0:
            raise ValueError("text_rotation_degrees is supported for columns only")
        return self


class AxisSegment(BaseModel):
    """Original-coordinate half-open bounds and an explicit cell count."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    start: Annotated[float, Field(strict=True, ge=0)]
    end: Annotated[float, Field(strict=True, gt=0)]
    count: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def validate_bounds(self) -> "AxisSegment":
        if self.end <= self.start:
            raise ValueError("segment end must exceed start")
        return self


def validate_segments(
    segments: tuple[AxisSegment, ...] | None, start: float, end: float, count: int, axis: str
) -> None:
    if segments is None:
        return
    if not segments:
        raise ValueError(f"{axis}_segments must be nonempty")
    if segments[0].start != start or segments[-1].end != end:
        raise ValueError(f"{axis}_segments must span the heatmap ROI endpoints")
    previous = start
    for segment in segments:
        if not start <= segment.start < segment.end <= end or segment.start < previous:
            raise ValueError(f"{axis}_segments must be ordered, non-overlapping and in bounds")
        previous = segment.end
    if sum(segment.count for segment in segments) != count:
        raise ValueError(f"{axis}_segments counts contradict total {axis}")


def segment_totals(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    value = dict(value)
    for axis in ("rows", "columns"):
        segments = value.get(f"{axis}_segments")
        if value.get(axis) is None and segments:
            parsed = tuple(AxisSegment.model_validate(s) for s in segments)
            value[axis] = sum(s.count for s in parsed)
    return value


class GridConfig(BaseModel):
    """One shared heatmap grid for numerical sampling and label assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    heatmap_bbox: BoundingBox
    rows: Annotated[int, Field(strict=True, gt=0)]
    columns: Annotated[int, Field(strict=True, gt=0)]
    labels: LabelsConfig | None = None
    rows_segments: tuple[AxisSegment, ...] | None = None
    columns_segments: tuple[AxisSegment, ...] | None = None

    @model_validator(mode="before")
    @classmethod
    def resolve_totals(cls, value: Any) -> Any:
        return segment_totals(value)

    @model_validator(mode="after")
    def validate_grid_extent(self) -> "GridConfig":
        left, top, right, bottom = self.heatmap_bbox
        if right <= left or bottom <= top:
            raise ValueError("heatmap_bbox must have positive width and height")
        validate_segments(self.rows_segments, top, bottom, self.rows, "rows")
        validate_segments(self.columns_segments, left, right, self.columns, "columns")
        return self


class ScaleAnchor(BaseModel):
    """Original-image axial pixel coordinate; the first/last sampled pixels are endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    position: Annotated[float, Field(ge=0)]
    value: float
    origin: Literal["manual", "ocr", "reviewed"] = "manual"
    status: Literal["suggested", "reviewed"] = "reviewed"


class LegendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    ocr: OCRConfig = Field(default_factory=lambda: OCRConfig(engine="rapidocr", upscale_factor=1))
    search_pixels: Annotated[int, Field(strict=True, ge=16, le=256)] = 128
    horizontal_search_pixels: Annotated[int, Field(strict=True, ge=16, le=256)] = 64
    max_ticks: Annotated[int, Field(strict=True, ge=2, le=32)] = 16
    max_ocr_pixels: Annotated[int, Field(strict=True, ge=1000, le=16000000)] = 4000000
    max_ocr_seconds: Annotated[float, Field(gt=0, le=120)] = 60.0
    max_axis_offset_pixels: Annotated[float, Field(gt=0, le=32)] = 6.0
    endpoint_tolerance_pixels: Annotated[float, Field(ge=0, le=8)] = 2.0
    decimal_separator: Literal[".", ","] = "."
    align_tick_fields: bool = False
    retry_unconstrained_numeric: bool = False
    retry_white_padding_pixels: tuple[Annotated[int, Field(strict=True, ge=0, le=2)], ...] = ()

    @field_validator("ocr", mode="before")
    @classmethod
    def tick_engine_defaults(cls, value: Any) -> Any:
        if isinstance(value, dict):
            factor = 3 if value.get("engine") == "tesseract" else 1
            return {"engine": "rapidocr", "upscale_factor": factor, **value}
        return value

    @model_validator(mode="before")
    @classmethod
    def tick_padding_defaults(cls, value: Any) -> Any:
        if isinstance(value, dict) and "white_padding_pixels" not in value:
            ocr = value.get("ocr")
            engine = (
                ocr.get("engine", "rapidocr")
                if isinstance(ocr, dict)
                else (ocr.engine if isinstance(ocr, OCRConfig) else "rapidocr")
            )
            return {"white_padding_pixels": 2 if engine == "rapidocr" else 0, **value}
        return value

    @field_validator("retry_white_padding_pixels")
    @classmethod
    def validate_tick_retries(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if len(value) > 2 or len(set(value)) != len(value):
            raise ValueError("at most two distinct tick padding retries")
        return value

    crop_padding_pixels: Annotated[int, Field(strict=True, ge=0, le=2)] = 0
    white_padding_pixels: Annotated[int, Field(strict=True, ge=0, le=2)] = 0


class ScaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    model: Literal["linear", "piecewise_linear"] = "linear"
    source: Literal["manual_endpoints", "manual_ticks", "ocr_ticks"]
    anchors: tuple[ScaleAnchor, ...] = ()
    legend: LegendOptions = Field(default_factory=LegendOptions)
    quantity: Annotated[str, Field(min_length=1)] = "displayed colourbar values"
    max_value_residual_fraction: Annotated[float, Field(ge=0, le=0.05)] = 0.01
    max_position_residual_pixels: Annotated[float, Field(ge=0, le=8)] = 2.0

    @model_validator(mode="after")
    def validate_source(self) -> "ScaleConfig":
        if self.model == "piecewise_linear":
            if self.source == "manual_endpoints":
                raise ValueError("piecewise_linear requires positional anchors")
            if self.source == "manual_ticks" and len(self.anchors) < 3:
                raise ValueError("piecewise_linear requires at least three anchors")
        if self.source == "manual_ticks" and len(self.anchors) < 2:
            raise ValueError("manual_ticks requires at least two anchors")
        if self.source != "manual_ticks" and self.anchors:
            raise ValueError("anchors belong to manual_ticks; OCR evidence is not configuration")
        return self


class LegendConfig(BaseModel):
    """Legend-only inspection does not require a heatmap or numerical endpoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    colourbar_bbox: BoundingBox
    colourbar_orientation: Literal["vertical", "horizontal"]
    legend: LegendOptions = Field(default_factory=LegendOptions)
    scale_model: Literal["linear", "piecewise_linear"] = "linear"


class ExtractionConfig(GridConfig):
    """Exclusive pixel bounds; bar start is top (vertical) or left (horizontal)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    colourbar_bbox: BoundingBox
    colourbar_orientation: Literal["vertical", "horizontal"]
    colourbar_start_value: float | None = None
    colourbar_end_value: float | None = None
    scale: ScaleConfig | None = None
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
        endpoint_source = self.scale is None or self.scale.source == "manual_endpoints"
        if endpoint_source:
            if self.colourbar_start_value is None or self.colourbar_end_value is None:
                raise ValueError("manual_endpoints requires start and end values")
            if not np.isfinite(self.colourbar_end_value - self.colourbar_start_value):
                raise ValueError("colourbar numeric range must be finite")
            if self.colourbar_start_value == self.colourbar_end_value:
                raise ValueError("colourbar endpoints must differ")
        elif self.colourbar_start_value is not None or self.colourbar_end_value is not None:
            raise ValueError("tick scale conflicts with manual endpoint values")
        return self


@dataclass(frozen=True)
class ColourCandidateComponent:
    """Inclusive bar sample endpoints and their actual calibrated numeric range."""

    first_index: int
    last_index: int
    first_position: float
    last_position: float
    minimum_value: float | None
    maximum_value: float | None
    calibrated: bool = True


@dataclass(frozen=True)
class CellColourEvidence:
    """Engineering near-match sets, not calibrated statistical confidence intervals."""

    kind: Literal["local", "broad_contiguous", "disconnected_near_matches"]
    components: tuple[ColourCandidateComponent, ...]
    position_span_fraction: float
    value_span: float | None


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
    range_limit: NDArray[np.int8] | None = None
    colour_evidence: tuple[CellColourEvidence | None, ...] | None = None


Axis: TypeAlias = Literal["rows", "columns"]
ReviewLabels: TypeAlias = dict[tuple[Axis, int], str]
TextBox: TypeAlias = tuple[float, float, float, float]
LabelStatus: TypeAlias = Literal[
    "accepted", "low_confidence", "missing", "ambiguous", "manual", "generated", "unreviewed"
]


@dataclass(frozen=True)
class OCRToken:
    """Raw evidence; adapter boxes are crop pixels until explicitly transformed."""

    text: str
    bbox: TextBox
    confidence: float
    line_id: tuple[int, ...] | None = None
    polygon: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class LabelCandidate:
    text: str
    bbox: TextBox
    confidence: float
    token_indices: tuple[int, ...]
    slot_index: int | None = None
    polygon: tuple[tuple[float, float], ...] = ()
    baseline: tuple[tuple[float, float], ...] = ()
    attachment: tuple[float, float] | None = None
    geometry_issues: tuple[str, ...] = ()


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
    suggested_text: str | None = None


@dataclass(frozen=True)
class AxisLabelResults:
    axis: Axis
    slots: tuple[LabelResult, ...]
    tokens: tuple[OCRToken, ...] = ()
    candidates: tuple[LabelCandidate, ...] = ()
    unmatched_candidate_indices: tuple[int, ...] = ()
    geometry_evidence: dict[str, Any] | None = None


@dataclass(frozen=True)
class OCRRun:
    """Raw engine evidence in resized-crop pixels and observed runtime metadata."""

    tokens: tuple[OCRToken, ...]
    engine_version: str
    wrapper_version: str
    language: str
    available_languages: tuple[str, ...]
    mode: str
    preprocessing: str = "grayscale_bicubic"
    white_padding_pixels: int = 0
    engine_name: str = "Tesseract"
    geometry_kind: str = "detected_text"
    score_kind: str = "engine_confidence_0_100"
    model_sha256: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LabelExtractionResult:
    config: GridConfig
    axes: tuple[AxisLabelResults, ...]
    ocr_runs: tuple[tuple[Axis, OCRRun], ...] = ()
    unreviewed_axes: tuple[AxisLabelResults, ...] | None = None
    review_sha256: str | None = None
    review_filename: str | None = None
    slot_ocr_runs: tuple[tuple[Axis, int, BoundingBox, OCRRun], ...] = ()
    skipped_ocr_attempts: tuple[dict[str, Any], ...] = ()
    oriented_ocr_records: tuple[dict[str, Any], ...] = ()

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

    count_label_geometry: Annotated[bool, Field(strict=True)] = False
    propose_column_rotations: Annotated[bool, Field(strict=True)] = False
    propose_arbitrary_column_angles: Annotated[bool, Field(strict=True)] = False
    column_attachment_endpoint: Literal["nearest", "start", "end"] = "nearest"
    max_label_ocr_calls: Annotated[int, Field(strict=True, ge=1, le=24)] = 12
    max_label_ocr_seconds: Annotated[float, Field(gt=0, le=120)] = 60.0


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
    rows_segments: tuple[AxisSegment, ...] | None = None
    columns_segments: tuple[AxisSegment, ...] | None = None
    detector: DetectorOptions = Field(default_factory=DetectorOptions)

    @model_validator(mode="before")
    @classmethod
    def resolve_totals(cls, value: Any) -> Any:
        return segment_totals(value)

    @model_validator(mode="after")
    def validate_requests(self) -> "DetectionHints":
        for axis in ("rows", "columns"):
            segments = getattr(self, f"{axis}_segments")
            if segments is not None:
                if self.heatmap_bbox is None:
                    raise ValueError("explicit segments require a supplied heatmap_bbox")
                coordinate = 1 if axis == "rows" else 0
                validate_segments(
                    segments,
                    self.heatmap_bbox[coordinate],
                    self.heatmap_bbox[coordinate + 2],
                    getattr(self, axis),
                    axis,
                )
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
    text_rotation_degrees: float = 0


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
    observation: str = "legacy_correlated_unknown"
    excluded_gridline_pixels: bool = False


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
    origin: Literal["manual", "reviewed_detection", "measured_inference"] = "manual"
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
    heatmap_id: str = "heatmap-manual"
    trial_id: str | None = None
    text_rotation_degrees: float = 0
    raw_tokens: tuple[OCRToken, ...] = ()
    transform: dict[str, Any] = Field(default_factory=dict)
    ocr_pixels: int = 0
    ocr_seconds: float = 0.0
    orientation_status: Literal["legacy", "unique", "ambiguous", "unsupported", "capped"] = "legacy"
    orientation_reasons: tuple[str, ...] = ()
    count_geometry_eligible: bool = True


class AcceptanceDecision(BaseModel):
    """A measured development decision, separate from permission to export automatically."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    requirement: str
    status: Literal["accepted", "review_required", "unresolved"]
    reasons: tuple[str, ...]
    evidence: dict[str, Any] = Field(default_factory=dict)
    candidate_ids: tuple[str, ...] = ()
    policy_version: str


class AcceptanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decisions: tuple[AcceptanceDecision, ...]
    overall: AcceptanceDecision
    automatic_export_allowed: Literal[False] = False
    identifier_verification: Literal["not_requested", "review_required", "user_supplied"]


class BlockAxisProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    axis: Axis
    segments: tuple[AxisSegment, ...]
    gaps: tuple[tuple[float, float], ...]
    count: int
    pitch: float
    status: Literal["accepted", "review_required", "unresolved"]
    reasons: tuple[str, ...]
    evidence: dict[str, Any]
    alternatives: tuple[dict[str, Any], ...] = ()


class BlockGridProposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    heatmap_id: str
    bbox: BoundingBox
    axes: tuple[BlockAxisProposal, ...]
    status: Literal["accepted", "review_required", "unresolved"]
    reasons: tuple[str, ...]
    evidence: dict[str, Any] = Field(default_factory=dict)


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
    acceptance: AcceptanceReport | None = None
    bar_search_evidence: tuple[dict[str, Any], ...] = ()
    block_proposals: tuple[BlockGridProposal, ...] = ()
    heatmap_search_boxes: tuple[BoundingBox, ...] = ()


class SavedDetection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1, 2, 3] = 1
    image_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    image_dimensions: tuple[
        Annotated[int, Field(strict=True, gt=0)], Annotated[int, Field(strict=True, gt=0)]
    ]
    detector_version: str
    detector_thresholds: dict[str, Any] = Field(default_factory=dict)
    automation_policy: dict[str, Any] = Field(default_factory=dict)
    analysis_transforms: tuple[dict[str, int], ...] = ()
    hints: DetectionHints
    result: DetectionResult
    associations: tuple[str, ...] = ()
    proposed_config_complete: bool
    missing_extraction_fields: tuple[str, ...]


class LinearScaleFit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    slope: float
    intercept: float
    normalised_positions: tuple[float, ...]
    value_residuals: tuple[float, ...]
    position_residuals_pixels: tuple[float, ...]
    recognised_span: tuple[float, float]
    condition_number: float
    tests_linearity: bool
    endpoint_coverage: bool
    within_residual_limits: bool = True


class LegendTick(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    index: Annotated[int, Field(strict=True, ge=0)]
    side: Literal["left", "right", "above", "below"]
    position: float
    position_source: Literal["stroke", "text_centre"]
    stroke_bbox: tuple[float, float, float, float] | None = None
    text_bbox: tuple[float, float, float, float] | None = None
    raw_text: str = ""
    normalised_text: str = ""
    normalisation_actions: tuple[str, ...] = ()
    parsed_value: float | None = None
    confidence: float | None = None
    token_indices: tuple[int, ...] = ()
    localisation_record_index: int | None = None
    numeric_record_index: int | None = None
    numeric_token_indices: tuple[int, ...] = ()
    ocr_bbox: BoundingBox | None = None
    field_evidence: dict[str, Any] | None = None
    numeric_alternatives: tuple[dict[str, Any], ...] = ()
    status: Literal["suggested", "unresolved", "ambiguous", "reviewed", "excluded"] = "unresolved"
    reasons: tuple[str, ...] = ()


class LegendOCRRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    side: Literal["left", "right", "above", "below"]
    crop_bbox: BoundingBox
    purpose: Literal["localisation", "numeric_line"]
    run: OCRRun
    masked_stroke_boxes: tuple[tuple[float, float, float, float], ...] = ()


class LegendResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    config: LegendConfig
    ticks: tuple[LegendTick, ...]
    ocr_records: tuple[LegendOCRRecord, ...] = ()
    fit: LinearScaleFit | None = None
    status: Literal["review_required", "unresolved", "reviewed"]
    reasons: tuple[str, ...]
    development_eligible: bool = False
    automatic_export_allowed: Literal[False] = False
    review_sha256: str | None = None
    reviewed_anchors: tuple[ScaleAnchor, ...] = ()
    unreviewed_fit: LinearScaleFit | None = None
    unreviewed_reasons: tuple[str, ...] = ()
    calibration_evidence: dict[str, Any] | None = None
