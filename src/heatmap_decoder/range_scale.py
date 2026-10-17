"""Recover reviewed expression ranges without inventing point values inside them."""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .extract import extract
from .models import Axis, CellFlag, ExtractionConfig, RGBArray


class RangeBand(BaseModel):
    """Half-open normalized bar span (the last sample at 1 is included)."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    label: Annotated[str, Field(min_length=1)]
    start: Annotated[float, Field(ge=0, le=1)]
    end: Annotated[float, Field(ge=0, le=1)]
    lower: float | None = None
    upper: float | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = False

    @model_validator(mode="after")
    def validate_band(self) -> "RangeBand":
        if not self.label.strip() or any(c in self.label for c in "\r\n\t"):
            raise ValueError("range labels must be nonempty single-line strings")
        if self.end <= self.start:
            raise ValueError("range band must have positive extent")
        if self.lower is not None and self.upper is not None and self.upper <= self.lower:
            raise ValueError("range upper bound must exceed lower bound")
        return self


class RangeExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    position_config: ExtractionConfig
    quantity: Annotated[str, Field(min_length=1)]
    bands: Annotated[tuple[RangeBand, ...], Field(min_length=2, max_length=32)]

    @model_validator(mode="after")
    def validate_ranges(self) -> "RangeExtractionRequest":
        config = self.position_config
        if (
            config.colourbar_start_value != 0
            or config.colourbar_end_value != 1
            or config.scale is not None
        ):
            raise ValueError("range extraction requires a 0..1 positional config without a scale")
        if len({b.label for b in self.bands}) != len(self.bands):
            raise ValueError("range labels must be unique")
        if any(a.end > b.start for a, b in zip(self.bands, self.bands[1:])):
            raise ValueError("range bands must be ordered and nonoverlapping")
        return self


def extract_ranges(
    image: RGBArray,
    request: RangeExtractionRequest,
    *,
    file_labels: Mapping[Axis, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Use all near-colour candidates, including disconnected/plateau matches.

    Positional ambiguity within one reviewed band still identifies that range.
    Candidates crossing a band boundary, uncovered spans, poor colour matches,
    and missing cells remain unresolved. Numeric bounds are user-supplied ranges,
    not statistical intervals or original expression measurements.
    """
    result = extract(image, request.position_config, file_labels=file_labels)
    config = request.position_config
    coordinate = 1 if config.colourbar_orientation == "vertical" else 0
    length = config.colourbar_bbox[coordinate + 2] - config.colourbar_bbox[coordinate]
    # Map every actual sampled position; gaps never acquire a range by proximity.
    owners = [
        next(
            (
                i
                for i, b in enumerate(request.bands)
                if b.start <= j / (length - 1) < b.end or (j == length - 1 and b.end == 1)
            ),
            None,
        )
        for j in range(length)
    ]
    records: list[dict[str, Any]] = []
    for index, evidence in enumerate(result.colour_evidence or ()):
        flags = int(result.flags.flat[index])
        candidates = (
            {
                owners[j]
                for component in evidence.components
                if evidence is not None
                for j in range(component.first_index, component.last_index + 1)
            }
            if evidence is not None
            else set()
        )
        matched = sorted(i for i in candidates if i is not None)
        status = (
            "missing"
            if flags & CellFlag.MISSING
            else "poor_colour_match"
            if flags & CellFlag.POOR_MATCH
            else "uncovered_range"
            if None in candidates or not candidates
            else "range_ambiguous"
            if len(matched) != 1
            else "matched"
        )
        band = request.bands[matched[0]] if status == "matched" else None
        distance = float(result.colour_distance.flat[index]) if evidence is not None else None
        records.append(
            {
                "row": index // config.columns,
                "column": index % config.columns,
                "status": status,
                "range": band.model_dump(mode="json") if band else None,
                "candidate_ranges": [request.bands[i].label for i in matched],
                "colour_distance": distance,
                "positional_flags": flags,
            }
        )
    return {
        "shape": [config.rows, config.columns],
        "quantity": request.quantity,
        "value_kind": "reviewed_range",
        "point_values_recovered": False,
        "statistical_intervals": False,
        "complete": all(r["status"] == "matched" for r in records),
        "bands": [b.model_dump(mode="json") for b in request.bands],
        "records": records,
    }
