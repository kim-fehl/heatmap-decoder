"""Generate positional names from explicitly supplied categorical legend swatches."""

from collections import Counter
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calibration import to_lab
from .models import AxisSegment, BoundingBox, RGBArray, validate_segments
from .sampling import validate_rgb


class Swatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    label: Annotated[str, Field(strict=True, min_length=1)]
    bbox: BoundingBox

    @model_validator(mode="after")
    def validate_swatch(self) -> "Swatch":
        if not self.label.strip() or any(c in self.label for c in "\n\r\t"):
            raise ValueError("category labels must be nonempty single-line strings")
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("swatch bbox must have positive extent")
        return self


class CategoryTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Annotated[str, Field(strict=True, min_length=1)]
    bbox: BoundingBox
    swatches: Annotated[tuple[Swatch, ...], Field(min_length=2)]

    @model_validator(mode="after")
    def validate_track(self) -> "CategoryTrack":
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("track bbox must have positive extent")
        if len({s.label for s in self.swatches}) != len(self.swatches):
            raise ValueError("category labels must be unique within each track")
        return self


class GroupLabelsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    axis: Literal["rows", "columns"] = "columns"
    start: Annotated[int, Field(strict=True, ge=0)]
    end: Annotated[int, Field(strict=True, gt=0)]
    count: Annotated[int, Field(strict=True, gt=0, le=10000)]
    segments: tuple[AxisSegment, ...] | None = None
    tracks: Annotated[tuple[CategoryTrack, ...], Field(min_length=1)]
    max_colour_distance: Annotated[float, Field(gt=0)] = 12.0
    min_colour_margin: Annotated[float, Field(gt=0)] = 3.0
    minimum_match_fraction: Annotated[float, Field(gt=0.5, le=1)] = 0.8

    @model_validator(mode="after")
    def validate_request(self) -> "GroupLabelsRequest":
        if self.end <= self.start:
            raise ValueError("axis end must exceed start")
        validate_segments(self.segments, self.start, self.end, self.count, self.axis)
        if len({t.name for t in self.tracks}) != len(self.tracks):
            raise ValueError("track names must be unique")
        coordinate = 0 if self.axis == "columns" else 1
        for track in self.tracks:
            if track.bbox[coordinate] > self.start or track.bbox[coordinate + 2] < self.end:
                raise ValueError("each track must cover the supplied axis extent")
        return self


def _crop(image: RGBArray, bbox: tuple[int, int, int, int]) -> RGBArray:
    left, top, right, bottom = bbox
    if not (0 <= left < right <= image.shape[1] and 0 <= top < bottom <= image.shape[0]):
        raise ValueError("categorical region lies outside image")
    return image[top:bottom, left:right]


def infer_group_labels(image: RGBArray, request: GroupLabelsRequest) -> dict[str, Any]:
    """Classify inset pixels, reject unknown/mixed colors, retain all track evidence.

    Names identify display position within a group, never original sample identity.
    No OCR or legend discovery is performed: labels/swatches are reviewed inputs.
    """
    validate_rgb(image)
    intervals: list[tuple[float, float]] = []
    segments = request.segments or (
        AxisSegment(start=float(request.start), end=float(request.end), count=request.count),
    )
    for segment in segments:
        edges = np.linspace(segment.start, segment.end, segment.count + 1)
        intervals.extend(zip(edges[:-1], edges[1:], strict=True))
    references = []
    for track in request.tracks:
        _crop(image, track.bbox)
        colours = []
        for swatch in track.swatches:
            pixels = _crop(image, swatch.bbox).reshape(-1, image.shape[2])
            if image.shape[2] == 4:
                pixels = pixels[pixels[:, 3] == 255]
            if not len(pixels):
                raise ValueError("legend swatch contains no opaque pixels")
            colours.append(np.median(pixels[:, :3], axis=0))
        labs = to_lab(np.asarray(colours))
        pairwise = np.linalg.norm(labs[:, None] - labs[None, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)
        if np.min(pairwise) < request.min_colour_margin:
            raise ValueError("legend swatches are too similar to distinguish")
        references.append(labs)
    records: list[dict[str, Any]] = []
    combinations = []
    for index, (start, end) in enumerate(intervals):
        evidence = []
        categories = []
        for track, labs in zip(request.tracks, references, strict=True):
            x, y, r, b = track.bbox
            # Avoid strip borders and inter-column antialiasing; enough width remains
            # to detect a mixed strip rather than trusting a single center pixel.
            if request.axis == "columns":
                x, r = (
                    int(np.ceil(start + (end - start) * 0.2)),
                    int(np.ceil(end - (end - start) * 0.2)),
                )
                y, b = y + (b - y) // 4, b - (b - y) // 4
            else:
                y, b = (
                    int(np.ceil(start + (end - start) * 0.2)),
                    int(np.ceil(end - (end - start) * 0.2)),
                )
                x, r = x + (r - x) // 4, r - (r - x) // 4
            pixels = _crop(image, (x, y, r, b)).reshape(-1, image.shape[2])
            distances = np.linalg.norm(to_lab(pixels[:, :3].astype(float))[:, None] - labs, axis=2)
            nearest = np.argmin(distances, axis=1)
            ordered = np.sort(distances, axis=1)
            valid = (ordered[:, 0] <= request.max_colour_distance) & (
                ordered[:, 1] - ordered[:, 0] >= request.min_colour_margin
            )
            if image.shape[2] == 4:
                valid &= pixels[:, 3] == 255
            fractions = [float(np.mean(valid & (nearest == i))) for i in range(len(labs))]
            winner = int(np.argmax(fractions))
            accepted = fractions[winner] >= request.minimum_match_fraction
            categories.append(track.swatches[winner].label if accepted else None)
            evidence.append(
                {
                    "track": track.name,
                    "sample_bbox": [x, y, r, b],
                    "category": categories[-1],
                    "status": "matched" if accepted else "unresolved",
                    "category_fractions": {
                        s.label: f for s, f in zip(track.swatches, fractions, strict=True)
                    },
                }
            )
        combinations.append(tuple(categories))
        records.append({"index": index, "tracks": evidence, "generated_name": None})
    complete = all(all(c is not None for c in group) for group in combinations)
    if complete:
        ordinals: Counter[tuple[str | None, ...]] = Counter()
        for record, group in zip(records, combinations, strict=True):
            ordinals[group] += 1
            record["generated_name"] = "__".join(str(c) for c in group) + f"__{ordinals[group]:02d}"
        if len({r["generated_name"] for r in records}) != len(records):
            raise ValueError("category text produces colliding names; use distinct legend labels")
    return {
        "axis": request.axis,
        "complete": complete,
        "name_kind": "category_combination_plus_display_ordinal",
        "original_sample_ids_recovered": False,
        "legend_labels_are_reviewed_inputs": True,
        "records": records,
    }
