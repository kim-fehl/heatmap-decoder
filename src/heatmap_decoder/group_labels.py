"""Sample group metadata from reviewed swatches, lines, or labeled rectangles."""

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Annotated, Any, Literal

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calibration import to_lab
from .models import (
    Axis,
    AxisSegment,
    BoundingBox,
    GridConfig,
    OCRConfig,
    RGBArray,
    validate_segments,
)
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


class GroupRegion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    bbox: BoundingBox
    text_bbox: BoundingBox | None = None
    label: Annotated[str, Field(strict=True, min_length=1)] | None = None

    @model_validator(mode="after")
    def validate_region(self) -> "GroupRegion":
        for box in (self.bbox, self.text_bbox):
            if box is not None and (box[2] <= box[0] or box[3] <= box[1]):
                raise ValueError("group region must have positive extent")
        if self.label is not None and (
            not self.label.strip() or any(c in self.label for c in "\n\r\t")
        ):
            raise ValueError("group labels must be single-line strings")
        return self


class CategoryTrack(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Annotated[str, Field(strict=True, min_length=1)] | None = None
    bbox: BoundingBox
    source: Literal["swatches", "rectangles", "lines"] = "swatches"
    swatches: tuple[Swatch, ...] = ()
    regions: tuple[GroupRegion, ...] = ()

    @model_validator(mode="after")
    def validate_track(self) -> "CategoryTrack":
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("track bbox must have positive extent")
        if self.source == "swatches":
            if len(self.swatches) < 2 or self.regions:
                raise ValueError("swatch tracks require at least two swatches and no regions")
            if len({s.label for s in self.swatches}) != len(self.swatches):
                raise ValueError("category labels must be unique within each track")
        elif not self.regions or self.swatches:
            raise ValueError("line/rectangle tracks require regions and no swatches")
        if self.source == "lines" and any(
            r.text_bbox is None and r.label is None for r in self.regions
        ):
            raise ValueError("unreviewed line groups require a text_bbox")
        return self


class GroupLabelsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    axis: Literal["rows", "columns"] = "columns"
    entity_kind: Literal["sample", "gene", "unspecified"] | None = None
    start: Annotated[int, Field(strict=True, ge=0)]
    end: Annotated[int, Field(strict=True, gt=0)]
    count: Annotated[int, Field(strict=True, gt=0, le=10000)]
    segments: tuple[AxisSegment, ...] | None = None
    tracks: Annotated[tuple[CategoryTrack, ...], Field(min_length=1)]
    grid: GridConfig | None = None
    ocr: OCRConfig = Field(default_factory=lambda: OCRConfig(engine="rapidocr", upscale_factor=1))
    max_colour_distance: Annotated[float, Field(gt=0)] = 12.0
    min_colour_margin: Annotated[float, Field(gt=0)] = 3.0
    minimum_match_fraction: Annotated[float, Field(gt=0.5, le=1)] = 0.8
    indistinguishable_swatches: Literal["error", "unresolved"] = "error"

    @model_validator(mode="after")
    def validate_request(self) -> "GroupLabelsRequest":
        if self.end <= self.start:
            raise ValueError("axis end must exceed start")
        validate_segments(self.segments, self.start, self.end, self.count, self.axis)
        if len({t.name or f"track_{i + 1}" for i, t in enumerate(self.tracks)}) != len(self.tracks):
            raise ValueError("track names must be unique")
        if sum(len(t.regions) for t in self.tracks) > 128:
            raise ValueError("at most 128 group regions per request")
        if self.grid is not None:
            coordinate = 0 if self.axis == "columns" else 1
            if (self.grid.heatmap_bbox[coordinate], self.grid.heatmap_bbox[coordinate + 2]) != (
                self.start,
                self.end,
            ):
                raise ValueError("sample label grid must match group axis extent")
            if getattr(self.grid, self.axis) != self.count:
                raise ValueError("sample label grid must match group count")
            default_segment = (
                AxisSegment(start=float(self.start), end=float(self.end), count=self.count),
            )
            if (getattr(self.grid, f"{self.axis}_segments") or default_segment) != (
                self.segments or default_segment
            ):
                raise ValueError("sample label grid must match group axis segments")
            if self.grid.labels is None or getattr(self.grid.labels, self.axis) is None:
                raise ValueError("grid must request sample labels on the group axis")
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


def _region_track(
    image: RGBArray,
    request: GroupLabelsRequest,
    track: CategoryTrack,
    intervals: list[tuple[float, float]],
    deadline: float,
    pixels_remaining: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Snap shape geometry, independently of text width; retain raw OCR evidence."""
    from .ocr import recognize

    coordinate = 0 if request.axis == "columns" else 1
    regions = sorted(track.regions, key=lambda r: r.bbox[coordinate])
    pitch = float(np.median([end - start for start, end in intervals]))
    observed: list[dict[str, Any]] = []
    for region in regions:
        _crop(image, region.bbox)
        text, confidence, polarity, run = None, None, None, None
        if region.label is None:
            box = region.text_bbox or region.bbox
            crop = _crop(image, box)[:, :, :3]
            border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]))
            background = np.median(border, axis=0)
            contrast = np.linalg.norm(crop.astype(float) - background, axis=2) > 70
            dark, light = (crop.max(axis=2) < 120) & contrast, (crop.min(axis=2) > 200) & contrast
            polarity = "light" if light.sum() > dark.sum() else "dark"
            # Select observed text polarity, excluding colored JPEG fringes.
            prepared = ImageOps.expand(
                Image.fromarray(
                    np.where(light if polarity == "light" else dark, 0, 255).astype(np.uint8)
                ),
                border=8,
                fill=255,
            )
            if track.source == "lines":
                # Printed rules locate spans; their separate labels are one
                # ordinary text line. Preserve antialiasing on white paper.
                prepared = ImageOps.expand(Image.fromarray(crop).convert("L"), border=8, fill=255)
            factor = request.ocr.upscale_factor
            if (
                prepared.width * prepared.height * factor * factor > pixels_remaining[0]
                or time.monotonic() >= deadline
            ):
                observed.append(
                    {"bbox": region.bbox, "label": None, "status": "ocr_budget_exhausted"}
                )
                continue
            prepared = prepared.resize(
                (prepared.width * factor, prepared.height * factor), Image.Resampling.BICUBIC
            )
            pixels_remaining[0] -= prepared.width * prepared.height
            evidence = recognize(
                prepared,
                request.ocr,
                mode="text_line" if track.source == "lines" else "localisation",
                timeout_seconds=min(30, max(0.001, deadline - time.monotonic())),
            )
            run = asdict(evidence)
            tokens = [t for t in evidence.tokens if t.text.strip()]
            text = " ".join(t.text.strip() for t in tokens) or None
            confidence = min((t.confidence for t in tokens), default=None)
        accepted = region.label is not None or (
            text is not None and confidence is not None and confidence >= request.ocr.min_confidence
        )
        observed.append(
            {
                "bbox": region.bbox,
                "text_bbox": region.text_bbox or region.bbox,
                "label": region.label if region.label is not None else text if accepted else None,
                "raw_text": text,
                "confidence": confidence,
                "text_polarity": polarity,
                "status": "reviewed"
                if region.label is not None
                else "ocr_candidate"
                if accepted
                else "unresolved",
                "ocr": run,
                "ocr_transform": {
                    "crop_bbox": region.text_bbox or region.bbox,
                    "upscale_factor": request.ocr.upscale_factor,
                    "white_padding_pixels": 8,
                    "preprocessing": "grayscale" if track.source == "lines" else "polarity_mask",
                }
                if region.label is None
                else None,
            }
        )
    reasons: list[str] = []
    boundaries: list[float] = [float(request.start)]
    if pitch < 2:
        reasons.append("subpixel_group_boundaries_require_reviewed_counts")
    if (
        regions[0].bbox[coordinate] - request.start > pitch
        or request.end - regions[-1].bbox[coordinate + 2] > pitch
    ):
        reasons.append("group_regions_do_not_establish_outer_extent")
    edges = np.asarray([start for start, _ in intervals] + [intervals[-1][1]])
    for a, b in zip(regions, regions[1:]):
        gap = b.bbox[coordinate] - a.bbox[coordinate + 2]
        midpoint = (a.bbox[coordinate + 2] + b.bbox[coordinate]) / 2
        index = int(np.argmin(abs(edges - midpoint)))
        if (
            gap < 0
            or gap > pitch * (2 if track.source == "lines" else 0.7)
            or abs(edges[index] - midpoint) > 0.35 * pitch
        ):
            reasons.append("ambiguous_group_boundary")
        boundaries.append(float(edges[index]))
    boundaries.append(request.end)
    if any(b <= a for a, b in zip(boundaries, boundaries[1:])):
        reasons.append("empty_or_overlapping_group_span")
    assigned: list[dict[str, Any]] = []
    for start, end in intervals:
        indices = [
            i for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])) if a <= start and end <= b
        ]
        region_index = indices[0] if len(indices) == 1 and not reasons else None
        category = observed[region_index]["label"] if region_index is not None else None
        assigned.append(
            {
                "category": category,
                "status": "matched" if category is not None else "unresolved",
                "region_index": region_index,
                "reasons": list(dict.fromkeys(reasons)),
            }
        )
    for i, entry in enumerate(observed):
        entry["inferred_span"] = [boundaries[i], boundaries[i + 1]] if not reasons else None
    return assigned, observed


def infer_group_labels(
    image: RGBArray,
    request: GroupLabelsRequest,
    *,
    file_labels: Mapping[Axis, Sequence[str]] | None = None,
) -> dict[str, Any]:
    """Classify inset pixels, reject unknown/mixed colors, retain all track evidence.

    Names identify display position within a group, never original sample identity.
    Optional named-sample OCR uses the existing label extractor. Group categories
    remain independent metadata and never replace observed sample identities.
    """
    validate_rgb(image)
    intervals: list[tuple[float, float]] = []
    segments = request.segments or (
        AxisSegment(start=float(request.start), end=float(request.end), count=request.count),
    )
    for segment in segments:
        edges = np.linspace(segment.start, segment.end, segment.count + 1)
        intervals.extend(zip(edges[:-1], edges[1:], strict=True))
    sample_slots = None
    if request.grid is not None:
        from .extract import extract_labels

        labels = extract_labels(image, request.grid, file_labels=file_labels)
        sample_slots = next(axis.slots for axis in labels.axes if axis.axis == request.axis)
    references = []
    region_assignments: dict[int, list[dict[str, Any]]] = {}
    track_evidence: list[dict[str, Any]] = []
    deadline = time.monotonic() + 60
    pixels_remaining = [16000000]
    for track_index, track in enumerate(request.tracks):
        _crop(image, track.bbox)
        if track.source != "swatches":
            assignments, observations = _region_track(
                image, request, track, intervals, deadline, pixels_remaining
            )
            region_assignments[track_index] = assignments
            track_evidence.append(
                {
                    "track": track.name or f"track_{track_index + 1}",
                    "source": track.source,
                    "regions": observations,
                }
            )
            references.append(np.empty((0, 3)))
            continue
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
        if (
            np.min(pairwise) < request.min_colour_margin
            and request.indistinguishable_swatches == "error"
        ):
            raise ValueError("legend swatches are too similar to distinguish")
        references.append(labs)
    records: list[dict[str, Any]] = []
    combinations = []
    for index, (start, end) in enumerate(intervals):
        evidence = []
        categories = []
        for track_index, (track, labs) in enumerate(zip(request.tracks, references, strict=True)):
            name = track.name or f"track_{track_index + 1}"
            if track.source != "swatches":
                record = {"track": name, **region_assignments[track_index][index]}
                evidence.append(record)
                categories.append(record["category"])
                continue
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
            candidate_pixels = (distances <= request.max_colour_distance) & (
                distances <= ordered[:, :1] + request.min_colour_margin
            )
            if image.shape[2] == 4:
                candidate_pixels &= pixels[:, 3:4] == 255
            candidate_fractions = np.mean(candidate_pixels, axis=0)
            winner = int(np.argmax(fractions))
            accepted = fractions[winner] >= request.minimum_match_fraction
            categories.append(track.swatches[winner].label if accepted else None)
            evidence.append(
                {
                    "track": name,
                    "sample_bbox": [x, y, r, b],
                    "category": categories[-1],
                    "status": "matched" if accepted else "unresolved",
                    "category_fractions": {
                        s.label: f for s, f in zip(track.swatches, fractions, strict=True)
                    },
                    "candidate_categories": [
                        s.label
                        for s, fraction in zip(track.swatches, candidate_fractions, strict=True)
                        if fraction >= request.minimum_match_fraction
                    ],
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
    entity_kind = request.entity_kind or ("sample" if request.axis == "columns" else "unspecified")
    for index, record in enumerate(records):
        sample = sample_slots[index] if sample_slots is not None else None
        record["entity_name"] = sample.final_text if sample is not None else None
        record["entity_raw_text"] = sample.raw_text if sample is not None else None
        record["entity_status"] = sample.status if sample is not None else "not_requested"
        # Compatibility fields describe samples only, never row gene identities.
        record["sample_name"] = record["entity_name"] if entity_kind == "sample" else None
        record["sample_raw_text"] = record["entity_raw_text"] if entity_kind == "sample" else None
        record["sample_status"] = (
            record["entity_status"] if entity_kind == "sample" else "not_applicable"
        )
        record["display_name"] = (
            sample.final_text if sample is not None else record["generated_name"]
        )
    if sample_slots is not None:
        complete = complete and all(
            s.status in {"accepted", "manual"} and s.final_text for s in sample_slots
        )
    return {
        "axis": request.axis,
        "entity_kind": entity_kind,
        "complete": complete,
        "name_kind": f"visible_{entity_kind}_names"
        if sample_slots is not None
        else "category_combination_plus_display_ordinal",
        "original_sample_ids_recovered": False,
        "visible_sample_names_recovered": entity_kind == "sample"
        and sample_slots is not None
        and all(r["sample_name"] for r in records),
        "visible_entity_names_recovered": sample_slots is not None
        and all(r["entity_name"] for r in records),
        "legend_labels_are_reviewed_inputs": all(
            t.source == "swatches" or all(r.label is not None for r in t.regions)
            for t in request.tracks
        ),
        "track_names": [t.name or f"track_{i + 1}" for i, t in enumerate(request.tracks)],
        "track_evidence": track_evidence,
        "records": records,
    }
