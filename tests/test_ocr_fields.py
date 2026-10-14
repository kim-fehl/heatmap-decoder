"""Padded field geometry, fixed ownership, alignment abstention and shared budgets."""

import numpy as np
import pytest
from PIL import Image, ImageDraw

from heatmap_decoder.labels import CropTransform, assign_candidates
from heatmap_decoder.legend import search_regions, stroke_ticks
from heatmap_decoder.models import (
    GridConfig,
    LabelCandidate,
    LabelsConfig,
    LegendConfig,
    LegendOptions,
    OCRLabelSource,
)
from heatmap_decoder.ocr_fields import aligned_tick_fields


@pytest.mark.parametrize("angle", [0, 90, -90])
@pytest.mark.parametrize("pad", [1, 2])
def test_white_padding_round_trip_original_coordinates(angle, pad):
    transform = CropTransform((10, 20, 50, 100), 3, angle, pad)
    box = (12.0, 23.0, 40.0, 97.0)
    np.testing.assert_allclose(transform.to_original(transform.to_crop(box)), box)
    prepared = transform.prepare(Image.new("RGB", (100, 120), "black"))
    assert prepared.size == transform.crop_size
    assert prepared.getpixel((0, 0)) == 255
    assert prepared.getpixel((prepared.width // 2, prepared.height // 2)) == 0


def test_owned_slot_does_not_spread_a_rounding_overlap_to_neighbours():
    config = GridConfig(
        heatmap_bbox=(0, 0, 50, 61),
        rows=3,
        columns=1,
        labels=LabelsConfig(
            rows=OCRLabelSource(source="ocr", bbox=(60, 0, 100, 61), recognition="slots")
        ),
    )
    # The raster crop rounds outwards; its glyph extends <1px over a fractional slot edge.
    candidate = LabelCandidate("raw glyph", (62, 21, 90, 41), 95, (), slot_index=1)
    result = assign_candidates([candidate], config, "rows")
    assert [s.status for s in result.slots] == ["missing", "accepted", "missing"]
    assert result.slots[1].raw_text == "raw glyph"


def test_owned_slot_off_centre_evidence_stays_unresolved_in_its_own_slot():
    config = GridConfig(
        heatmap_bbox=(0, 0, 50, 60),
        rows=3,
        columns=1,
        labels=LabelsConfig(
            rows=OCRLabelSource(source="ocr", bbox=(60, 0, 100, 60), recognition="slots")
        ),
    )
    candidate = LabelCandidate("neighbour", (62, 39, 90, 45), 99, (), slot_index=1)
    result = assign_candidates([candidate], config, "rows")
    assert result.slots[1].status == "ambiguous"
    assert "outside_owned_slot_field" in result.slots[1].issues
    assert result.slots[2].status == "missing"


def field_fixture(orientation, side, competing=False):
    image = Image.new("RGB", (260, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 229, 39), fill=(80, 110, 130))
    for i, position in enumerate([50, 90, 130, 170, 210]):
        draw.rectangle((position, 40, position + 1, 48), fill="black")
        draw.rectangle((position - 4, 60, position + 4, 80), fill="black")
        if i < 2:
            draw.rectangle((position - 13, 70, position - 8, 72), fill="black")
        if competing:
            draw.rectangle((position - 4, 90, position + 4, 110), fill="black")
    bbox = (30, 30, 230, 40)
    if orientation == "vertical":
        image = image.transpose(Image.Transpose.TRANSPOSE)
        bbox = (30, 30, 40, 230)
        if side == "left":
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            bbox = (180, 30, 190, 230)
    elif side == "above":
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        bbox = (30, 180, 230, 190)
    config = LegendConfig(
        colourbar_bbox=bbox,
        colourbar_orientation=orientation,
        legend=LegendOptions(
            align_tick_fields=True, crop_padding_pixels=2, horizontal_search_pixels=128
        ),
    )
    pixels = np.array(image)
    region = dict(search_regions(pixels, config))[side]
    return pixels, config, region


@pytest.mark.parametrize(
    "orientation,side",
    [("horizontal", "below"), ("horizontal", "above"), ("vertical", "left"), ("vertical", "right")],
)
def test_shared_band_and_all_five_fields_preserve_observed_sign(orientation, side):
    image, config, region = field_fixture(orientation, side)
    strokes = stroke_ticks(image, config, side, region)
    fields, evidence = aligned_tick_fields(image, config, side, region, strokes)
    assert len(fields) == 5
    assert evidence["reason"] is None
    cross = 0 if orientation == "vertical" else 1
    assert len({box[cross] for box in fields.values()}) == 1
    assert len({box[cross + 2] for box in fields.values()}) == 1
    axial = 1 if orientation == "vertical" else 0
    assert fields[50.5][axial] <= 50 - 13
    assert fields[50.5][axial + 2] >= 50 + 5


def test_competing_text_bands_abstain_instead_of_choosing_by_ocr():
    image, config, region = field_fixture("horizontal", "below", competing=True)
    strokes = stroke_ticks(image, config, "below", region)
    fields, evidence = aligned_tick_fields(image, config, "below", region, strokes)
    assert not fields
    assert evidence["reason"] == "missing_or_competing_shared_text_bands"


def test_conflicting_numeric_padding_readings_remain_ambiguous(monkeypatch):
    from heatmap_decoder.legend import inspect_legend
    from heatmap_decoder.models import LegendOCRRecord, OCRRun, OCRToken

    image, config, _ = field_fixture("horizontal", "below")
    config = config.model_copy(
        update={"legend": config.legend.model_copy(update={"retry_white_padding_pixels": (1,)})}
    )

    def crop(image, config, side, box, purpose, budget, strokes=(), *, white_padding_pixels=None):
        white = (
            config.legend.white_padding_pixels
            if white_padding_pixels is None
            else white_padding_pixels
        )
        tokens = ()
        if purpose == "numeric_line":
            transform = CropTransform(box, 3, white_padding_pixels=white)
            tokens = (OCRToken("1" if white == 0 else "2", transform.to_crop(box), 99),)
        run = OCRRun(tokens, "fake", "fake", "eng", ("eng",), purpose, white_padding_pixels=white)
        return LegendOCRRecord(side=side, crop_bbox=box, purpose=purpose, run=run)

    monkeypatch.setattr("heatmap_decoder.legend._ocr_crop", crop)
    result = inspect_legend(image, config)
    ticks = [t for t in result.ticks if t.position_source == "stroke"]
    assert len(ticks) == 5
    assert all(t.parsed_value is None and t.status == "ambiguous" for t in ticks)
    assert all("contradictory_numeric_values" in t.reasons for t in ticks)
    assert all(len(t.numeric_alternatives) == 2 for t in ticks)
    assert not result.automatic_export_allowed


def test_numeric_padding_retries_share_the_existing_call_cap(monkeypatch):
    from heatmap_decoder.legend import inspect_legend
    from heatmap_decoder.models import OCRRun, OCRToken

    image, config, _ = field_fixture("horizontal", "below")
    config = config.model_copy(
        update={
            "legend": config.legend.model_copy(
                update={"max_ticks": 5, "retry_white_padding_pixels": (1,)}
            )
        }
    )
    calls = []

    def engine(crop, settings, **kwargs):
        mode = kwargs["mode"]
        calls.append(mode)
        tokens = () if mode == "localisation" else (OCRToken("1", (3, 3, 9, 15), 95),)
        return OCRRun(tokens, "fake", "fake", "eng", ("eng",), mode)

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", engine)
    result = inspect_legend(image, config)
    assert calls.count("numeric_line") == 5
    assert any("call cap exceeded" in reason for reason in result.reasons)
    assert not result.automatic_export_allowed


def test_recovered_tick_preserves_explicit_decimal_separator(monkeypatch):
    from heatmap_decoder.legend import inspect_legend
    from heatmap_decoder.models import LegendOCRRecord, OCRRun, OCRToken

    image, config, _ = field_fixture("horizontal", "below")
    config = config.model_copy(
        update={
            "legend": config.legend.model_copy(
                update={"retry_white_padding_pixels": (1,), "decimal_separator": ","}
            )
        }
    )

    def crop(image, config, side, box, purpose, budget, strokes=(), *, white_padding_pixels=None):
        white = (
            config.legend.white_padding_pixels
            if white_padding_pixels is None
            else white_padding_pixels
        )
        tokens = ()
        if purpose == "numeric_line" and white == 1:
            transform = CropTransform(box, 3, white_padding_pixels=white)
            tokens = (OCRToken("-1,5", transform.to_crop(box), 99),)
        return LegendOCRRecord(
            side=side,
            crop_bbox=box,
            purpose=purpose,
            run=OCRRun(
                tokens, "fake", "fake", "eng", ("eng",), purpose, white_padding_pixels=white
            ),
        )

    monkeypatch.setattr("heatmap_decoder.legend._ocr_crop", crop)
    result = inspect_legend(image, config)
    ticks = [t for t in result.ticks if t.position_source == "stroke"]
    assert len(ticks) == 5
    assert all(t.parsed_value == -1.5 and t.normalised_text == "-1.5" for t in ticks)
    assert all("explicit_decimal_comma" in t.normalisation_actions for t in ticks)
    assert all(t.status == "suggested" for t in ticks)
    assert not result.development_eligible  # repeated values cannot define a scale


def test_vertical_common_band_cannot_drop_a_separate_minus_component():
    image = Image.new("RGB", (180, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 29, 219), fill=(80, 110, 130))
    for i, position in enumerate([40, 80, 120, 160, 200]):
        draw.rectangle((30, position, 39, position + 1), fill="black")
        draw.rectangle((62, position - 5, 72, position + 5), fill="black")
        if i < 2:
            draw.rectangle((50, position, 55, position + 1), fill="black")
    config = LegendConfig(
        colourbar_bbox=(20, 20, 30, 220),
        colourbar_orientation="vertical",
        legend=LegendOptions(align_tick_fields=True, crop_padding_pixels=2),
    )
    pixels = np.array(image)
    region = dict(search_regions(pixels, config))["right"]
    strokes = stroke_ticks(pixels, config, "right", region)
    fields, evidence = aligned_tick_fields(pixels, config, "right", region, strokes)
    assert len(fields) == 5
    assert evidence["reason"] is None
    assert all(box[0] <= 50 and box[2] >= 73 for box in fields.values())


def test_right_only_gene_padding_keeps_row_height_and_original_boxes(monkeypatch):
    from heatmap_decoder.extract import extract_labels
    from heatmap_decoder.models import OCRConfig, OCRRun, OCRToken

    sizes = []

    def engine(crop, settings, **kwargs):
        sizes.append(crop.size)
        return OCRRun(
            (OCRToken("Gene1", (3, 3, 12, 12), 95),), "fake", "fake", "eng", ("eng",), "text_line"
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", engine)
    config = GridConfig(
        heatmap_bbox=(0, 0, 10, 20),
        rows=2,
        columns=1,
        labels=LabelsConfig(
            rows=OCRLabelSource(
                source="ocr", bbox=(10, 0, 20, 20), recognition="slots", crop_right_padding_pixels=4
            ),
            ocr=OCRConfig(),
        ),
    )
    result = extract_labels(np.full((20, 30, 3), 255, dtype=np.uint8), config)
    assert sizes == [(42, 30), (42, 30)]
    assert [r[2] for r in result.slot_ocr_runs] == [(10, 0, 24, 10), (10, 10, 24, 20)]
    assert [a.bbox for a in result.axes[0].tokens] == [(11, 1, 14, 4), (11, 11, 14, 14)]


def test_raw_numeric_comparison_preserves_readable_zero_under_shared_budget(monkeypatch):
    from heatmap_decoder.legend import inspect_legend
    from heatmap_decoder.models import OCRRun, OCRToken

    image, config, _ = field_fixture("horizontal", "below")
    config = config.model_copy(
        update={"legend": config.legend.model_copy(update={"retry_unconstrained_numeric": True})}
    )
    calls = []

    def engine(crop, settings, **kwargs):
        mode = kwargs["mode"]
        constrained = kwargs["numeric_characters_only"]
        calls.append((mode, constrained))
        tokens = ()
        if mode == "numeric_line" and not constrained:
            tokens = (OCRToken("0", (3, 3, 9, 15), 90),)
        return OCRRun(tokens, "fake", "fake", "eng", ("eng",), mode)

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", engine)
    result = inspect_legend(image, config)
    ticks = [t for t in result.ticks if t.position_source == "stroke"]
    assert len(ticks) == 5
    assert all(t.parsed_value == 0 and t.raw_text == "0" for t in ticks)
    assert any(t.status == "suggested" for t in ticks)
    assert not result.development_eligible  # identical zero values contradict calibration
    assert calls.count(("numeric_line", False)) == 5
    assert len(result.ocr_records) == 12
    assert not result.automatic_export_allowed


def test_extra_right_padding_is_bounded_and_row_only():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OCRLabelSource(
            source="ocr", bbox=(0, 0, 20, 20), recognition="slots", crop_right_padding_pixels=17
        )
    with pytest.raises(ValidationError):
        LabelsConfig(
            columns=OCRLabelSource(
                source="ocr", bbox=(0, 0, 20, 20), recognition="slots", crop_right_padding_pixels=1
            )
        )
    with pytest.raises(ValidationError):
        OCRLabelSource(
            source="ocr", bbox=(0, 0, 20, 20), recognition="sparse", crop_right_padding_pixels=1
        )
