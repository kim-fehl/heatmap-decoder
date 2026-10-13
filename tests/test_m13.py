"""Exact half-open geometry and upright grouping without an OCR dependency."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from heatmap_decoder.extract import extract_labels
from heatmap_decoder.io import label_provenance
from heatmap_decoder.labels import CropTransform, assign_candidates, group_tokens, transform_tokens
from heatmap_decoder.models import GridConfig, OCRLabelSource, OCRRun, OCRToken


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


@pytest.mark.parametrize("angle", [0, 90, -90])
@pytest.mark.parametrize("factor", [1, 3, 5])
def test_exact_half_open_round_trips(angle, factor):
    transform = CropTransform((17, 23, 58, 96), factor, angle)
    for box in [
        (17, 23, 58, 96),
        (17, 23, 18, 24),
        (57, 95, 58, 96),
        (20, 30, 45, 80),
        (17.5, 23.5, 58, 96),
    ]:
        assert transform.to_original(transform.to_crop(box)) == pytest.approx(box)
    width, height = transform.crop_size
    for box in [(0, 0, width, height), (0, 0, 1, 1), (width - 1, height - 1, width, height)]:
        assert transform.to_crop(transform.to_original(box)) == pytest.approx(box)


@pytest.mark.parametrize("angle", [90, -90])
def test_transpose_pixels_and_border_boxes(angle):
    pixels = np.arange(35, dtype=np.uint8).reshape(5, 7)
    image = Image.fromarray(pixels)
    transform = CropTransform((0, 0, 7, 5), 1, angle)
    expected = np.rot90(pixels, -1 if angle == 90 else 1)
    np.testing.assert_array_equal(np.asarray(transform.prepare(image)), expected)
    for y in range(5):
        for x in range(7):
            u, v, _, _ = transform.to_crop((x, y, x + 1, y + 1))
            assert expected[int(v), int(u)] == pixels[y, x]


@pytest.mark.parametrize("angle", [True, False, 45, 180, 90.0, "90", None])
def test_rotation_schema_rejects_unsupported_types_and_values(angle):
    with pytest.raises(ValidationError, match="text_rotation_degrees"):
        OCRLabelSource(source="ocr", bbox=(0, 0, 10, 20), text_rotation_degrees=angle)


def test_default_and_column_only_scope():
    assert OCRLabelSource(source="ocr", bbox=(0, 0, 10, 20)).text_rotation_degrees == 0
    payload = config().model_dump()
    payload["labels"]["rows"] = payload["labels"].pop("columns")
    with pytest.raises(ValidationError, match="columns only"):
        GridConfig.model_validate(payload)


@pytest.mark.parametrize("angle", [90, -90])
@pytest.mark.parametrize("recognition", ["sparse", "slots"])
def test_group_upright_assign_original_x_and_keep_run_links(monkeypatch, angle, recognition):
    cfg = config(angle, recognition)
    original = []
    # Two separated words in each original vertical label; +/- signs reverse upright line order.
    for i in range(3):
        x = 24 + 40 * i
        boxes = [(x, 25, x + 10, 45), (x, 49, x + 10, 75)]
        if angle == 90:
            boxes.reverse()
        original.extend(
            [
                OCRToken(f"Sample{i}", boxes[0], 95, (1, 1, 1)),
                OCRToken("A_b-C", boxes[1], 94, (1, 1, 1)),
            ]
        )
    calls = []

    def recognize(image, settings, *, mode="localisation", timeout_seconds=30):
        index = len(calls)
        bbox = cfg.labels.columns.bbox
        if recognition == "slots":
            bbox = (10 + 40 * index, 20, 50 + 40 * index, 100)
            selected = original[index * 2 : index * 2 + 2]
        else:
            selected = original
        transform = CropTransform(bbox, settings.upscale_factor, angle)
        assert image.size == transform.crop_size
        calls.append(mode)
        tokens = tuple(replace(t, bbox=transform.to_crop(t.bbox)) for t in selected)
        return OCRRun(tokens, "fake", "fake", "eng", ("eng",), mode)

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    result = extract_labels(np.full((160, 140, 3), 255, dtype=np.uint8), cfg)
    columns = result.axes[1]
    assert [s.final_text for s in columns.slots] == [f"Sample{i} A_b-C" for i in range(3)]
    assert len(columns.tokens) == 6
    assert all(len(c.token_indices) == 2 for c in columns.candidates)
    assert columns.tokens == tuple(original)
    evidence = label_provenance(
        result, input_filename="fake.png", input_sha256="x", image_shape=(160, 140, 3)
    )
    transforms = (
        list(evidence["ocr"]["transforms"].values())
        if recognition == "sparse"
        else [r["transform"] for r in evidence["ocr"]["slot_runs"]]
    )
    assert all(t["text_rotation_degrees"] == angle for t in transforms)
    assert all(t["grouping_frame"] == "upright OCR pixels" for t in transforms)


@pytest.mark.parametrize("angle", [90, -90])
def test_rotated_missing_duplicate_and_span_behaviour(angle):
    cfg = config(angle)
    transform = CropTransform(cfg.labels.columns.bbox, 3, angle)
    originals = [
        OCRToken("Duplicate", (24, 25, 34, 70), 95, (1, 1, 1)),
        OCRToken("Duplicate", (104, 25, 114, 70), 95, (1, 1, 1)),
    ]
    tokens = tuple(replace(t, bbox=transform.to_crop(t.bbox)) for t in originals)
    candidates = group_tokens(tokens, cfg, "columns", transform=transform)
    result = assign_candidates(
        candidates, cfg, "columns", tokens=transform_tokens(tokens, transform)
    )
    assert result.slots[1].status == "missing"
    assert result.slots[0].duplicate and result.slots[2].duplicate
    spanning = OCRToken("span", transform.to_crop((40, 25, 65, 70)), 95, (1, 1, 1))
    result = assign_candidates(
        group_tokens([spanning], cfg, "columns", transform=transform), cfg, "columns"
    )
    assert result.slots[0].status == result.slots[1].status == "ambiguous"
