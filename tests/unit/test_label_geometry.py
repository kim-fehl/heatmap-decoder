"""Label geometry contracts and fixtures."""

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from heatmap_decoder.labels import CropTransform
from heatmap_decoder.models import GridConfig, LabelsConfig, OCRLabelSource
from tests.support.angle_fixtures import ANGLES
from tests.support.rotation_fixtures import config


@pytest.mark.parametrize("angle", ANGLES + [37.25, -22.7, 150, -150, 180, 370])
@pytest.mark.parametrize("factor", [1, 3])
def test_affine_point_polygon_round_trips(angle, factor):
    transform = CropTransform((17, 23, 142, 78), factor, angle, 2)
    polygon = ((17.5, 23.5), (100, 25.2), (130.6, 60), (20, 70.2))
    assert np.allclose(
        transform.polygon_to_original(transform.polygon_to_crop(polygon)), polygon, atol=1e-10
    )
    for p in polygon:
        assert transform.point_to_original(transform.point_to_crop(p)) == pytest.approx(
            p, abs=1e-10
        )
    metadata = transform.metadata()
    assert len(metadata["forward_affine"]) == len(metadata["inverse_affine"]) == 6
    if angle not in (0, 90, -90):
        assert "expanded" in metadata["interpolation"]


@pytest.mark.parametrize(
    "value", [True, False, float("nan"), float("inf"), -float("inf"), "45", None]
)
def test_angles_reject_invalid(value):
    with pytest.raises(ValidationError):
        OCRLabelSource(source="ocr", bbox=(0, 0, 10, 20), text_rotation_degrees=value)
    with pytest.raises(ValueError):
        CropTransform((0, 0, 10, 20), text_rotation_degrees=value)


@pytest.mark.parametrize(
    "value,expected", [(90.0, 90), (370, 10), (-190, 170), (180, -180), (720, 0), (37.25, 37.25)]
)
def test_angle_normalisation(value, expected):
    assert (
        OCRLabelSource(
            source="ocr", bbox=(0, 0, 10, 20), text_rotation_degrees=value
        ).text_rotation_degrees
        == expected
    )


def test_gene_angles_remain_unsupported():
    with pytest.raises(ValidationError, match="columns only"):
        LabelsConfig(rows={"source": "ocr", "bbox": (0, 0, 20, 30), "text_rotation_degrees": 15})


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


@pytest.mark.parametrize("angle", [True, False, float("inf"), float("nan"), "90", None])
def test_rotation_schema_rejects_unsupported_types_and_values(angle):
    with pytest.raises(ValidationError, match="text_rotation_degrees"):
        OCRLabelSource(source="ocr", bbox=(0, 0, 10, 20), text_rotation_degrees=angle)


def test_default_and_column_only_scope():
    assert OCRLabelSource(source="ocr", bbox=(0, 0, 10, 20)).text_rotation_degrees == 0
    payload = config().model_dump()
    payload["labels"]["rows"] = payload["labels"].pop("columns")
    with pytest.raises(ValidationError, match="columns only"):
        GridConfig.model_validate(payload)
