"""Quality contracts and fixtures."""

import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image, ImageFilter

from heatmap_decoder import CellFlag, ExtractionConfig, extract, extract_file
from heatmap_decoder.calibration import match_with_quality
from heatmap_decoder.diagnostics import diagnostic_image
from tests.support.quality import reference


def test_mask_and_partial_coverage():
    image, cfg = reference()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[:8, :8] = True
    mask[2:4, 10:14] = True  # half of the second cell's sampled interior
    image[2:4, 10:14] = [255, 0, 255]  # ignored masked outliers
    result = extract(image, cfg, mask=mask)
    assert result.flags[0, 0] == CellFlag.MISSING
    assert np.isnan(result.values[0, 0]) and np.isnan(result.colour_distance[0, 0])
    assert np.isnan(result.sampled_rgb[0, 0]).all()
    assert result.valid_fraction[0, 1] == 0.5
    assert result.values[0, 1] == 32
    mask[4, 10] = True
    assert extract(image, cfg, mask=mask).flags[0, 1] == CellFlag.MISSING


def test_alpha_and_sentinel_missing_data():
    image, cfg = reference()
    rgba = np.concatenate((image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)), axis=2)
    rgba[:8, :8, 3] = 0
    result = extract(rgba, cfg)
    assert np.isnan(result.values[0, 0])
    image[:8, :8] = [254, 0, 254]
    cfg = ExtractionConfig(**(cfg.model_dump() | {"missing_colours": [[255, 0, 255]]}))
    assert extract(image, cfg).flags[0, 0] == CellFlag.MISSING
    # Ordinary white is a valid endpoint unless explicitly configured as missing.
    image, cfg = reference()
    image[:8, :8] = 255
    assert extract(image, cfg).values[0, 0] == 255


def test_all_cells_missing_and_invalid_masks():
    image, cfg = reference()
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[:32, :32] = True
    result = extract(image, cfg, mask=mask)
    assert np.isnan(result.values).all()
    assert np.all(result.flags == CellFlag.MISSING)
    for invalid in (np.zeros((4, 4), dtype=bool), np.zeros(image.shape[:2])):
        with pytest.raises(ValueError, match="boolean array"):
            extract(image, cfg, mask=invalid)
    mask[:, 43] = True
    with pytest.raises(ValueError, match="colourbar sampling strip"):
        extract(image, cfg, mask=mask)


def test_exports_hashes_nan_flags_and_reproducibility(tmp_path):
    image, cfg = reference()
    image[:8, :8] = [255, 0, 255]
    rgba = np.concatenate((image, np.full((*image.shape[:2], 1), 255, dtype=np.uint8)), axis=2)
    rgba[8:16, :8, 3] = 0
    path = tmp_path / "source.png"
    Image.fromarray(rgba).save(path)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[:8, 8:16] = True
    result = extract_file(path, cfg, tmp_path / "first", mask=mask)
    extract_file(path, cfg, tmp_path / "second", mask=mask)
    for name in ("matrix.csv", "quality.csv", "provenance.json", "diagnostic.png"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    np.testing.assert_array_equal(
        np.loadtxt(tmp_path / "first/matrix.csv", delimiter=","), result.values
    )
    metadata = json.loads((tmp_path / "first/provenance.json").read_text())
    assert metadata["input"]["filename"] == path.name
    assert metadata["input"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["input"]["mask_sha256"] == hashlib.sha256(mask.tobytes()).hexdigest()
    assert metadata["config"] == cfg.model_dump(mode="json")
    assert metadata["qc_thresholds"]["max_colour_distance"] == cfg.max_colour_distance
    assert metadata["quality"]["flag_fractions"]["MISSING"] == 2 / 16
    with (tmp_path / "first/quality.csv").open() as stream:
        quality = list(csv.DictReader(stream))
    assert len(quality) == 16
    assert "POOR_MATCH" in quality[0]["flags"]
    assert quality[1]["flags"] == "MISSING"
    assert quality[1]["candidate_value"] == "nan"
    with Image.open(tmp_path / "first/diagnostic.png") as diagnostic:
        assert diagnostic.mode == "RGB"
        assert diagnostic.getpixel((20 + 18 + 6, 35 + 8)) == (150, 150, 150)
    np.testing.assert_array_equal(result.values, extract(rgba, cfg, mask=mask).values)
    assert diagnostic_image(result).height > cfg.rows * 18


def colourful_reference():
    t = np.linspace(0, 1, 512)
    bar = np.rint(np.stack((255 * t, 255 * (1 - t), 1020 * t * (1 - t)), axis=1)).astype(np.uint8)
    indices = np.random.default_rng(45).integers(0, 512, (20, 20))
    image = np.full((512, 144, 3), 255, dtype=np.uint8)
    image[:120, :120] = np.repeat(np.repeat(bar[indices], 6, axis=0), 6, axis=1)
    image[:, 130:140] = bar[:, None]
    cfg = ExtractionConfig(
        heatmap_bbox=(0, 0, 120, 120),
        colourbar_bbox=(130, 0, 140, 512),
        rows=20,
        columns=20,
        colourbar_orientation="vertical",
        colourbar_start_value=-2,
        colourbar_end_value=2,
        max_colour_distance=0.5,
        ambiguity_colour_tolerance=0.3,
    )
    return image, cfg


@pytest.mark.parametrize("degradation", ["jpeg", "blur", "gridlines_blur"])
def test_degraded_images_increase_diagnostics(tmp_path, degradation):
    image, cfg = colourful_reference()
    clean = extract(image, cfg)
    assert not clean.flags.any()
    if degradation == "jpeg":
        path = tmp_path / "compressed.jpg"
        Image.fromarray(image).save(path, quality=8, subsampling=2)
        with Image.open(path) as raster:
            degraded = np.array(raster.convert("RGB"))
    else:
        if degradation == "gridlines_blur":
            image[:120:6, :120] = 0
            image[:120, :120:6] = 0
        degraded = np.array(Image.fromarray(image).filter(ImageFilter.GaussianBlur(1.5)))
    result = extract(degraded, cfg)
    print(
        f"{degradation}: mean DeltaE={result.colour_distance.mean():.3f}; "
        f"flagged={np.mean(result.flags != 0):.1%}"
    )
    assert result.colour_distance.mean() > clean.colour_distance.mean() + 0.3
    assert np.mean(result.flags != 0) > 0.1
    assert np.any(result.flags & CellFlag.POOR_MATCH)


def test_gridline_avoidance():
    image, cfg = colourful_reference()
    clean = extract(image, cfg)
    image[:120:6, :120] = 0
    image[:120, :120:6] = 0
    lined = extract(image, cfg)
    np.testing.assert_array_equal(lined.values, clean.values)
    np.testing.assert_array_equal(lined.flags, clean.flags)


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
def test_missing_bar_strip_in_both_orientations(orientation):
    image, cfg = reference()
    if orientation == "horizontal":
        image = image.transpose(1, 0, 2).copy()
        cfg = ExtractionConfig(
            **(
                cfg.model_dump()
                | {"colourbar_bbox": (0, 40, 256, 48), "colourbar_orientation": orientation}
            )
        )
    mask = np.zeros(image.shape[:2], dtype=bool)
    if orientation == "vertical":
        mask[20, 43] = True
    else:
        mask[43, 20] = True
    with pytest.raises(ValueError, match="colourbar sampling strip"):
        extract(image, cfg, mask=mask)


def test_exports_do_not_overwrite_input(tmp_path):
    image, cfg = reference()
    path = tmp_path / "diagnostic.png"
    Image.fromarray(image).save(path)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite the input"):
        extract_file(path, cfg, tmp_path)
    assert path.read_bytes() == original


def test_all_missing_export_is_json_safe(tmp_path):
    image, cfg = reference()
    path = tmp_path / "source.png"
    Image.fromarray(image).save(path)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[:32, :32] = True
    extract_file(path, cfg, tmp_path / "results", mask=mask)
    metadata = json.loads((tmp_path / "results/provenance.json").read_text())
    assert metadata["quality"]["flagged_fraction"] == 1
    assert np.isnan(np.loadtxt(tmp_path / "results/matrix.csv", delimiter=",")).all()


def test_qc_distance_and_poor_match_retains_estimate():
    image, cfg = reference()
    clean = extract(image, cfg)
    assert not clean.flags.any()
    np.testing.assert_allclose(clean.colour_distance, 0)
    image[:8, :8] = [255, 0, 255]
    result = extract(image, cfg)
    assert result.flags[0, 0] & CellFlag.POOR_MATCH
    assert result.colour_distance[0, 0] > cfg.max_colour_distance
    assert np.isfinite(result.inferred_values[0, 0])
    np.testing.assert_array_equal(result.flags[1:], clean.flags[1:])
    # A poor, but unambiguous estimate remains available with its flag.
    relaxed = cfg.model_copy(update={"ambiguity_position_fraction": 1.0})
    assert np.isfinite(extract(image, relaxed).values[0, 0])


def test_repeated_colours_and_near_matches_are_ambiguous():
    bar = np.array([[0, 0, 255], [255, 255, 255], [255, 0, 0], [0, 0, 255]], dtype=float)
    match = match_with_quality(np.array([[0, 0, 254]], dtype=float), bar)
    assert match.ambiguous[0]
    assert match.distance[0] > 0
    image, cfg = reference()
    image[:, 40:48] = np.concatenate((np.arange(128), np.arange(127, -1, -1)))[:, None, None]
    image[:8, :8] = 40
    result = extract(image, cfg)
    assert result.flags[0, 0] & CellFlag.AMBIGUOUS
    assert np.isnan(result.values[0, 0])
    assert np.isfinite(result.inferred_values[0, 0])


def test_saturated_endpoint_and_constant_bar():
    image, cfg = reference()
    image[:64, 40:48] = 0
    image[:8, :8] = 0
    result = extract(image, cfg)
    assert result.flags[0, 0] & CellFlag.AMBIGUOUS
    assert np.isnan(result.values[0, 0])
    image[:, 40:48] = 0
    result = extract(image, cfg)
    assert np.all(result.flags & CellFlag.AMBIGUOUS)
    assert np.isnan(result.values).all()


def test_local_near_matches_do_not_create_ambiguity():
    image, cfg = reference()
    result = extract(image, cfg)
    assert not np.any(result.flags & CellFlag.AMBIGUOUS)
    bar = np.array([[0, 0, 0], [0, 0, 0], [255, 255, 255]], dtype=float)
    match = match_with_quality(np.zeros((1, 3)), bar, ambiguity_position_fraction=0.6)
    assert not match.ambiguous[0]
    assert match.indices[0] == 0  # deterministic nearest-match tie
