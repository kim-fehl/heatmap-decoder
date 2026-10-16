"""Segmented grids contracts and fixtures."""

from dataclasses import asdict

import numpy as np
import pytest
from pydantic import ValidationError

from benchmarks.m17.fixtures import fixture
from heatmap_decoder.automation import AutomationRequest, automate
from heatmap_decoder.blocks import propose_axis
from heatmap_decoder.extract import extract
from heatmap_decoder.geometry import axis_slots, cell_slices, resolve_grid, validate_geometry
from heatmap_decoder.labels import assign_candidates
from heatmap_decoder.models import (
    AxisSegment,
    DetectionHints,
    ExtractionConfig,
    GridConfig,
    LabelCandidate,
)


@pytest.mark.parametrize("family", ["none", "horizontal", "vertical", "both"])
@pytest.mark.parametrize("derivative", ["clean", "jpeg", "blur"])
def test_explicit_reconstruction(family, derivative):
    image, config, truth = fixture(family=family, derivative=derivative)
    result = extract(image, config)
    assert result.values.shape == truth.shape
    finite = np.isfinite(result.values)
    assert finite.all() if derivative == "clean" else finite.mean() >= 0.95
    assert np.abs(result.values[finite] - truth[finite]).max() <= 0.05
    assert np.abs(result.values[finite] - truth[finite]).mean() < 0.02
    grid = resolve_grid(config)
    for axis in (grid.rows, grid.columns):
        assert len(axis.slots) == sum(s.count for s in axis.segments)
        assert [s.index for s in axis.slots] == list(range(len(axis.slots)))
        for s in axis.slots:
            assert not any(a < s.end and b > s.start for a, b in axis.gaps)
    for ys, xs in cell_slices(config):
        assert ys.stop > ys.start and xs.stop > xs.start


def test_totals_and_regular_compatibility():
    image, c, _ = fixture(family="none")
    payload = c.model_dump(exclude={"rows", "columns"})
    inferred = ExtractionConfig.model_validate(payload)
    assert inferred.rows == c.rows and inferred.columns == c.columns
    regular = c.model_copy(update={"rows_segments": None, "columns_segments": None})
    assert axis_slots(c, "rows") == axis_slots(regular, "rows")
    assert np.array_equal(extract(image, c).values, extract(image, regular).values)


@pytest.mark.parametrize(
    "segments,total",
    [
        ([], 2),
        ([dict(start=0, end=10, count=0)], 2),
        ([dict(start=True, end=10, count=2)], 2),
        ([dict(start=0, end=float("inf"), count=2)], 2),
        ([dict(start=1, end=20, count=2)], 2),
        ([dict(start=0, end=21, count=2)], 2),
        ([dict(start=0, end=12, count=1), dict(start=10, end=20, count=1)], 2),
        ([dict(start=10, end=20, count=1), dict(start=0, end=10, count=1)], 2),
        ([dict(start=0, end=20, count=3)], 2),
    ],
)
def test_invalid_segments(segments, total):
    with pytest.raises(ValidationError):
        GridConfig(heatmap_bbox=(0, 0, 20, 20), rows=total, columns=2, rows_segments=segments)


def test_inset_resolution_and_unequal_pitch():
    image, c, _ = fixture()
    payload = c.model_dump()
    payload["rows_segments"] = [
        dict(start=32, end=33, count=2),
        dict(start=40, end=c.heatmap_bbox[3], count=c.rows - 2),
    ]
    bad = ExtractionConfig.model_validate(payload)
    with pytest.raises(ValueError, match="resolution"):
        validate_geometry(bad, image.shape)
    config = GridConfig(
        heatmap_bbox=(0, 0, 40, 50),
        columns=2,
        rows_segments=[dict(start=0, end=10, count=2), dict(start=20, end=50, count=3)],
    )
    assert [(s.start, s.end) for s in axis_slots(config, "rows")] == [
        (0, 5),
        (5, 10),
        (20, 30),
        (30, 40),
        (40, 50),
    ]
    assert resolve_grid(config).rows.ownership == ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2))


def test_label_ownership_hash_and_gaps():
    _, c, _ = fixture()
    for axis in ("rows", "columns"):
        candidates = []
        for s in axis_slots(c, axis):
            bbox = (
                (0, s.centre - 1, 10, s.centre + 1)
                if axis == "rows"
                else (s.centre - 1, 0, s.centre + 1, 10)
            )
            candidates.append(LabelCandidate(str(s.index), bbox, 99, ()))
        result = assign_candidates(candidates, c, axis)
        assert [s.raw_text for s in result.slots] == [str(i) for i in range(len(candidates))]
    g = GridConfig.model_validate(c.model_dump(include=set(GridConfig.model_fields)))
    assert resolve_grid(g).metadata()["sha256"] == resolve_grid(c).metadata()["sha256"]
    changed = c.model_copy(update={"columns_segments": None})
    assert resolve_grid(changed).metadata()["sha256"] != resolve_grid(c).metadata()["sha256"]
    assert asdict(resolve_grid(g).columns)["gaps"]


@pytest.mark.parametrize("family", ["horizontal", "vertical", "both"])
def test_automatic_common_pitch_accepts_entire_matrix(family):
    image, c, truth = fixture(family=family)
    h = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    r = automate(image, AutomationRequest(hints=h, numeric_only=True), experimental_export=True)
    assert r.complete.status == "accepted", r.complete
    assert r.diagnostic_config.heatmap_bbox == c.heatmap_bbox
    assert (r.diagnostic_config.rows, r.diagnostic_config.columns) == truth.shape
    assert axis_slots(r.diagnostic_config, "rows") == axis_slots(c, "rows")
    assert axis_slots(r.diagnostic_config, "columns") == axis_slots(c, "columns")
    assert np.abs(extract(image, r.diagnostic_config).values - truth).max() < 0.05
    # M19 qualifies the sharp measured common-pitch scope with passing QC.
    assert r.automatic_export_allowed
    assert r.numeric_complete.status == "accepted"


def test_missing_band_is_ambiguous():
    image, c, _ = fixture(family="horizontal", challenge="missing")
    p = propose_axis(image, c.heatmap_bbox, "rows")
    assert p.status != "accepted"
    assert any(not h["gaps"] for h in p.alternatives)
    assert any(h["gaps"] for h in p.alternatives)


@pytest.mark.parametrize(
    "challenge,family",
    [("white", "none"), ("annotation", "horizontal"), ("panels", "vertical"), ("cropped", "both")],
)
def test_misleading_cases_do_not_complete(challenge, family):
    image, c, _ = fixture(family=family, challenge=challenge)
    h = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=0,
        colourbar_end_value=1,
    )
    r = automate(image, AutomationRequest(hints=h, numeric_only=True), experimental_export=True)
    assert r.complete.status != "accepted"
    assert not r.automatic_export_allowed


def test_wrong_supplied_count_cannot_confirm_block():
    image, c, _ = fixture()
    p = propose_axis(image, c.heatmap_bbox, "rows", c.rows + 1)
    assert p.status != "accepted"
    assert p.evidence["count_is_confirming_signal"] is False


def test_multiple_variable_gaps_constant_neighbours_and_strokes():
    image = np.full((100, 170, 3), 255, dtype=np.uint8)
    segments = (
        AxisSegment(start=5, end=35, count=2),
        AxisSegment(start=40, end=85, count=3),
        AxisSegment(start=97, end=157, count=4),
    )
    c = ExtractionConfig(
        heatmap_bbox=(5, 5, 157, 65),
        rows=4,
        columns_segments=segments,
        colourbar_bbox=(5, 80, 106, 90),
        colourbar_orientation="horizontal",
        colourbar_start_value=0,
        colourbar_end_value=1,
    )
    profile = np.stack((np.arange(101) * 2, np.full(101, 80), 200 - np.arange(101)), axis=1).astype(
        np.uint8
    )
    image[80:90, 5:106] = profile
    truth = np.random.default_rng(17).integers(10, 90, (4, 9))
    truth[:, 1] = truth[:, 0]  # genuine neighbouring constant cells
    for row in axis_slots(c, "rows"):
        for col in axis_slots(c, "columns"):
            image[int(row.start) : int(row.end), int(col.start) : int(col.end)] = profile[
                truth[row.index, col.index]
            ]
            image[int(row.start) : int(row.end), int(col.start)] = 0  # thin strokes
    r = extract(image, c)
    assert np.isfinite(r.values).all()
    assert np.abs(r.values - truth / 100).max() < 0.01
    assert resolve_grid(c).columns.gaps == ((35, 40), (85, 97))
    assert len(resolve_grid(c).columns.slots) == 9


def test_explicit_geometry_is_authoritative_for_detection_and_export(tmp_path):
    import json

    from PIL import Image

    from heatmap_decoder.io import extract_file

    image, c, truth = fixture(challenge="missing")
    # The explicit reviewed geometry may remove an ambiguous integer-pitch band.
    h = DetectionHints(
        heatmap_bbox=c.heatmap_bbox,
        requested=(),
        rows_segments=c.rows_segments,
        columns_segments=c.columns_segments,
        colourbar_bbox=c.colourbar_bbox,
        colourbar_orientation=c.colourbar_orientation,
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    r = automate(image, AutomationRequest(hints=h, numeric_only=True))
    assert r.complete.status == "accepted"
    assert r.diagnostic_config.rows_segments == c.rows_segments
    source = tmp_path / "figure.png"
    Image.fromarray(image).save(source)
    extracted = extract_file(source, c, tmp_path / "output")
    metadata = json.loads((tmp_path / "output/provenance.json").read_text())
    assert metadata["grid_sha256"] == resolve_grid(c).metadata()["sha256"]
    assert metadata["resolved_grid"]["rows"]["gaps"]
    assert np.abs(extracted.values - truth).max() < 0.05


def test_unresolved_full_union_cannot_accept_a_partial_block():
    image, c, _ = fixture(seed=181001, challenge="missing")
    h = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    r = automate(image, AutomationRequest(hints=h, numeric_only=True), experimental_export=True)
    assert r.complete.status != "accepted"
    assert not r.automatic_export_allowed


def test_palette_association_uses_resolved_block_interiors():
    image, c, truth = fixture(seed=180032)
    h = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    r = automate(image, AutomationRequest(hints=h, numeric_only=True))
    assert r.complete.status == "accepted"
    assert r.diagnostic_config.heatmap_bbox == c.heatmap_bbox
    assert np.abs(extract(image, r.diagnostic_config).values - truth).max() < 0.05


def test_small_component_above_white_data_band_is_not_discarded():
    image, c, _ = fixture(seed=191003, family="none", challenge="white")
    h = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    r = automate(image, AutomationRequest(hints=h, numeric_only=True), experimental_export=True)
    assert r.complete.status != "accepted"
    assert not r.automatic_export_allowed


def test_separator_plateau_inside_blurred_halo_is_not_resolved():
    image, c, _ = fixture(seed=190038, derivative="blur")
    h = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    r = automate(image, AutomationRequest(hints=h, numeric_only=True))
    assert r.complete.status != "accepted"
