"""Capability separation and runtime OCR calibration must preserve uncertainty."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder import AutomationRequest, DetectionHints, ScaleConfig, automate, extract
from heatmap_decoder.cli import app
from heatmap_decoder.legend import evaluate_legend
from heatmap_decoder.models import LegendConfig, LegendTick, OCRRun, OCRToken, ScaleAnchor

ROOT = Path(__file__).resolve().parents[1]
M17 = runpy.run_path(str(ROOT / "benchmarks/m17/fixtures.py"))
M18 = runpy.run_path(str(ROOT / "benchmarks/m18/fixtures.py"))


def ticks(config, interior=False, inconsistent=False):
    vertical = config.colourbar_orientation == "vertical"
    origin = config.colourbar_bbox[1 if vertical else 0]
    offsets = (20, 90, 160) if interior else (0, 90, 180)
    values = (0, 0.8, 1) if inconsistent else (0, 0.5, 1)
    return tuple(
        LegendTick(
            index=i,
            side="right" if vertical else "below",
            position=origin + p,
            position_source="stroke",
            raw_text=str(v),
            parsed_value=v,
            confidence=99,
            status="suggested",
        )
        for i, (p, v) in enumerate(zip(offsets, values))
    )


def request(c, *, ocr=False):
    hints = DetectionHints(
        heatmap_bbox=c.heatmap_bbox,
        colourbar_bbox=c.colourbar_bbox,
        colourbar_orientation=c.colourbar_orientation,
        rows=c.rows,
        columns=c.columns,
        rows_segments=c.rows_segments,
        columns_segments=c.columns_segments,
        requested=(),
        colourbar_start_value=None if ocr else c.colourbar_start_value,
        colourbar_end_value=None if ocr else c.colourbar_end_value,
    )
    return AutomationRequest(
        hints=hints, numeric_only=True, scale=ScaleConfig(source="ocr_ticks") if ocr else None
    )


@pytest.mark.parametrize("interior", [False, True])
def test_ocr_anchors_keep_origin_span_and_cannot_be_loaded_from_config(monkeypatch, interior):
    image, c, _ = M17["fixture"](170002)
    observed = evaluate_legend(
        LegendConfig(
            colourbar_bbox=c.colourbar_bbox, colourbar_orientation=c.colourbar_orientation
        ),
        ticks(c, interior),
    )
    monkeypatch.setattr("heatmap_decoder.automation.inspect_legend", lambda *_: observed)
    result = automate(image, request(c, ocr=True))
    assert result.diagnostic_config.scale.source == "ocr_ticks"
    assert len(result.ocr_anchors) == 3
    assert all(a.origin == "ocr" and a.status == "suggested" for a in result.ocr_anchors)
    with pytest.raises(ValueError, match="legend inspection"):
        extract(image, result.diagnostic_config)
    values = extract(image, result.diagnostic_config, ocr_anchors=result.ocr_anchors)
    assert np.isfinite(values.values).all() == (not interior)
    assert result.numeric_state == ("partial" if interior else "complete")
    if interior:
        assert np.any(values.flags & 8)
        assert result.numeric_complete.status != "accepted"


def test_inconsistent_or_unresolved_ticks_never_choose_subset_or_piecewise(monkeypatch):
    image, c, _ = M17["fixture"](170002)
    config = LegendConfig(
        colourbar_bbox=c.colourbar_bbox, colourbar_orientation=c.colourbar_orientation
    )
    for readings in (
        ticks(c, inconsistent=True),
        ticks(c)[:2],
        ticks(c) + (ticks(c)[0].model_copy(update={"status": "unresolved"}),),
    ):
        observed = evaluate_legend(config, readings)
        monkeypatch.setattr("heatmap_decoder.automation.inspect_legend", lambda *_: observed)
        result = automate(image, request(c, ocr=True), experimental_export=True)
        assert result.diagnostic_config is None
        assert not result.numeric_retention_allowed
        assert not result.automatic_export_allowed


def test_composed_run_retains_numeric_without_authorizing_high_confidence_names(
    monkeypatch, tmp_path
):
    def recognize(image, settings, **kwargs):
        return OCRRun(
            (OCRToken("UnverifiedGene", (0, 0, image.width, image.height), 99),),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "text_line",
            geometry_kind="source_field",
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    image, grid, c, *_ = M18["fixture"](30, blocked=True)
    c = c.model_copy(update={"labels": grid.labels})
    settings = request(c).model_copy(
        update={
            "numeric_only": False,
            "hints": request(c).hints.model_copy(update={"labels": c.labels}),
        }
    )
    result = automate(image, settings, experimental_export=True)
    assert result.numeric_complete.status == "accepted"
    assert result.label_geometry_complete.status == "accepted"
    assert result.identifiers_complete.status != "accepted"
    assert result.numeric_retention_allowed and not result.automatic_export_allowed
    assert len(result.label_evidence) == 4
    figure, config = tmp_path / "figure.png", tmp_path / "request.json"
    Image.fromarray(image).save(figure)
    config.write_text(settings.model_dump_json())
    out = tmp_path / "out"
    cli = CliRunner().invoke(
        app,
        [
            "automate",
            str(figure),
            "--request",
            str(config),
            "--out",
            str(out),
            "--experimental-export",
        ],
    )
    assert cli.exit_code == 5, cli.output
    assert not (out / "matrix.csv").exists()
    data = (out / "numeric_positional/numeric_positional.csv").read_text()
    assert data.startswith("positional_row_id,column_0,column_1")
    assert "UnverifiedGene" not in data
    assert "UnverifiedGene" in (out / "label_evidence.json").read_text()
    metadata = json.loads((out / "numeric_positional/provenance.json").read_text())
    assert metadata["publication"]["labelled_request_fulfilled"] is False


def test_runtime_ocr_fit_does_not_relax_registered_residuals():
    image, c, _ = M17["fixture"](170002)
    c = c.model_copy(
        update={
            "colourbar_start_value": None,
            "colourbar_end_value": None,
            "scale": ScaleConfig(source="ocr_ticks", max_value_residual_fraction=0.05),
        }
    )
    anchors = tuple(
        ScaleAnchor(position=t.position, value=t.parsed_value, origin="ocr", status="suggested")
        for t in ticks(c, inconsistent=True)
    )
    with pytest.raises(ValueError):
        extract(image, c, ocr_anchors=anchors)


def test_failed_label_engine_does_not_discard_numeric_result(monkeypatch):
    image, grid, c, *_ = M18["fixture"](30, blocked=True)
    c = c.model_copy(update={"labels": grid.labels})
    settings = request(c).model_copy(
        update={
            "numeric_only": False,
            "hints": request(c).hints.model_copy(update={"labels": c.labels}),
        }
    )

    def unavailable(*args, **kwargs):
        raise RuntimeError("engine unavailable")

    monkeypatch.setattr("heatmap_decoder.automation.extract_labels", unavailable)
    result = automate(image, settings, experimental_export=True)
    assert result.numeric_state == "complete"
    assert result.numeric_retention_allowed
    assert result.label_geometry_complete.status != "accepted"
    assert not result.automatic_export_allowed


def test_qualified_ocr_partial_export_is_named_and_cannot_claim_completion(monkeypatch, tmp_path):
    image, c, _ = M17["fixture"](170002)
    observed = evaluate_legend(
        LegendConfig(
            colourbar_bbox=c.colourbar_bbox, colourbar_orientation=c.colourbar_orientation
        ),
        ticks(c, interior=True),
    )
    monkeypatch.setattr("heatmap_decoder.automation.inspect_legend", lambda *_: observed)
    monkeypatch.setitem(
        __import__("heatmap_decoder.automation", fromlist=["M19_QUALIFICATION"]).M19_QUALIFICATION,
        "linear_ocr_scale",
        True,
    )
    settings = request(c, ocr=True)
    figure, config = tmp_path / "figure.png", tmp_path / "request.json"
    Image.fromarray(image).save(figure)
    config.write_text(settings.model_dump_json())
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "automate",
            str(figure),
            "--request",
            str(config),
            "--out",
            str(out),
            "--experimental-export",
        ],
    )
    assert result.exit_code == 5, result.output
    assert not (out / "matrix.csv").exists()
    assert (out / "numeric_positional/numeric_partial.csv").exists()
    assert "nan" in (out / "numeric_positional/numeric_partial.csv").read_text()
    metadata = json.loads((out / "automation.json").read_text())
    assert metadata["numeric_state"] == "partial"
    assert not metadata["automatic_export_allowed"]


@pytest.mark.parametrize("seed,challenge", [(293002, "white"), (293004, "panels")])
def test_weak_aligned_components_cannot_disappear_from_full_grid_competition(seed, challenge):
    import runpy

    fixture = runpy.run_path(str(ROOT / "benchmarks/m19/fixtures.py"))["fixture"]
    image, c, *_ = fixture(seed, "both", challenge=challenge)
    hints = DetectionHints(
        requested=("heatmap", "colourbar"),
        infer_rows=True,
        infer_columns=True,
        colourbar_start_value=c.colourbar_start_value,
        colourbar_end_value=c.colourbar_end_value,
    )
    result = automate(
        image, AutomationRequest(numeric_only=True, hints=hints), experimental_export=True
    )
    assert result.detection.heatmap_search_boxes
    assert result.numeric_state == "unresolved"
    assert not result.numeric_retention_allowed
    ordinary, c, *_ = fixture(seed, "both")
    hints = hints.model_copy(
        update={
            "colourbar_start_value": c.colourbar_start_value,
            "colourbar_end_value": c.colourbar_end_value,
        }
    )
    assert (
        automate(ordinary, AutomationRequest(numeric_only=True, hints=hints)).numeric_state
        == "complete"
    )
