"""Scientific regression checks for the current-phase review's implementation blockers."""

import importlib.util
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont
from typer.testing import CliRunner

from heatmap_decoder import AutomationRequest, DetectionHints, automate, extract
from heatmap_decoder.acceptance import boundary_evidence, count_decision
from heatmap_decoder.cli import app
from heatmap_decoder.detection import colourbar_candidates
from heatmap_decoder.models import CountProposal, CountSignal, DetectionResult, OCRRun, OCRToken

try:
    ImageFont.truetype("DejaVuSans.ttf", 18)
except OSError:
    pytest.skip("deterministic image fixtures require DejaVu fonts", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
M12 = runpy.run_path(str(ROOT / "examples/m12/generate.py"))
M10 = runpy.run_path(str(ROOT / "examples/m10/generate.py"))
QA = runpy.run_path(str(ROOT / "examples/corrections/generate.py"))


def request_for(truth, **extra):
    return AutomationRequest(
        hints=DetectionHints(
            rows=truth["rows"],
            columns=truth["columns"],
            requested=("heatmap", "colourbar"),
            colourbar_start_value=truth["start_value"],
            colourbar_end_value=truth["end_value"],
            **extra,
        ),
        numeric_only=True,
    )


def fixture(index=0, degradation="clean", renderer="pillow"):
    return QA["make_fixture"](
        dict(index=index, seed=236000 + index, renderer=renderer, degradation=degradation)
    )


@pytest.mark.parametrize("index", [2, 3])
def test_missed_clean_bars_recovered_and_numeric_result(index):
    image, t = M12["make_fixture"](
        dict(
            index=index,
            seed=122000 + index,
            family=("vertical_flank", "horizontal_cap", "cross_axis")[index % 3],
            degradation="clean",
        )
    )
    result = automate(image, request_for(t))
    bars = [c for c in result.detection.candidates if c.kind == "colourbar"]
    assert len(bars) == 1
    assert bars[0].bbox == t["colourbar_bbox"]
    assert result.complete.status == "accepted"
    recovered = extract(image, result.diagnostic_config)
    assert (
        np.abs(recovered.values - np.asarray(t["expected_matrix"])).mean()
        / abs(t["end_value"] - t["start_value"])
        < 0.02
    )


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("kind", ["diverging", "grayscale"])
def test_moving_asymmetric_tick_text_does_not_move_core(orientation, side, kind):
    expected = (170, 50, 186, 250) if orientation == "vertical" else (70, 170, 420, 186)
    observations = []
    for extra_gap in (0, 16):
        image = np.full((330, 530, 3), 255, np.uint8)
        x, y, r, s = expected
        profile = QA["palette"](
            np.linspace(0, 1, s - y if orientation == "vertical" else r - x), kind
        )
        image[y:s, x:r] = profile[:, None] if orientation == "vertical" else profile[None]
        figure = Image.fromarray(image)
        draw = ImageDraw.Draw(figure)
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        for p, text in zip(
            np.linspace(
                y if orientation == "vertical" else x,
                s - 1 if orientation == "vertical" else r - 1,
                5,
            ),
            ("-100.5", "-4", "0", "2.00", "75"),
            strict=True,
        ):
            if orientation == "vertical":
                px = (
                    x - 14 - font.getlength(text) - extra_gap
                    if side == "before"
                    else r + 14 + extra_gap
                )
                draw.text((px, p), text, font=font, anchor="lm", fill="black")
                draw.line(
                    (x - 8, p, x - 2, p) if side == "before" else (r + 2, p, r + 8, p), fill="black"
                )
            else:
                py = y - 24 - extra_gap if side == "before" else s + 24 + extra_gap
                draw.text((p, py), text, font=font, anchor="mm", fill="black")
                draw.line(
                    (p, y - 8, p, y - 2) if side == "before" else (p, s + 2, p, s + 8), fill="black"
                )
        bars = colourbar_candidates(np.asarray(figure), DetectionHints(rows=20, columns=6))
        assert len(bars) == 1
        observations.append(bars[0].bbox)
    assert observations == [expected, expected]


def competing_fixture():
    suite = runpy.run_path(str(ROOT / "benchmarks/m12/evaluate.py"))
    return next(
        (image, t)
        for case, image, t in suite["challenges"]("test")
        if case["challenge"] == "competing_bars"
    )


def test_two_bars_survive_measurement_and_experimental_export_is_blocked():
    image, t = competing_fixture()
    result = automate(image, request_for(t), experimental_export=True)
    bars = [c for c in result.detection.candidates if c.kind == "colourbar"]
    assert len(bars) == 2
    assert t["colourbar_bbox"] in [b.bbox for b in bars]
    assert not result.automatic_export_allowed
    decision = next(d for d in result.decisions if d.requirement == "colourbar")
    assert "competing_bars" in decision.reasons
    assert result.detection.bar_search_evidence


@pytest.mark.parametrize("degradation", ["clean", "blur", "jpeg"])
def test_robust_boundaries_and_supplied_count_phase(degradation):
    image, t = M12["make_fixture"](
        dict(index=0, seed=122000, family="vertical_flank", degradation=degradation)
    )
    extent = boundary_evidence(image, t["heatmap_bbox"], 2.6)
    assert extent.closed
    assert all(e.uncertainty_pixels <= 1 for e in extent.edges)
    assert all(e.outside_spread_rgb <= 8 for e in extent.edges)
    result = automate(image, request_for(t), experimental_export=True)
    assert result.complete.status == "accepted"
    assert result.automatic_export_allowed
    assert np.abs(np.array(result.diagnostic_config.heatmap_bbox) - t["heatmap_bbox"]).max() <= 1
    recovered = extract(image, result.diagnostic_config)
    assert np.abs(recovered.values - t["expected_matrix"]).mean() / 8 < 0.02


def test_partial_grid_edge_does_not_close_against_background():
    image, t = fixture()
    x, y, r, s = t["heatmap_bbox"]
    assert not boundary_evidence(image, (x, y + 26, r, s), 2.6).closed


def test_gridline_counts_resolve_into_new_immutable_config_without_truth():
    image, t = M10["make_layout"](rows=20, columns=6, gridlines=True)
    hints = DetectionHints(
        infer_rows=True,
        infer_columns=True,
        requested=("heatmap", "colourbar"),
        colourbar_start_value=2,
        colourbar_end_value=-2,
    )
    result = automate(image, AutomationRequest(hints=hints, numeric_only=True))
    assert result.complete.status == "accepted"
    assert hints.rows is hints.columns is None
    assert (result.diagnostic_config.rows, result.diagnostic_config.columns) == (20, 6)
    assert all(c.origin == "measured_inference" for c in result.detection.selected_counts.values())
    assert (
        np.abs(
            extract(image, result.diagnostic_config).values - (2 - 4 * np.array(t["values"]))
        ).mean()
        < 0.08
    )


def test_missing_raw_labels_preserve_holes_and_do_not_collapse_count(monkeypatch):
    image, t = M10["make_layout"](rows=20, columns=6, row_side="left", column_side="bottom")
    from heatmap_decoder.detection import _label_searches

    searches = iter(_label_searches(t["heatmap_bbox"], image.shape, 256))

    def recognize(crop, settings):
        side, region = next(searches)
        axis = "rows" if side in ("left", "right") else "columns"
        records = t["tokens"][axis] if side in ("left", "below") else []
        if axis == "rows":
            records = [record for i, record in enumerate(records) if i not in (4, 12)]
        factor = settings.upscale_factor
        tokens = tuple(
            OCRToken(
                r["text"], tuple((v - region[i % 2]) * factor for i, v in enumerate(r["bbox"])), 10
            )
            for r in records
        )
        return OCRRun(tokens, "fake", "fake", "eng", ("eng",), "sparse")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    result = automate(
        image,
        AutomationRequest(
            hints=DetectionHints(
                infer_rows=True,
                infer_columns=True,
                requested=("heatmap", "colourbar"),
                colourbar_start_value=2,
                colourbar_end_value=-2,
                detector={"count_label_geometry": True},
            ),
            numeric_only=True,
        ),
    )
    assert result.complete.status == "accepted"
    assert result.diagnostic_config.rows == 20
    proposal = next(
        p
        for p in result.detection.count_proposals
        if p.id == result.detection.selected_counts["rows"].proposal_id
    )
    assert len(proposal.signals["ocr_left"].positions) == 18
    assert proposal.signals["ocr_left"].coverage == 0.9


def test_competing_count_models_cannot_be_selected_by_rank():
    def proposal(count):
        return CountProposal(
            id=f"count-{count}",
            axis="rows",
            count=count,
            heatmap_id="heatmap-001",
            bbox=(20, 20, 200, 380),
            pitch=360 / count,
            phase=20,
            score=1.0,
            reasons=(),
            signals={
                "colour_transitions": CountSignal(
                    positions=(40, 60),
                    alignment=1.0,
                    coverage=1.0,
                    kind="boundaries",
                    observation="cell_colours_excluding_strokes",
                    excluded_gridline_pixels=True,
                ),
                "ocr_left": CountSignal(
                    positions=(30, 50),
                    alignment=1.0,
                    coverage=1.0,
                    kind="centres",
                    observation="unassigned_text_geometry",
                ),
            },
        )

    result = DetectionResult(
        candidates=(),
        selected={},
        status="review_required",
        unresolved_requirements=(),
        count_proposals=(proposal(6), proposal(12)),
    )
    decision = count_decision(result, DetectionHints(infer_rows=True, columns=6), "rows")
    assert decision.status == "review_required"
    assert "competing_count_lattices" in decision.reasons


def test_runtime_opt_in_cannot_come_from_saved_request():
    image, t = fixture()
    data = request_for(t).model_dump()
    data["experimental_export"] = True
    with pytest.raises(ValueError, match="Extra inputs"):
        AutomationRequest.model_validate(data)
    assert not automate(image, request_for(t)).automatic_export_allowed


def test_cli_experimental_numeric_export_and_provenance(tmp_path):
    image, t = fixture()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    request = tmp_path / "request.json"
    request.write_text(request_for(t).model_dump_json())
    out = tmp_path / "out"
    result = CliRunner().invoke(
        app,
        [
            "automate",
            str(figure),
            "--request",
            str(request),
            "--out",
            str(out),
            "--experimental-export",
        ],
    )
    assert result.exit_code == 0, result.output
    matrix = np.loadtxt(out / "matrix.csv", delimiter=",")
    assert matrix.shape == (20, 6)
    assert np.abs(matrix - t["expected_matrix"]).mean() / 8 < 0.02
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["publication"]["status"] == "experimental"
    assert provenance["automation"]["stable_release"] is False
    assert provenance["automation"]["runtime_control"] == "--experimental-export"
    assert provenance["recovery"]["original_expression_counts_recovered"] is False


def test_failed_experimental_run_preserves_existing_matrix(tmp_path):
    image, t = competing_fixture()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    request = tmp_path / "request.json"
    request.write_text(request_for(t).model_dump_json())
    out = tmp_path / "out"
    out.mkdir()
    (out / "matrix.csv").write_text("existing\n")
    result = CliRunner().invoke(
        app,
        [
            "automate",
            str(figure),
            "--request",
            str(request),
            "--out",
            str(out),
            "--experimental-export",
            "--overwrite",
        ],
    )
    assert result.exit_code == 5, result.output
    assert (out / "matrix.csv").read_text() == "existing\n"
    assert not (out / "quality.csv").exists()


def test_inferred_export_is_separately_scoped():
    image, _ = M10["make_layout"](rows=20, columns=6, gridlines=True)
    result = automate(
        image,
        AutomationRequest(
            hints=DetectionHints(
                infer_rows=True,
                infer_columns=True,
                requested=("heatmap", "colourbar"),
                colourbar_start_value=2,
                colourbar_end_value=-2,
            ),
            numeric_only=True,
        ),
        experimental_export=True,
    )
    assert result.complete.status == "accepted"
    assert not result.automatic_export_allowed
    assert "experimental_inferred_count_export_not_validated" in result.release_reasons


@pytest.mark.parametrize("renderer", ["pillow", "matplotlib"])
def test_layout_qa_runs_before_recognition(renderer):
    if renderer == "matplotlib" and importlib.util.find_spec("matplotlib") is None:
        pytest.skip("optional benchmark renderer")
    image, t = fixture(8, renderer=renderer)
    QA["assert_layout"](t, image.shape)
    t["text_boxes"][0]["bbox"] = t["colourbar_bbox"]
    with pytest.raises(AssertionError, match="text/core overlap"):
        QA["assert_layout"](t, image.shape)


@pytest.mark.ocr_integration
@pytest.mark.skipif(os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1", reason="real engine opt-in")
@pytest.mark.parametrize("engine", ["tesseract", "rapidocr"])
def test_label_geometry_count_inference_accepts_or_safely_abstains(engine):
    image, _ = M10["make_layout"](rows=20, columns=6, row_side="left", column_side="bottom")
    result = automate(
        image,
        AutomationRequest(
            hints=DetectionHints(
                infer_rows=True,
                infer_columns=True,
                requested=("heatmap", "colourbar"),
                colourbar_start_value=2,
                colourbar_end_value=-2,
                detector={"count_label_geometry": True},
                labels={"ocr": {"engine": engine}},
            ),
            numeric_only=True,
        ),
    )
    # M17 measures the regular lattice independently of label spelling/OCR.
    # Both engines may now complete numeric geometry without identifier acceptance.
    assert result.complete.status == "accepted"
    assert (result.diagnostic_config.rows, result.diagnostic_config.columns) == (20, 6)
    assert result.detection.selected_counts["rows"].origin == "measured_inference"
    assert result.detection.selected_counts["columns"].origin == "measured_inference"
    assert not result.automatic_export_allowed
    assert result.detection.acceptance.identifier_verification == "not_requested"


def test_diagnostic_ticks_are_inspected_on_both_competing_bars(monkeypatch):
    from heatmap_decoder.models import LegendResult, ScaleConfig

    image, t = competing_fixture()
    inspected = []

    def inspect(image, config):
        inspected.append(config.colourbar_bbox)
        return LegendResult(config=config, ticks=(), status="unresolved", reasons=("diagnostic",))

    monkeypatch.setattr("heatmap_decoder.automation.inspect_legend", inspect)
    base = request_for(t)
    hints = base.hints.model_copy(
        update={"colourbar_start_value": None, "colourbar_end_value": None}
    )
    result = automate(
        image,
        AutomationRequest(hints=hints, numeric_only=True, scale=ScaleConfig(source="ocr_ticks")),
    )
    assert len(inspected) == len(set(inspected)) == 2
    assert len(result.legend_diagnostics) == 2
    assert result.legend is None
    assert result.diagnostic_config is None
    assert not result.automatic_export_allowed
