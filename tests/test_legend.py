"""M11 spatial scale fitting, bounded association, review and scientific QC."""

import csv
import hashlib
import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder import ExtractionConfig, extract
from heatmap_decoder.calibration import fit_linear_scale, normalise_bar_positions, scale_values
from heatmap_decoder.cli import app
from heatmap_decoder.io import (
    UnresolvedLegendError,
    extract_file,
    inspect_legend_file,
    load_legend_review,
)
from heatmap_decoder.legend import (
    associate_tokens,
    evaluate_legend,
    inspect_legend,
    parse_numeric_tick,
    search_regions,
)
from heatmap_decoder.models import (
    LegendConfig,
    LegendTick,
    OCRRun,
    OCRToken,
    ScaleAnchor,
    ScaleConfig,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = runpy.run_path(str(ROOT / "examples/m11/generate.py"))


@pytest.mark.parametrize(
    "raw,value",
    [
        ("-2", -2),
        ("−1.25", -1.25),
        ("+.75", 0.75),
        (" 2. ", 2),
        ("- 2", -2),
        ("1.5e-3", 0.0015),
        ("-2E+2", -200),
        ("10^3", 1000),
        ("10^-3", 0.001),
    ],
)
def test_numeric_grammar_preserves_raw_and_records_normalisation(raw, value):
    parsed = parse_numeric_tick(raw)
    assert parsed.raw_text == raw
    assert parsed.value == value
    assert parsed.reason is None
    if raw != parsed.normalised_text or "^" in raw:
        assert parsed.actions


@pytest.mark.parametrize(
    "raw",
    ["O.1", "1,234", "2 mg", "10³", "1e", "--2", "NaN", "inf", "1e999", "x10^3", "1 000", "HSP70A"],
)
def test_unparsed_and_overflow_never_silently_corrected(raw):
    assert parse_numeric_tick(raw).value is None


def test_decimal_comma_requires_explicit_locale():
    assert parse_numeric_tick("-1,5").value is None
    assert parse_numeric_tick("-1,5", ",").value == -1.5
    assert parse_numeric_tick("1,2,3", ",").value is None
    assert parse_numeric_tick("1,234.5", ",").value is None


def anchors_for(box, orientation, count=5, descending=True, nonuniform=False, interior=False):
    start, last = (box[1], box[3] - 1) if orientation == "vertical" else (box[0], box[2] - 1)
    t = np.linspace(0, 1, count) ** (1.5 if nonuniform else 1)
    if interior:
        t = 0.15 + 0.7 * t
    values = 2 - 4 * t if descending else -2 + 4 * t
    return tuple(
        ScaleAnchor(position=float(start + x * (last - start)), value=float(value))
        for x, value in zip(t, values, strict=True)
    )


@pytest.mark.parametrize(
    "orientation,box", [("vertical", (20, 40, 34, 440)), ("horizontal", (20, 40, 420, 54))]
)
@pytest.mark.parametrize("count", [2, 3, 5, 7])
@pytest.mark.parametrize("descending", [False, True])
def test_fit_uses_actual_nonuniform_pixel_positions(orientation, box, count, descending):
    anchors = anchors_for(box, orientation, count, descending, nonuniform=True)
    fit = fit_linear_scale(anchors, box, orientation)
    assert fit.slope == pytest.approx(-4 if descending else 4)
    assert fit.intercept == pytest.approx(2 if descending else -2)
    assert fit.endpoint_coverage
    assert fit.tests_linearity == (count >= 3)
    assert max(abs(v) for v in fit.value_residuals) < 1e-12
    coordinates = np.array(
        [box[1], box[3] - 1] if orientation == "vertical" else [box[0], box[2] - 1], float
    )
    np.testing.assert_array_equal(normalise_bar_positions(coordinates, box, orientation), [0, 1])


@pytest.mark.parametrize(
    "case", ["duplicate", "zero_range", "reversal", "nonlinear", "close", "outside"]
)
def test_invalid_scale_cannot_fall_back_to_endpoints(case):
    box = (20, 40, 34, 440)
    a = list(anchors_for(box, "vertical", 5))
    if case == "duplicate":
        a[2] = a[1].model_copy(update={"value": 100})
    elif case == "zero_range":
        a = [p.model_copy(update={"value": 1}) for p in a]
    elif case == "reversal":
        a[2] = a[2].model_copy(update={"value": 3})
    elif case == "nonlinear":
        a[2] = a[2].model_copy(update={"value": 0.4})
    elif case == "close":
        a = [ScaleAnchor(position=40 + i * 0.0001, value=i) for i in range(5)]
    else:
        a[0] = a[0].model_copy(update={"position": 10})
    with pytest.raises(ValueError):
        fit_linear_scale(tuple(a), box, "vertical")


def fixture_config(truth, scale):
    return ExtractionConfig(
        heatmap_bbox=truth["heatmap_bbox"],
        colourbar_bbox=truth["colourbar_bbox"],
        colourbar_orientation=truth["orientation"],
        rows=truth["rows"],
        columns=truth["columns"],
        scale=scale,
    )


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("count", [2, 3, 5, 7])
def test_known_large_matrix_reconstructed_from_spatial_fit(orientation, count):
    image, truth = FIXTURE["make_fixture"](orientation=orientation, count=count, nonuniform=True)
    anchors = anchors_for(truth["colourbar_bbox"], orientation, count, nonuniform=True)
    config = fixture_config(
        truth, ScaleConfig(source="manual_ticks", anchors=anchors, quantity="row z-score")
    )
    result = extract(image, config)
    error = np.abs(result.values - np.array(truth["expected_matrix"]))
    assert np.isfinite(result.values).all()
    assert error.mean() / 4 < 0.02
    assert error.max() / 4 < 0.02


def test_legacy_endpoint_scale_is_bit_for_bit_identical():
    config = ExtractionConfig.model_validate_json((ROOT / "examples/config.json").read_text())
    expected = np.linspace(config.colourbar_start_value, config.colourbar_end_value, 320)
    np.testing.assert_array_equal(scale_values(config, 320), expected)
    explicit = config.model_copy(update={"scale": ScaleConfig(source="manual_endpoints")})
    np.testing.assert_array_equal(scale_values(explicit, 320), expected)
    payload = config.model_dump()
    payload["scale"] = {
        "source": "manual_ticks",
        "anchors": [{"position": 0, "value": 1}, {"position": 1, "value": 2}],
    }
    with pytest.raises(ValueError, match="conflicts"):
        ExtractionConfig.model_validate(payload)


def test_ocr_anchor_evidence_cannot_be_loaded_as_manual_acceptance():
    image, truth = FIXTURE["make_fixture"]()
    anchors = tuple(
        a.model_copy(update={"origin": "ocr", "status": "suggested"})
        for a in anchors_for(truth["colourbar_bbox"], "vertical")
    )
    config = fixture_config(truth, ScaleConfig(source="manual_ticks", anchors=anchors))
    with pytest.raises(ValueError, match="explicitly supplied/reviewed"):
        extract(image, config)
    with pytest.raises(ValueError, match="requires at least two"):
        ScaleConfig(source="manual_ticks", anchors=())


def ticks_from_anchors(anchors, position_source="stroke"):
    return tuple(
        LegendTick(
            index=i,
            side="right",
            position=a.position,
            position_source=position_source,
            raw_text=str(a.value),
            parsed_value=a.value,
            confidence=99,
            status="suggested",
        )
        for i, a in enumerate(anchors)
    )


@pytest.mark.parametrize("count", [2, 3, 5, 7])
def test_tick_number_endpoint_coverage_and_release_gates(count):
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    result = evaluate_legend(
        config, ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", count))
    )
    assert result.development_eligible == (count >= 3)
    assert result.status == "review_required"
    assert not result.automatic_export_allowed
    if count == 2:
        assert "two_ticks_cannot_test_linearity" in result.reasons
    interior = evaluate_legend(
        config,
        ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", count, interior=True)),
    )
    assert interior.development_eligible == (count >= 3)
    assert "interior_ticks_limit_calibration_span" in interior.reasons


def test_unparsed_tick_and_text_centre_fallback_block_eligibility():
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    ticks = list(ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 5)))
    ticks[2] = ticks[2].model_copy(
        update={"raw_text": "O", "parsed_value": None, "status": "unresolved"}
    )
    result = evaluate_legend(config, tuple(ticks))
    assert result.status == "unresolved"
    assert not result.development_eligible
    fallback = evaluate_legend(
        config, ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical"), "text_centre")
    )
    assert not fallback.development_eligible
    assert "text_centres_are_weaker_position_evidence" in fallback.reasons


def test_nonlinear_scale_keeps_residuals_and_never_removes_outlier():
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    ticks = list(ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 5)))
    ticks[2] = ticks[2].model_copy(update={"parsed_value": 0.4})
    result = evaluate_legend(config, tuple(ticks))
    assert result.fit is not None
    assert len(result.fit.value_residuals) == 5
    assert not result.fit.within_residual_limits
    assert "anchors_contradict_linear_scale" in result.reasons
    assert result.status == "unresolved"


def fake_engine(monkeypatch, truth, *, bad_text=None):
    # Geometry-controlled adapter; deterministic tests don't require Tesseract.
    def recognize(
        crop, settings, *, mode="localisation", timeout_seconds=30, numeric_characters_only=True
    ):
        factor = settings.upscale_factor
        if mode == "localisation":
            # Search both sides in fixed order; actual crop origin is inferred from
            # the call sequence below, never from recognised biological strings.
            side, box = next(searches)
            tokens = []
            if side == truth["tick_side"]:
                for index, t in enumerate(truth["tick_tokens"]):
                    text = bad_text if bad_text is not None and index == 2 else t["text"]
                    b = tuple((v - box[i % 2]) * factor for i, v in enumerate(t["bbox"]))
                    tokens.append(OCRToken(text, b, 99))
        else:
            # Numeric pass is bounded to one line. Match its call sequence to
            # independently generated truth; no scale fit guides the OCR stub.
            index = next(numbers)
            text = (
                bad_text
                if bad_text is not None and index == 2
                else truth["tick_tokens"][index]["text"]
            )
            tokens = [OCRToken(text, (3, 3, crop.width - 3, crop.height - 3), 99)]
        return OCRRun(tuple(tokens), "fake-engine", "fake-wrapper", "eng", ("eng",), mode)

    config = LegendConfig(
        colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation=truth["orientation"]
    )
    searches = iter(
        search_regions(
            np.zeros((truth["heatmap_bbox"][3] + 220, truth["heatmap_bbox"][2] + 340, 3), np.uint8),
            config,
        )
    )
    numbers = iter(range(len(truth["tick_tokens"])))
    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)


@pytest.mark.parametrize(
    "orientation,side",
    [("vertical", "left"), ("vertical", "right"), ("horizontal", "above"), ("horizontal", "below")],
)
def test_stroke_localisation_and_numeric_adapter_on_all_sides(monkeypatch, orientation, side):
    image, truth = FIXTURE["make_fixture"](orientation=orientation, side=side, count=5)
    fake_engine(monkeypatch, truth)
    config = LegendConfig(colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation=orientation)
    result = inspect_legend(image, config)
    assert len(result.ticks) == 5
    assert [t.position for t in result.ticks] == truth["positions"]
    assert [t.raw_text for t in result.ticks] == [t["text"] for t in truth["tick_tokens"]]
    assert result.development_eligible
    assert {r.purpose for r in result.ocr_records} == {"localisation", "numeric_line"}
    assert any(r.masked_stroke_boxes for r in result.ocr_records)


def test_ambiguous_and_misleading_text_retained_without_scale_guess():
    config = LegendConfig(colourbar_bbox=(100, 50, 114, 250), colourbar_orientation="vertical")
    region = (114, 36, 210, 264)
    strokes = (LegendTick(index=0, side="right", position=100, position_source="stroke"),)
    # Original pixel boxes after factor 3: two different numbers at one stroke.
    tokens = (
        OCRToken("2", (30, 174, 60, 210), 99),
        OCRToken("3", (180, 174, 210, 210), 99),
        OCRToken("Panel", (60, 0, 210, 30), 99),
        OCRToken("HSP70A", (60, 354, 240, 390), 99),
    )
    run = OCRRun(tokens, "fake", "fake", "eng", ("eng",), "localisation")
    ticks = associate_tokens(config, "right", strokes, run, region)
    assert ticks[0].status == "ambiguous"
    assert any(t.raw_text == "HSP70A" and t.parsed_value is None for t in ticks)
    assert any("title_or_caption" in reason for t in ticks for reason in t.reasons)
    assert not evaluate_legend(config, ticks).development_eligible


def test_missing_minus_is_not_repaired_to_make_a_linear_fit():
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    ticks = list(ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 5)))
    ticks[-1] = ticks[-1].model_copy(update={"raw_text": "2", "parsed_value": 2})
    result = evaluate_legend(config, tuple(ticks))
    assert result.fit is None
    assert any("monotonic" in reason for reason in result.reasons)
    assert result.ticks[-1].parsed_value == 2


def write_review(path, result, digest, **changes):
    rows = [
        dict(
            index=t.index,
            image_sha256=digest,
            include="yes",
            position=t.position,
            value=t.parsed_value if t.parsed_value is not None else 0,
        )
        for t in result.ticks
    ]
    rows[0].update(changes)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["index", "image_sha256", "include", "position", "value"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"image_sha256": "0" * 64}, "hash"),
        ({"index": 99}, "index"),
        ({"position": "nan"}, "finite"),
        ({"value": ""}, "non-empty"),
        ({"value": "inf"}, "finite"),
        ({"position": -1}, "inside"),
        ({"include": "maybe"}, "yes or no"),
    ],
)
def test_image_bound_review_validation(tmp_path, change, match):
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    result = evaluate_legend(
        config, ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 5))
    )
    path = tmp_path / "review.tsv"
    write_review(path, result, "a" * 64, **change)
    with pytest.raises(ValueError, match=match):
        load_legend_review(path, result, "a" * 64)


def test_review_preserves_raw_evidence_and_confirms_two_tick_fit(tmp_path):
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    result = evaluate_legend(
        config, ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 2))
    )
    path = tmp_path / "review.tsv"
    write_review(path, result, "a" * 64)
    reviewed = load_legend_review(path, result, "a" * 64)
    assert reviewed.status == "reviewed"
    assert reviewed.ticks == result.ticks
    assert reviewed.unreviewed_fit == result.fit
    assert reviewed.unreviewed_reasons == result.reasons
    assert reviewed.review_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not reviewed.automatic_export_allowed


def test_cli_legend_only_and_reviewed_scale_export(monkeypatch, tmp_path):
    image, truth = FIXTURE["make_fixture"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    config = fixture_config(truth, ScaleConfig(source="ocr_ticks", quantity="row z-score"))
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    legend_config = LegendConfig(
        colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation="vertical"
    )
    legend_path = tmp_path / "legend.json"
    legend_path.write_text(legend_config.model_dump_json())
    fake_engine(monkeypatch, truth)
    result = CliRunner().invoke(
        app,
        ["legend", str(figure), "--config", str(legend_path), "--out", str(tmp_path / "review")],
    )
    assert result.exit_code == 6, result.output
    assert (tmp_path / "review/legend_diagnostic.png").exists()
    fake_engine(monkeypatch, truth)
    blocked = CliRunner().invoke(
        app, ["extract", str(figure), "--config", str(path), "--out", str(tmp_path / "blocked")]
    )
    assert blocked.exit_code == 6, blocked.output
    assert not (tmp_path / "blocked/matrix.csv").exists()
    fake_engine(monkeypatch, truth)
    exported = CliRunner().invoke(
        app,
        [
            "extract",
            str(figure),
            "--config",
            str(path),
            "--legend-review",
            str(tmp_path / "review/legend_review.tsv"),
            "--out",
            str(tmp_path / "result"),
        ],
    )
    assert exported.exit_code == 0, exported.output
    matrix = np.loadtxt(tmp_path / "result/matrix.csv", delimiter=",")
    assert np.abs(matrix - np.array(truth["expected_matrix"])).mean() / 4 < 0.02
    metadata = json.loads((tmp_path / "result/provenance.json").read_text())
    assert metadata["recovery"]["quantity"] == "row z-score"
    assert not metadata["recovery"]["original_expression_counts_recovered"]
    assert metadata["legend"]["unreviewed_evidence"]["result"]["status"] == "review_required"
    assert metadata["legend"]["result"]["review_sha256"]
    assert metadata["config"]["scale"]["source"] == "manual_ticks"


def test_unavailable_ocr_and_existing_output_protection(monkeypatch, tmp_path):
    def missing(*a, **kw):
        raise ValueError("Install heatmap-decoder[ocr] and local Tesseract")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", missing)
    image, truth = FIXTURE["make_fixture"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    config = fixture_config(truth, ScaleConfig(source="ocr_ticks"))
    with pytest.raises(UnresolvedLegendError, match="Tesseract"):
        extract_file(figure, config, tmp_path / "result")
    assert (tmp_path / "result/legend_ticks.tsv").exists()
    assert not (tmp_path / "result/matrix.csv").exists()
    (tmp_path / "result/matrix.csv").write_text("old\n")
    with pytest.raises(ValueError, match="already exist"):
        extract_file(figure, config, tmp_path / "result")
    assert (tmp_path / "result/matrix.csv").read_text() == "old\n"


@pytest.mark.ocr_integration
@pytest.mark.skipif(os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1", reason="opt-in OCR")
def test_real_engine_clean_five_ticks_and_reviewed_reconstruction(tmp_path):
    image, truth = FIXTURE["make_fixture"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    config = LegendConfig(colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation="vertical")
    result = inspect_legend_file(figure, config, tmp_path / "review")
    assert len(result.ticks) == 5
    assert [t.raw_text for t in result.ticks] == [t["text"] for t in truth["tick_tokens"]]
    assert [t.position for t in result.ticks] == truth["positions"]
    assert result.development_eligible
    extracted = extract_file(
        figure,
        fixture_config(truth, ScaleConfig(source="ocr_ticks", quantity="row z-score")),
        tmp_path / "output",
        legend_review=tmp_path / "review/legend_review.tsv",
    )
    assert np.abs(extracted.values - np.array(truth["expected_matrix"])).mean() / 4 < 0.02
    assert np.isfinite(extracted.values).all()


def test_endpoint_matches_are_potential_bounds_not_original_expression(tmp_path):
    from heatmap_decoder.io import provenance, write_quality_csv

    image, truth = FIXTURE["make_fixture"]()
    image = image.copy()
    x, y, r, s = truth["colourbar_bbox"]
    hx, hy, hr, hs = truth["heatmap_bbox"]
    image[hy : hy + 22, hx : hx + 72] = image[y, x]
    image[hy : hy + 22, hx + 72 : hx + 144] = image[s - 1, x]
    config = fixture_config(
        truth,
        ScaleConfig(
            source="manual_ticks",
            anchors=anchors_for(truth["colourbar_bbox"], "vertical"),
            quantity="row z-score",
        ),
    )
    result = extract(image, config)
    write_quality_csv(result, tmp_path / "qc.csv")
    rows = list(csv.DictReader((tmp_path / "qc.csv").open()))
    assert rows[0]["value_kind"] == "possible_lower_bound"
    assert rows[1]["value_kind"] == "possible_upper_bound"
    assert rows[0]["flags"] == "OK"  # Exact endpoint match does not establish clipping.
    metadata = provenance(result, input_filename="figure.png", input_sha256="a" * 64)
    assert metadata["recovery"]["quantity"] == "row z-score"
    assert not metadata["recovery"]["original_expression_counts_recovered"]
    assert metadata["scale"]["fit"]["endpoint_coverage"]


@pytest.mark.parametrize("change", ["duplicate", "missing", "reverse"])
def test_review_rejects_duplicate_missing_indices_and_reversed_values(tmp_path, change):
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    result = evaluate_legend(
        config, ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 3))
    )
    path = tmp_path / "review.tsv"
    write_review(path, result, "a" * 64)
    lines = path.read_text().splitlines()
    if change == "duplicate":
        lines.append(lines[1])
    elif change == "missing":
        lines.pop()
    else:
        parts = lines[2].split("\t")
        parts[-1] = "3"
        lines[2] = "\t".join(parts)
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        load_legend_review(path, result, "a" * 64)


def test_clipped_manual_bar_remains_review_only(monkeypatch):
    image, truth = FIXTURE["make_fixture"]()
    fake_engine(monkeypatch, truth)
    # Shift the bar to an image edge; inspect geometry rejects unsupported alpha
    # and clipping separately from the ability to fit recognised tick values.
    config = LegendConfig(colourbar_bbox=(0, 220, 14, 770), colourbar_orientation="vertical")
    monkeypatch.setattr(
        "heatmap_decoder.ocr.recognize",
        lambda *a, **k: OCRRun((), "fake", "fake", "eng", ("eng",), "fake"),
    )
    result = inspect_legend(image, config)
    assert not result.development_eligible
    assert "colourbar_touches_image_edge_possible_clipping" in result.reasons


def test_core_manual_ticks_do_not_import_optional_ocr(monkeypatch):
    monkeypatch.setattr("heatmap_decoder.ocr.recognize", lambda *a, **k: pytest.fail("no OCR"))
    image, truth = FIXTURE["make_fixture"]()
    config = fixture_config(
        truth,
        ScaleConfig(
            source="manual_ticks", anchors=anchors_for(truth["colourbar_bbox"], "vertical", 3)
        ),
    )
    assert extract(image, config).values.shape == (25, 8)


@pytest.mark.parametrize("text", ["1e-999", "10^-999", "1" * 257])
def test_underflow_and_unbounded_numeric_strings_are_unresolved(text):
    assert parse_numeric_tick(text).value is None
    assert parse_numeric_tick("0e-999").value == 0


def test_label_only_file_sources_do_not_trigger_legend(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "heatmap_decoder.legend.inspect_legend", lambda *a, **k: pytest.fail("no legend")
    )
    config = json.loads((ROOT / "examples/m11/labelled_config.json").read_text())
    config["labels"]["rows"]["path"] = str(ROOT / "examples/m11/genes.txt")
    config["labels"]["columns"]["path"] = str(ROOT / "examples/m11/samples.txt")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    result = CliRunner().invoke(
        app,
        [
            "labels",
            str(ROOT / "examples/m11/figure.png"),
            "--config",
            str(path),
            "--out",
            str(tmp_path / "labels"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "labels/labels.tsv").exists()
    assert not (tmp_path / "labels/legend_ticks.tsv").exists()


def test_review_input_cannot_be_overwritten_even_with_overwrite(monkeypatch, tmp_path):
    image, truth = FIXTURE["make_fixture"]()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    fake_engine(monkeypatch, truth)
    config = LegendConfig(colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation="vertical")
    result = inspect_legend_file(figure, config, tmp_path / "review")
    path = tmp_path / "review/legend_review.tsv"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        inspect_legend_file(figure, config, tmp_path / "review", review_path=path, overwrite=True)
    assert path.read_bytes() == before
    assert result.status == "review_required"


def test_targeted_ocr_calls_and_tick_count_are_bounded(monkeypatch):
    image, truth = FIXTURE["make_fixture"]()
    fake_engine(monkeypatch, truth)
    config = LegendConfig(
        colourbar_bbox=truth["colourbar_bbox"],
        colourbar_orientation="vertical",
        legend={"max_ticks": 2},
    )
    result = inspect_legend(image, config)
    assert len([r for r in result.ocr_records if r.purpose == "numeric_line"]) <= 2
    assert len(result.ticks) <= 2
    assert "legend_tick_cap_exceeded" in result.reasons
    assert not result.development_eligible


def test_high_confidence_clipped_numeric_text_remains_unresolved():
    config = LegendConfig(colourbar_bbox=(100, 50, 114, 250), colourbar_orientation="vertical")
    region = (114, 36, 178, 264)
    stroke = LegendTick(index=0, side="right", position=100, position_source="stroke")
    token = OCRToken("0.003", (6, 168, 192, 216), 99)
    run = OCRRun((token,), "fake", "fake", "eng", ("eng",), "localisation")
    result = associate_tokens(config, "right", (stroke,), run, region)
    assert result[0].parsed_value == 0.003
    assert result[0].confidence == 99
    assert "numeric_text_touches_search_edge" in result[0].reasons
    assert result[0].status == "unresolved"


def test_legend_job_deadline_is_explicit_and_never_falls_back(monkeypatch):
    import itertools

    import heatmap_decoder.legend as legend

    image, truth = FIXTURE["make_fixture"]()
    clock = itertools.chain([0.0], itertools.repeat(100.0))
    monkeypatch.setattr(legend.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr("heatmap_decoder.ocr.recognize", lambda *a, **kw: pytest.fail("deadline"))
    result = inspect_legend(
        image,
        LegendConfig(colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation="vertical"),
    )
    assert "legend_ocr_runtime_cap_exceeded" in result.reasons
    assert result.status == "unresolved"
    assert not result.automatic_export_allowed


@pytest.mark.parametrize(
    "raw,value",
    [
        ("—1", -1),
        ("–1.5", -1.5),
        ("‒2", -2),
        ("‐.5", -0.5),
        ("‑ 2", -2),
        ("1e—2", 0.01),
        ("10^–3", 0.001),
    ],
)
def test_observed_dash_is_a_minus_only_in_numeric_sign_positions(raw, value):
    parsed = parse_numeric_tick(raw)
    assert parsed.raw_text == raw
    assert parsed.value == value
    assert "-" in parsed.normalised_text
    assert any(action.startswith("observed_dash_to_minus:U+") for action in parsed.actions)


@pytest.mark.parametrize("raw", ["—", "1—2", "——1", "gene—1", "O.1", "1_2", "—NaN"])
def test_numeric_sign_constraints_do_not_guess_digits_ranges_or_identifiers(raw):
    assert parse_numeric_tick(raw).value is None
