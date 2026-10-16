"""Legend contracts and fixtures."""

import csv
import hashlib
import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder import extract
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
    search_regions,
)
from heatmap_decoder.models import (
    LegendConfig,
    LegendTick,
    OCRRun,
    OCRToken,
    ScaleConfig,
)
from tests.support.legend import ROOT, anchors_for, fixture_config, ticks_from_anchors
from tests.support.synthetic import legend_fixture


@pytest.mark.parametrize("orientation", ["vertical", "horizontal"])
@pytest.mark.parametrize("count", [2, 3, 5, 7])
def test_known_large_matrix_reconstructed_from_spatial_fit(orientation, count):
    image, truth = legend_fixture(orientation=orientation, count=count, nonuniform=True)
    anchors = anchors_for(truth["colourbar_bbox"], orientation, count, nonuniform=True)
    config = fixture_config(
        truth, ScaleConfig(source="manual_ticks", anchors=anchors, quantity="row z-score")
    )
    result = extract(image, config)
    error = np.abs(result.values - np.array(truth["expected_matrix"]))
    assert np.isfinite(result.values).all()
    assert error.mean() / 4 < 0.02
    assert error.max() / 4 < 0.02


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
    image, truth = legend_fixture(orientation=orientation, side=side, count=5)
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
    config = LegendConfig(
        colourbar_bbox=(100, 50, 114, 250),
        colourbar_orientation="vertical",
        legend={"ocr": {"engine": "tesseract"}},
    )
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
    image, truth = legend_fixture()
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
    image, truth = legend_fixture()
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


def test_endpoint_matches_are_potential_bounds_not_original_expression(tmp_path):
    from heatmap_decoder.io import provenance, write_quality_csv

    image, truth = legend_fixture()
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
    image, truth = legend_fixture()
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
    image, truth = legend_fixture()
    config = fixture_config(
        truth,
        ScaleConfig(
            source="manual_ticks", anchors=anchors_for(truth["colourbar_bbox"], "vertical", 3)
        ),
    )
    assert extract(image, config).values.shape == (25, 8)


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
    image, truth = legend_fixture()
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
    image, truth = legend_fixture()
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
    config = LegendConfig(
        colourbar_bbox=(100, 50, 114, 250),
        colourbar_orientation="vertical",
        legend={"ocr": {"engine": "tesseract"}},
    )
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

    image, truth = legend_fixture()
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


def test_horizontal_tickless_numbers_remain_separate_and_require_review():
    config = LegendConfig.model_validate(
        {
            "colourbar_bbox": [10, 10, 111, 20],
            "colourbar_orientation": "horizontal",
            "legend": {"tick_position_source": "text_centres", "ocr": {"upscale_factor": 1}},
        }
    )
    tokens = tuple(
        OCRToken(text, box, 99, (1, 1, 1))
        for text, box in [("-2", (8, 5, 12, 15)), ("0", (58, 5, 62, 15)), ("2", (108, 5, 112, 15))]
    )
    # Region coordinates are original pixels; x-offset shifts centers to 20,70,120.
    run = OCRRun(tokens, "fake", "fake", "eng", ("eng",), "fake")
    ticks = associate_tokens(config, "below", (), run, (0, 20, 130, 50))
    assert [t.parsed_value for t in ticks] == [-2, 0, 2]
    assert [t.position for t in ticks] == [10, 60, 110]
    result = evaluate_legend(config, ticks, ())
    assert result.status == "review_required"
    assert not result.development_eligible
    assert "text_centres_are_weaker_position_evidence" in result.reasons


@pytest.mark.parametrize("orientation,side", [("vertical", "below"), ("horizontal", "left")])
def test_invalid_legend_side_cannot_select_unrelated_text(orientation, side):
    with pytest.raises(ValueError, match="orientation"):
        LegendConfig.model_validate(
            {
                "colourbar_bbox": [1, 1, 10, 100],
                "colourbar_orientation": orientation,
                "legend": {"label_side": side},
            }
        )


def test_ocr_anchor_evidence_cannot_be_loaded_as_manual_acceptance():
    image, truth = legend_fixture()
    anchors = tuple(
        a.model_copy(update={"origin": "ocr", "status": "suggested"})
        for a in anchors_for(truth["colourbar_bbox"], "vertical")
    )
    config = fixture_config(truth, ScaleConfig(source="manual_ticks", anchors=anchors))
    with pytest.raises(ValueError, match="explicitly supplied/reviewed"):
        extract(image, config)
    with pytest.raises(ValueError, match="requires at least two"):
        ScaleConfig(source="manual_ticks", anchors=())
