"""Range/direction, selective decisions and observed string evidence regressions."""

import json
import os
import runpy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont
from typer.testing import CliRunner

from heatmap_decoder import (
    AutomationRequest,
    DetectionHints,
    ExtractionConfig,
    ScaleAnchor,
    ScaleConfig,
    automate,
    extract,
    extract_labels,
)
from heatmap_decoder.cli import app
from heatmap_decoder.io import label_provenance
from heatmap_decoder.legend import associate_tokens
from heatmap_decoder.models import GridConfig, LegendConfig, LegendTick, OCRRun, OCRToken

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = runpy.run_path(str(ROOT / "examples/m12/generate.py"))


def case_for(index):
    return dict(
        index=index,
        seed=129000 + index,
        family=("vertical_flank", "horizontal_cap", "cross_axis")[index % 3],
        degradation="clean",
    )


try:
    ImageFont.truetype("DejaVuSans.ttf", 18)
    FONT_AVAILABLE = True
except OSError:
    FONT_AVAILABLE = False


@pytest.mark.skipif(not FONT_AVAILABLE, reason="synthetic renderer requires DejaVuSans")
@pytest.mark.parametrize("index", range(45))
def test_realistic_ranges_and_anchor_positions(index):
    image, truth = FIXTURE["make_fixture"](case_for(index))
    anchors = tuple(ScaleAnchor(position=t["position"], value=t["value"]) for t in truth["ticks"])
    config = ExtractionConfig(
        heatmap_bbox=truth["heatmap_bbox"],
        colourbar_bbox=truth["colourbar_bbox"],
        colourbar_orientation=truth["orientation"],
        rows=truth["rows"],
        columns=truth["columns"],
        scale=ScaleConfig(source="manual_ticks", anchors=anchors),
    )
    recovered = extract(image, config)
    errors = np.abs(recovered.values - np.asarray(truth["expected_matrix"])) / abs(
        truth["end_value"] - truth["start_value"]
    )
    assert np.isfinite(recovered.values).all()
    assert errors.mean() < 0.02
    assert errors.max() < 0.05
    assert recovered.values.shape in ((20, 6), (25, 8), (30, 10))


def test_split_and_derivatives_are_fixed_by_base():
    groups = {}
    for split in ("development", "validation", "test"):
        for case in FIXTURE["cases"](split):
            groups.setdefault(case["seed"], set()).add(split)
    assert len(groups) == 75
    assert all(len(splits) == 1 for splits in groups.values())
    prereg = json.loads((ROOT / "benchmarks/m12/preregister.json").read_text())
    assert prereg["ranges"] == [[-4, 4], [-3, 3], [-2, 2], [-4, 2], [-3, 4]]


@pytest.mark.parametrize("factor", [1, 3])
def test_nearby_horizontal_tick_words_do_not_merge(factor):
    config = LegendConfig(
        colourbar_bbox=(20, 30, 91, 46),
        colourbar_orientation="horizontal",
        legend={"ocr": {"upscale_factor": factor}},
    )
    strokes = tuple(
        LegendTick(index=i, side="below", position=p, position_source="stroke")
        for i, p in enumerate((20, 55, 90))
    )
    # Adjacent labels meet the old line-join heuristic but belong to distinct strokes.
    run = OCRRun(
        tuple(
            OCRToken(
                text, (factor * (p - 10), 25 * factor, factor * (p - 4), 40 * factor), 95, (1, 1, 1)
            )
            for text, p in zip(("-4", "0", "2"), (20, 55, 90), strict=True)
        ),
        "mock",
        "mock",
        "eng",
        ("eng",),
        "sparse",
    )
    observed = associate_tokens(config, "below", strokes, run, (7, 46, 108, 110))
    assert [t.raw_text for t in observed] == ["-4", "0", "2"]
    assert [t.parsed_value for t in observed] == [-4, 0, 2]
    assert all(t.status == "suggested" for t in observed)


def test_conflicting_identity_and_scale_requests_are_rejected():
    hints = DetectionHints(rows=20, columns=6, requested=("heatmap", "colourbar", "labels_rows"))
    with pytest.raises(ValueError, match="identifiers"):
        AutomationRequest(hints=hints, numeric_only=True)
    endpoints = hints.model_copy(
        update={
            "requested": ("heatmap", "colourbar"),
            "colourbar_start_value": 4,
            "colourbar_end_value": -4,
        }
    )
    with pytest.raises(ValueError, match="conflict"):
        AutomationRequest(hints=endpoints, numeric_only=True, scale=ScaleConfig(source="ocr_ticks"))


def test_supplied_and_reviewed_capabilities_compose_without_export():
    base = runpy.run_path(str(ROOT / "examples/m10/generate.py"))
    image, truth = base["make_layout"](rows=20, columns=6)
    bar = truth["colourbar_bbox"]
    request = AutomationRequest(
        hints=DetectionHints(
            rows=20, columns=6, requested=(), heatmap_bbox=truth["heatmap_bbox"], colourbar_bbox=bar
        ),
        numeric_only=True,
        scale=ScaleConfig(
            source="manual_ticks",
            anchors=(
                ScaleAnchor(position=bar[1], value=2),
                ScaleAnchor(position=bar[3] - 1, value=-4),
            ),
        ),
    )
    result = automate(image, request)
    decisions = {d.requirement: d for d in result.decisions}
    assert decisions["rows"].status == decisions["columns"].status == "accepted"
    assert decisions["scale"].status == "accepted"
    assert result.complete.status == "accepted"
    assert result.diagnostic_config is not None
    assert not result.automatic_export_allowed


def test_inferred_axes_resolve_with_measured_common_pitch():
    base = runpy.run_path(str(ROOT / "examples/m10/generate.py"))
    image, _ = base["make_layout"](rows=20, columns=6)
    request = AutomationRequest(
        hints=DetectionHints(
            infer_rows=True,
            infer_columns=True,
            requested=("heatmap", "colourbar"),
            colourbar_start_value=4,
            colourbar_end_value=-2,
        ),
        numeric_only=True,
    )
    result = automate(image, request)
    assert result.diagnostic_config is not None
    assert result.diagnostic_config.rows == 20
    assert result.diagnostic_config.columns == 6
    assert not result.automatic_export_allowed
    decisions = {d.requirement: d for d in result.decisions}
    assert decisions["rows"].status == "accepted"
    assert decisions["columns"].status == "accepted"
    assert decisions["rows"].evidence["observations"]["count_is_confirming_signal"] is False


def test_cli_abstention_keeps_existing_matrix(tmp_path):
    base = runpy.run_path(str(ROOT / "examples/m10/generate.py"))
    image, truth = base["make_layout"](rows=20, columns=6)
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    config = tmp_path / "request.json"
    config.write_text(
        json.dumps(
            dict(
                numeric_only=True,
                hints=dict(
                    rows=20,
                    columns=6,
                    heatmap_bbox=truth["heatmap_bbox"],
                    colourbar_bbox=truth["colourbar_bbox"],
                    requested=[],
                    colourbar_start_value=4,
                    colourbar_end_value=-2,
                ),
            )
        )
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "matrix.csv").write_text("existing\n")
    result = CliRunner().invoke(
        app, ["automate", str(figure), "--request", str(config), "--out", str(out)]
    )
    assert result.exit_code == 5, result.output
    assert (out / "matrix.csv").read_text() == "existing\n"
    assert (out / "roi_diagnostic.png").exists()
    assert not json.loads((out / "automation.json").read_text())["automatic_export_allowed"]
    again = CliRunner().invoke(
        app, ["automate", str(figure), "--request", str(config), "--out", str(out)]
    )
    assert again.exit_code == 2


@pytest.mark.ocr_integration
@pytest.mark.skipif(
    os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1", reason="local engine opt-in"
)
def test_slot_ocr_confusable_genes_and_raw_provenance():
    image, truth = FIXTURE["make_fixture"](case_for(0))
    boxes = np.asarray([t["bbox"] for t in truth["tokens"]["rows"]])
    region = (
        int(boxes[:, 0].min()) - 5,
        int(boxes[:, 1].min()) - 5,
        int(boxes[:, 2].max()) + 5,
        int(boxes[:, 3].max()) + 5,
    )
    config = GridConfig(
        heatmap_bbox=truth["heatmap_bbox"],
        rows=20,
        columns=6,
        labels={"rows": {"source": "ocr", "bbox": region, "recognition": "slots"}},
    )
    result = extract_labels(image, config)
    assert [s.raw_text for s in result.axes[0].slots[:3]] == ["HSP70A", "HSP70B", "HSP70D"]
    assert len(result.slot_ocr_runs) == 20
    provenance = label_provenance(
        result, input_filename="figure.png", input_sha256="hash", image_shape=image.shape
    )
    assert len(provenance["ocr"]["slot_runs"]) == 20
    assert provenance["ocr"]["availability"] == "available"
    assert config.labels.ocr.engine == "rapidocr"
    assert all(r[3].mode == "rapidocr-recognize-field" for r in result.slot_ocr_runs)
    assert all(r[3].model_sha256 for r in result.slot_ocr_runs)
