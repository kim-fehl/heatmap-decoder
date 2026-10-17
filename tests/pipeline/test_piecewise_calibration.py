"""Piecewise calibration contracts and fixtures."""

import numpy as np

from heatmap_decoder.extract import extract
from heatmap_decoder.models import CellFlag, ExtractionConfig, OCRConfig, ScaleConfig
from tests.support.piecewise_calibration import anchors


def test_piecewise_export_scope_and_coverage():
    from heatmap_decoder.automation import AutomationRequest, automate
    from heatmap_decoder.models import DetectionHints

    image = np.full((60, 150, 3), 255, dtype=np.uint8)
    bar = np.stack([np.arange(101) * 2, np.full(101, 100), np.full(101, 20)], axis=1)
    image[:8, 10:111] = bar
    image[20:40, 10:50] = bar[50]
    hints = DetectionHints(
        heatmap_bbox=(10, 20, 50, 40),
        rows=1,
        columns=2,
        colourbar_bbox=(10, 0, 111, 8),
        colourbar_orientation="horizontal",
        requested=(),
    )
    scale = ScaleConfig(model="piecewise_linear", source="manual_ticks", anchors=anchors())
    r = automate(
        image,
        AutomationRequest(hints=hints, numeric_only=True, scale=scale),
        experimental_export=True,
    )
    assert r.automatic_export_allowed
    assert r.complete.status == "accepted"
    partial = scale.model_copy(update={"anchors": anchors((20, 60, 100))})
    r = automate(
        image,
        AutomationRequest(hints=hints, numeric_only=True, scale=partial),
        experimental_export=True,
    )
    assert not r.automatic_export_allowed
    assert any("unsupported_calibration_span" in d.reasons for d in r.decisions)


def test_ocr_retry_budget_and_raw_conflicts(monkeypatch):
    from heatmap_decoder.extract import extract_labels
    from heatmap_decoder.models import GridConfig, LabelsConfig, OCRLabelSource, OCRRun, OCRToken

    count = 0

    def engine(image, settings, **kwargs):
        nonlocal count
        count += 1
        text = "alpha" if count % 3 != 0 else "beta"
        return OCRRun(
            (OCRToken(text, (3, 3, 15, 15), 95),), "fake", "fake", "eng", ("eng",), "text_line"
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", engine)
    c = GridConfig(
        heatmap_bbox=(0, 0, 30, 600),
        rows=60,
        columns=1,
        labels=LabelsConfig(
            rows=OCRLabelSource(source="ocr", bbox=(40, 0, 80, 600), recognition="slots"),
            ocr=OCRConfig(retry_variants=("contrast", "threshold")),
        ),
    )
    r = extract_labels(np.full((600, 80, 3), 255, dtype=np.uint8), c)
    assert count == 128
    assert len(r.skipped_ocr_attempts) == 52
    assert r.axes[0].slots[0].status == "ambiguous"
    assert len(r.axes[0].candidates) > 60
    assert r.slot_ocr_runs[2][3].preprocessing == "threshold"
    assert r.axes[0].slots[-1].status == "missing"


def test_reviewed_piecewise_preserves_failed_linear_comparison(tmp_path):
    from heatmap_decoder.io import load_legend_review
    from heatmap_decoder.models import LegendConfig, LegendResult, LegendTick

    knots = anchors()
    config = LegendConfig(
        colourbar_bbox=(10, 0, 111, 8),
        colourbar_orientation="horizontal",
        scale_model="piecewise_linear",
    )
    result = LegendResult(
        config=config,
        ticks=tuple(
            LegendTick(index=i, side="below", position=a.position, position_source="stroke")
            for i, a in enumerate(knots)
        ),
        status="review_required",
        reasons=("raw OCR unresolved",),
    )
    path = tmp_path / "review.tsv"
    path.write_text(
        "index\timage_sha256\tinclude\tposition\tvalue\n"
        + "".join(f"{i}\t{'a' * 64}\tyes\t{a.position}\t{a.value}\n" for i, a in enumerate(knots))
    )
    reviewed = load_legend_review(path, result, "a" * 64)
    assert reviewed.status == "reviewed"
    assert reviewed.calibration_evidence["model"] == "piecewise_linear"
    assert not reviewed.fit.within_residual_limits
    assert reviewed.unreviewed_reasons == ("raw OCR unresolved",)
    assert all(a.origin == "reviewed" for a in reviewed.reviewed_anchors)


def test_ocr_pixel_budget_checked_before_resize(monkeypatch):
    from heatmap_decoder.extract import extract_labels
    from heatmap_decoder.models import GridConfig, LabelsConfig, OCRLabelSource

    def forbidden(*args):
        raise AssertionError("must not allocate an over-budget crop")

    monkeypatch.setattr("heatmap_decoder.labels.CropTransform.prepare", forbidden)
    config = GridConfig(
        heatmap_bbox=(0, 0, 10, 10),
        rows=1,
        columns=1,
        labels=LabelsConfig(
            rows=OCRLabelSource(source="ocr", bbox=(10, 0, 20, 10), recognition="slots"),
            ocr=OCRConfig(upscale_factor=1000),
        ),
    )
    result = extract_labels(np.full((10, 20, 3), 255, dtype=np.uint8), config)
    assert len(result.skipped_ocr_attempts) == 1
    assert result.axes[0].slots[0].status == "missing"
    assert not result.slot_ocr_runs


def test_partial_span_is_unresolved():
    image = np.zeros((30, 111, 3), dtype=np.uint8)
    bar = np.stack([np.arange(101) * 2, np.full(101, 100), np.full(101, 20)], axis=1)
    image[:8, 10:111] = bar
    image[15:30, :30] = bar[0]
    image[15:30, 30:60] = bar[50]
    c = ExtractionConfig(
        heatmap_bbox=(0, 15, 60, 30),
        rows=1,
        columns=2,
        colourbar_bbox=(10, 0, 111, 8),
        colourbar_orientation="horizontal",
        scale=ScaleConfig(
            model="piecewise_linear", source="manual_ticks", anchors=anchors((20, 60, 100))
        ),
    )
    r = extract(image, c)
    assert np.isnan(r.values[0, 0])
    assert r.flags[0, 0] & CellFlag.UNCALIBRATED
    assert np.isfinite(r.values[0, 1])
    assert r.colour_evidence[0].components[0].minimum_value is None
