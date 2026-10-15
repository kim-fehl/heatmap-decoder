"""Deterministic regressions for label review, geometry, and tickless calibration."""

import numpy as np
import pytest

from heatmap_decoder.extract import extract_labels
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels
from heatmap_decoder.legend import associate_tokens, evaluate_legend
from heatmap_decoder.models import GridConfig, LegendConfig, OCRRun, OCRToken


def test_trimmed_rotated_slot_preserves_original_coordinates_and_review(monkeypatch):
    image = np.full((80, 50, 3), 255, np.uint8)
    image[40:60, 15:25] = 0
    config = GridConfig.model_validate(
        {
            "heatmap_bbox": [10, 0, 30, 20],
            "rows": 1,
            "columns": 1,
            "labels": {
                "columns": {
                    "source": "ocr",
                    "recognition": "slots",
                    "bbox": [10, 30, 30, 80],
                    "text_rotation_degrees": 90,
                    "trim_whitespace": True,
                    "require_review": True,
                },
                "ocr": {"engine": "rapidocr", "upscale_factor": 2},
            },
        }
    )
    seen = []

    def recognize(crop, settings, **kwargs):
        seen.append(crop.size)
        return OCRRun(
            (OCRToken("NC1", (0, 0, crop.width, crop.height), 99, (1, 1, 1)),),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "fake",
            geometry_kind="source_field",
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    result = extract_labels(image, config)
    assert seen == [(44, 24)]
    assert result.slot_ocr_runs[0][2] == (14, 39, 26, 61)
    slot = result.axes[1].slots[0]
    assert slot.raw_text == "NC1"
    assert slot.final_text is None and slot.status == "unreviewed"
    assert slot.bbox == (14, 39, 26, 61)
    assert "configured_identifier_review_required" in slot.issues


def test_sample_file_names_do_not_become_generated_group_names():
    image = np.full((20, 40, 3), 255, np.uint8)
    request = GroupLabelsRequest.model_validate(
        {
            "start": 0,
            "end": 40,
            "count": 2,
            "tracks": [
                {
                    "bbox": [0, 0, 40, 5],
                    "source": "rectangles",
                    "regions": [
                        {"bbox": [0, 0, 20, 5], "label": "A"},
                        {"bbox": [20, 0, 40, 5], "label": "B"},
                    ],
                }
            ],
            "grid": {
                "heatmap_bbox": [0, 5, 40, 20],
                "rows": 1,
                "columns": 2,
                "labels": {"columns": {"source": "file", "path": "samples.txt"}},
            },
        }
    )
    result = infer_group_labels(image, request, file_labels={"columns": ["NC1", "Dormancy1"]})
    assert result["complete"]
    assert [r["display_name"] for r in result["records"]] == ["NC1", "Dormancy1"]
    assert [r["generated_name"] for r in result["records"]] == ["A__01", "B__01"]
    assert result["visible_sample_names_recovered"]


def test_sample_grid_must_use_same_nonuniform_geometry():
    with pytest.raises(ValueError, match="segments"):
        GroupLabelsRequest.model_validate(
            {
                "start": 0,
                "end": 40,
                "count": 2,
                "tracks": [
                    {
                        "bbox": [0, 0, 40, 5],
                        "source": "rectangles",
                        "regions": [{"bbox": [0, 0, 40, 5], "label": "A"}],
                    }
                ],
                "grid": {
                    "heatmap_bbox": [0, 5, 40, 20],
                    "rows": 1,
                    "columns": 2,
                    "columns_segments": [
                        {"start": 0.0, "end": 10.0, "count": 1},
                        {"start": 20.0, "end": 40.0, "count": 1},
                    ],
                    "labels": {"columns": {"source": "file", "path": "samples.txt"}},
                },
            }
        )


def test_group_ocr_pixel_budget_abstains_without_engine_call(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("oversize OCR crop must not reach the engine")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", unexpected)
    request = GroupLabelsRequest.model_validate(
        {
            "start": 0,
            "end": 40,
            "count": 2,
            "ocr": {"upscale_factor": 1000},
            "tracks": [
                {
                    "bbox": [0, 0, 40, 10],
                    "source": "rectangles",
                    "regions": [{"bbox": [0, 0, 40, 10]}],
                }
            ],
        }
    )
    result = infer_group_labels(np.full((20, 40, 3), 255, np.uint8), request)
    assert not result["complete"]
    assert result["track_evidence"][0]["regions"][0]["status"] == "ocr_budget_exhausted"


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
