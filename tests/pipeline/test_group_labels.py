"""Group labels contracts and fixtures."""

import copy
import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder.cli import app
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels


def request():
    return {
        "start": 17,
        "end": 257,
        "count": 12,
        "tracks": [
            {
                "bbox": [17, 0, 257, 4],
                "source": "lines",
                "regions": [
                    {"bbox": [23, 0, 51, 3], "label": "small"},
                    {"bbox": [63, 0, 191, 3], "label": "large"},
                    {"bbox": [203, 0, 251, 3], "label": "small"},
                ],
            }
        ],
    }


def test_short_unequal_lines_extend_and_repeated_labels_stay_separate():
    result = infer_group_labels(
        np.full((10, 270, 3), 255, np.uint8), GroupLabelsRequest.model_validate(request())
    )
    assert result["complete"]
    assert result["track_names"] == ["track_1"]
    assert [r["tracks"][0]["region_index"] for r in result["records"]] == [0] * 2 + [1] * 7 + [
        2
    ] * 3
    assert all(r["sample_name"] is None for r in result["records"])
    assert len({r["generated_name"] for r in result["records"]}) == 12


@pytest.mark.parametrize("kind", ["large_gap", "overlap", "subpixel", "outer_extent"])
def test_ambiguous_regions_never_fabricate_membership(kind):
    payload = request()
    if kind == "large_gap":
        payload["tracks"][0]["regions"][1]["bbox"][0] = 120
    elif kind == "overlap":
        payload["tracks"][0]["regions"][1]["bbox"][0] = 40
    elif kind == "subpixel":
        payload["count"] = 500
    else:
        payload["tracks"][0]["regions"][0]["bbox"][0] = 40
    result = infer_group_labels(
        np.full((10, 270, 3), 255, np.uint8), GroupLabelsRequest.model_validate(payload)
    )
    assert not result["complete"]
    assert all(r["generated_name"] is None for r in result["records"])
    assert all(r["tracks"][0]["reasons"] for r in result["records"])


@pytest.mark.parametrize("kind", ["blank", "newline", "no_text_box", "duplicate_name", "outside"])
def test_invalid_group_requests_are_rejected(kind):
    payload = request()
    if kind == "blank":
        payload["tracks"][0]["regions"][0]["label"] = " "
    elif kind == "newline":
        payload["tracks"][0]["regions"][0]["label"] = "A\nB"
    elif kind == "no_text_box":
        payload["tracks"][0]["regions"][0].pop("label")
    elif kind == "duplicate_name":
        payload["tracks"][0]["name"] = "track_2"
        payload["tracks"].append(copy.deepcopy(payload["tracks"][0]))
    else:
        payload["tracks"][0]["regions"][0]["bbox"][3] = 500
    with pytest.raises(ValueError):
        infer_group_labels(
            np.full((10, 270, 3), 255, np.uint8), GroupLabelsRequest.model_validate(payload)
        )


def fixture():
    image = np.full((40, 80, 3), 255, dtype=np.uint8)
    image[0:10, 0:40] = [230, 20, 20]
    image[0:10, 40:80] = [20, 180, 20]
    image[20:30, 0:10] = [230, 20, 20]
    image[20:30, 10:20] = [20, 180, 20]
    request = {
        "start": 0,
        "end": 80,
        "count": 4,
        "tracks": [
            {
                "name": "group",
                "bbox": [0, 0, 80, 10],
                "swatches": [
                    {"label": "High_SE", "bbox": [0, 20, 10, 30]},
                    {"label": "Low_SE", "bbox": [10, 20, 20, 30]},
                ],
            }
        ],
    }
    return image, request


def test_names_groups_and_positional_ordinals():
    image, payload = fixture()
    result = infer_group_labels(image, GroupLabelsRequest.model_validate(payload))
    assert result["complete"] and not result["original_sample_ids_recovered"]
    assert [r["generated_name"] for r in result["records"]] == [
        "High_SE__01",
        "High_SE__02",
        "Low_SE__01",
        "Low_SE__02",
    ]


@pytest.mark.parametrize("kind", ["mixed", "unknown", "transparent"])
def test_uncertain_colors_block_complete_names(kind):
    image, payload = fixture()
    if kind == "mixed":
        image[0:10, 10:20] = [20, 180, 20]
    elif kind == "unknown":
        image[0:10, 0:20] = [0, 0, 255]
    else:
        image = np.dstack([image, np.full(image.shape[:2], 255, dtype=np.uint8)])
        image[0:10, 0:20, 3] = 0
    result = infer_group_labels(image, GroupLabelsRequest.model_validate(payload))
    assert not result["complete"]
    assert result["records"][0]["tracks"][0]["category"] is None
    assert all(r["generated_name"] is None for r in result["records"])


def test_similar_legend_and_invalid_regions_rejected():
    image, payload = fixture()
    image[20:30, 10:20] = image[20:30, 0:10]
    with pytest.raises(ValueError, match="too similar"):
        infer_group_labels(image, GroupLabelsRequest.model_validate(payload))
    image, payload = fixture()
    payload["tracks"][0]["swatches"][0]["bbox"] = [0, 20, 10, 100]
    with pytest.raises(ValueError, match="outside image"):
        infer_group_labels(image, GroupLabelsRequest.model_validate(payload))


def test_rows_and_nonuniform_segments():
    image, payload = fixture()
    payload.update(
        axis="rows",
        segments=[
            {"start": 0.0, "end": 40.0, "count": 3},
            {"start": 40.0, "end": 80.0, "count": 1},
        ],
    )
    # Transpose regions as well as the image, preserving the color identity.
    for track in payload["tracks"]:
        x, y, r, b = track["bbox"]
        track["bbox"] = [y, x, b, r]
        for swatch in track["swatches"]:
            x, y, r, b = swatch["bbox"]
            swatch["bbox"] = [y, x, b, r]
    result = infer_group_labels(
        image.transpose(1, 0, 2), GroupLabelsRequest.model_validate(payload)
    )
    assert result["complete"]
    assert [r["generated_name"] for r in result["records"]] == [
        "High_SE__01",
        "High_SE__02",
        "High_SE__03",
        "Low_SE__01",
    ]


def test_complex_labels_and_multiple_tracks():
    image, payload = fixture()
    payload["tracks"][0]["swatches"][0]["label"] = "BMAd-Pnpla2+/+"
    payload["tracks"][0]["swatches"][1]["label"] = "BMAd-Pnpla2-/-"
    second = json.loads(json.dumps(payload["tracks"][0]))
    second["name"] = "diet"
    second["swatches"][0]["label"] = "Ad libitum"
    second["swatches"][1]["label"] = "CR"
    payload["tracks"].append(second)
    result = infer_group_labels(image, GroupLabelsRequest.model_validate(payload))
    assert result["records"][0]["generated_name"] == "BMAd-Pnpla2+/+__Ad libitum__01"
    assert result["records"][-1]["generated_name"] == "BMAd-Pnpla2-/-__CR__02"


def test_cli_outputs_and_stale_names_removed_on_explicit_overwrite(tmp_path):
    image, payload = fixture()
    figure, request, out = tmp_path / "figure.png", tmp_path / "request.json", tmp_path / "out"
    Image.fromarray(image).save(figure)
    request.write_text(json.dumps(payload))
    args = ["group-labels", str(figure), "--request", str(request), "--out", str(out)]
    runner = CliRunner()
    assert runner.invoke(app, args).exit_code == 0
    assert (out / "columns.txt").read_text().splitlines()[0] == "High_SE__01"
    assert runner.invoke(app, args).exit_code == 2
    image[0:10, 0:20] = [0, 0, 255]
    Image.fromarray(image).save(figure)
    assert runner.invoke(app, args + ["--overwrite"]).exit_code == 4
    assert not (out / "columns.txt").exists()
    assert not json.loads((out / "groups.json").read_text())["complete"]


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
