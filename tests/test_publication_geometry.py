"""Group geometry extends the existing categorical workflow without assuming equal sizes."""

import copy

import numpy as np
import pytest

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
