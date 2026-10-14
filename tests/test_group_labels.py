"""Categorical colors provide group/position names without claiming sample IDs."""

import json

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder.cli import app
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels


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
