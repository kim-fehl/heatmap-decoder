"""Real-image regressions: decode pixels first, then compare independent references."""

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pytesseract")
if os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1" or not shutil.which("tesseract"):
    pytest.skip(
        "Set HEATMAP_DECODER_OCR_TESTS=1 with publication OCR dependencies", allow_module_level=True
    )
pytestmark = pytest.mark.ocr_integration
publication = pytest.importorskip("heatmap_decoder.publication")
decode = publication.decode
DecodeError = publication.DecodeError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def decoded():
    return {
        "covid": decode(
            ROOT / "examples/real_world/publication_headers/plos-ppat-1012368/figure-1c.png"
        ),
        "mir29": decode(
            ROOT / "examples/real_world/publication_headers/plos-pone-0244065/figure-7c.png"
        ),
        "tight": decode(
            ROOT / "examples/real_world/publication_headers/iscience-108897/figure-4a.jpg",
            expected_columns=1237,
        ),
        "psoriasis": decode(
            ROOT / "examples/real_world/publication_headers/plos-pone-0155215/figure-4a.png"
        ),
    }


def canonical(text):
    # Tesseract's English model reads Greek beta as Latin B; retain raw OCR in output.
    return re.sub(r"\s+", "", text).replace("β", "B").casefold()


@pytest.mark.parametrize(
    "key,folder,stem,rows",
    [
        ("covid", "plos-ppat-1012368", "figure-1c", 46),
        ("mir29", "plos-pone-0244065", "figure-7c", 100),
        ("psoriasis", "plos-pone-0155215", "figure-4a", 75),
    ],
)
def test_actual_images_recover_grid_and_annotation_tracks(decoded, key, folder, stem, rows):
    result = decoded[key]
    reference = json.loads(
        (
            ROOT / "examples/real_world/publication_headers" / folder / f"{stem}.annotations.json"
        ).read_text()
    )
    assert result["n_rows"] == rows
    assert result["n_columns"] == reference["n_columns"]
    assert len(result["column_annotations"]) == len(reference["column_annotations"])
    for actual, expected in zip(result["column_annotations"], reference["column_annotations"]):
        assert [(s["start"], s["end"]) for s in actual["segments"]] == [
            (s["start"], s["end"]) for s in expected["segments"]
        ]
        assert [canonical(s["label"]) for s in actual["segments"]] == [
            canonical(s["label"]) for s in expected["segments"]
        ]
    assert all(c["display_label"] is None and c["geo_sample_id"] is None for c in result["columns"])
    matrix = np.array(result["reconstructed_values"])
    assert matrix.shape == (rows, reference["n_columns"])
    assert np.isfinite(matrix).all()
    # This is palette agreement, not validation against biological expression values.
    assert result["value_quality"]["median_rgb_distance"] < 5


def test_tickless_colorbar_uses_label_centers_and_preserves_unlabeled_extremes(decoded):
    bar = decoded["mir29"]["colorbar"]
    assert [a["value"] for a in bar["anchors"]] == [2, 1, 0, -1]
    assert {a["anchor_source"] for a in bar["anchors"]} == {"label_center"}
    for anchor in bar["anchors"]:
        assert anchor["pixel"] == pytest.approx((anchor["bbox"][1] + anchor["bbox"][3]) / 2)
    f = bar["calibration"]
    top_value = f["slope"] * bar["bbox"][1] + f["intercept"]
    bottom_value = f["slope"] * (bar["bbox"][3] - 1) + f["intercept"]
    assert top_value > 2.4  # Outer labels are NOT the endpoints of this gradient.
    assert bottom_value < -1.4
    assert f["max_label_residual"] < 0.01


def test_ticked_colorbar_reads_negative_label_and_uses_ticks(decoded):
    anchors = decoded["covid"]["colorbar"]["anchors"]
    assert [a["value"] for a in anchors] == [-4, 0, 4]
    assert {a["anchor_source"] for a in anchors} == {"tick"}


def test_short_lines_are_extended_to_all_three_columns(decoded):
    result = decoded["mir29"]
    segments = result["column_annotations"][0]["segments"]
    # The second printed rule ends before the final column center. A literal
    # line-coverage/center test would lose that column; the inferred span must not.
    second = segments[1]
    center_of_last = np.mean(result["column_edges"][5:7])
    assert second["line_bbox"][2] < center_of_last
    assert second["end"] == 6
    assert second["inferred_span_pixels"][1] > second["line_bbox"][2]
    assert [i for s in segments for i in range(s["start"], s["end"])] == list(range(18))


def test_labels_inside_colored_panels_need_no_legend(decoded):
    result = decoded["tight"]
    assert result["n_rows"] == 65
    track = result["column_annotations"][0]
    assert track["kind"] == "colored_panel"
    assert track["display_name"] == "cluster"
    assert [s["label"] for s in track["segments"]] == ["cluster1", "cluster2", "cluster3"]
    assert len({tuple(s["color_rgb"]) for s in track["segments"]}) == 3
    for s in track["segments"]:
        assert s["ocr_words"]
        x0, y0, x1, y1 = s["header_bbox"]
        for word in s["ocr_words"]:
            assert x0 <= np.mean(word["bbox"][::2]) <= x1
            assert y0 <= np.mean(word["bbox"][1::2]) <= y1


def test_subpixel_samples_are_not_fabricated_from_pixel_width(decoded):
    result = decoded["tight"]
    assert result["status"] == "resolution_limited"
    assert result["expected_n_columns"] == 1237
    assert result["sample_count_source"] == "caller_provided_metadata"
    assert result["n_columns"] is None
    assert result["column_edges"] is None
    assert result["columns"] is None
    assert result["reconstructed_values"] is None
    assert all(
        s["start"] is None and s["end"] is None for s in result["column_annotations"][0]["segments"]
    )
    assert len(result["matrix_blocks"]) == 3
    for a, b in zip(result["matrix_blocks"], result["matrix_blocks"][1:]):
        assert a[2] < b[0]  # Separators are not counted as data columns.


def test_external_count_cannot_silently_override_visible_grid():
    with pytest.raises(DecodeError, match="conflicts"):
        decode(
            ROOT / "examples/real_world/publication_headers/plos-pone-0244065/figure-7c.png",
            expected_columns=19,
        )


def test_white_text_and_outlined_rectangles_are_two_independent_tracks(decoded):
    result = decoded["psoriasis"]
    tracks = result["column_annotations"]
    assert len(tracks) == 2  # Outline borders must not create extra line tracks.
    assert [s["text_polarity"] for s in tracks[0]["segments"]] == ["dark", "dark"]
    assert [s["text_polarity"] for s in tracks[1]["segments"]] == ["light"] * 5
    assert [(s["label"], s["start"], s["end"]) for s in tracks[0]["segments"]] == [
        ("Acral", 0, 16),
        ("Non-Acral", 16, 50),
    ]
    # Repeated category names are distinct spans, not dictionary keys to merge.
    normals = [s for s in tracks[1]["segments"] if s["label"] == "Normal"]
    assert [(s["start"], s["end"]) for s in normals] == [(0, 7), (16, 40)]
    assert tracks[1]["segments"][1]["raw_text"] == "PP"  # Do not invent a clipped P.


def test_row_side_strip_is_not_used_as_the_color_key(decoded):
    result = decoded["psoriasis"]
    bar = result["colorbar"]
    assert bar["orientation"] == "horizontal"
    assert [a["value"] for a in bar["anchors"]] == [-2, 0, 2]
    assert bar["bbox"][0] < 100  # Actual key at upper left, not the tall row strip.
    assert result["matrix_bbox"][3] == 1530  # Preserve the bottom white margin.


def test_publication_cli_extracts_and_protects_existing_files(tmp_path):
    from typer.testing import CliRunner

    from heatmap_decoder.cli import app

    runner = CliRunner()
    source = ROOT / "examples/real_world/publication_headers/plos-pone-0155215/figure-4a.png"
    out = tmp_path / "publication.json"
    command = ["publication", str(source), "--out", str(out)]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    original = out.read_bytes()
    extraction = json.loads(original)
    assert (extraction["n_rows"], extraction["n_columns"]) == (75, 50)
    assert len(extraction["column_annotations"]) == 2
    refused = runner.invoke(app, command)
    assert refused.exit_code == 2
    assert out.read_bytes() == original
    protected = runner.invoke(
        app, ["publication", str(source), "--out", str(source), "--overwrite"]
    )
    assert protected.exit_code == 2
