"""Publication headers contracts and fixtures."""

import hashlib
import json
import re

import pytest
from typer.testing import CliRunner

from heatmap_decoder.cli import app
from heatmap_decoder.extract import UnresolvedLabelsError, extract, extract_labels
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels
from heatmap_decoder.io import load_file_labels, load_raster
from heatmap_decoder.legend import inspect_legend
from heatmap_decoder.models import ExtractionConfig, GridConfig, LegendConfig
from tests.support.publication_headers import BASE


def canonical(text):
    return re.sub(r"\s+", "", text).replace("β", "B").casefold()


@pytest.fixture(scope="module")
def groups():
    result = {}
    for path in sorted(BASE.glob("*/groups-request.json")):
        folder = path.parent
        source = next(
            p
            for p in folder.glob("figure-*.*")
            if p.suffix in {".jpg", ".png"} and "reduced" not in p.name
        )
        result[folder.name] = infer_group_labels(
            load_raster(source), GroupLabelsRequest.model_validate_json(path.read_bytes())
        )
    return result


@pytest.mark.parametrize(
    "folder,stem",
    [
        ("plos-ppat-1012368", "figure-1c"),
        ("plos-pone-0244065", "figure-7c"),
        ("plos-pone-0155215", "figure-4a"),
    ],
)
def test_group_ocr_and_geometry_against_independent_visual_reference(groups, folder, stem):
    actual = groups[folder]
    reference = json.loads((BASE / folder / f"{stem}.annotations.json").read_text())
    assert actual["complete"]
    for i, track in enumerate(reference["column_annotations"]):
        expected = [s["label"] for s in track["segments"] for _ in range(s["start"], s["end"])]
        assert [canonical(r["tracks"][i]["category"]) for r in actual["records"]] == [
            canonical(x) for x in expected
        ]
    assert all(r["sample_name"] is None for r in actual["records"])
    assert not actual["original_sample_ids_recovered"]
    assert not actual["legend_labels_are_reviewed_inputs"]


def test_short_rule_extension_ignores_label_width(groups):
    regions = groups["plos-pone-0244065"]["track_evidence"][0]["regions"]
    assert regions[1]["inferred_span"][1] > regions[1]["bbox"][2]
    assert regions[1]["text_bbox"][0] < regions[1]["bbox"][0]
    assert (
        sum(r["tracks"][0]["region_index"] == 1 for r in groups["plos-pone-0244065"]["records"])
        == 3
    )


def test_two_tracks_white_text_and_repeated_normal(groups):
    result = groups["plos-pone-0155215"]
    assert len(result["track_evidence"]) == 2
    assert [r["text_polarity"] for r in result["track_evidence"][1]["regions"]] == ["light"] * 5
    assert result["track_evidence"][1]["regions"][1]["raw_text"] == "PP"
    assert [r["tracks"][1]["region_index"] for r in result["records"]] == [0] * 7 + [1] * 3 + [
        2
    ] * 6 + [3] * 24 + [4] * 10


def test_dense_clusters_read_headers_but_abstain_from_individual_samples(groups):
    result = groups["iscience-108897"]
    assert not result["complete"]
    assert [r["raw_text"] for r in result["track_evidence"][0]["regions"]] == [
        "cluster1",
        "cluster2",
        "cluster3",
    ]
    assert all(r["inferred_span"] is None for r in result["track_evidence"][0]["regions"])
    assert all(r["generated_name"] is None for r in result["records"])
    assert (
        "subpixel_group_boundaries_require_reviewed_counts"
        in result["records"][0]["tracks"][0]["reasons"]
    )


def test_original_zip_image_hash_and_named_samples(groups):
    folder = BASE / "omtn-2021-07-019"
    source = folder / "figure-3a.jpg"
    provenance = json.loads((folder / "article.json").read_text())["image_provenance"]
    assert load_raster(source).shape[:2] == (3039, 939)
    assert len(source.read_bytes()) == 504248
    assert hashlib.sha256(source.read_bytes()).hexdigest() == provenance["received_sha256"]
    result = groups[folder.name]
    assert result["complete"]
    assert [r["display_name"] for r in result["records"]] == (
        folder / "samples.txt"
    ).read_text().splitlines()
    assert [r["tracks"][0]["category"] for r in result["records"]] == ["A"] * 3 + ["C"] * 3 + [
        "B"
    ] * 3
    assert result["visible_sample_names_recovered"]


def test_tickless_legend_uses_numeric_text_centres_with_review_gate():
    folder = BASE / "plos-pone-0244065"
    config = LegendConfig.model_validate_json((folder / "legend-config.json").read_bytes())
    result = inspect_legend(load_raster(folder / "figure-7c.png"), config)
    ticks = sorted(
        (t for t in result.ticks if t.parsed_value is not None), key=lambda t: t.position
    )
    assert [t.parsed_value for t in ticks] == [2, 1, 0, -1]
    for tick in ticks:
        assert tick.position_source == "text_centre"
        assert tick.position == pytest.approx((tick.text_bbox[1] + tick.text_bbox[3]) / 2)
    assert not result.development_eligible
    assert "text_centres_are_weaker_position_evidence" in result.reasons
    assert result.fit is not None
    assert not result.fit.endpoint_coverage


def test_tiny_gene_identifiers_are_measured_and_require_review():
    folder = BASE / "omtn-2021-07-019"
    config = GridConfig.model_validate_json((folder / "labels-config.json").read_bytes())
    image = load_raster(folder / "figure-3a.jpg")
    result = extract_labels(image, config)
    refs = (folder / "genes.txt").read_text().splitlines()
    rows = result.axes[0].slots
    exact = sum(s.raw_text == ref for s, ref in zip(rows, refs, strict=True))
    assert len(rows) == len(refs) == 128
    assert exact >= 80  # Minimum full-resolution regression; not a claim of perfect OCR.
    assert not any(s.status == "accepted" for s in rows)
    assert result.unresolved_count >= 128
    assert [s.raw_text for s in result.axes[1].slots] == (
        folder / "samples.txt"
    ).read_text().splitlines()
    numeric = json.loads((folder / "config.json").read_text())
    numeric["labels"] = config.labels.model_dump(mode="json")
    numeric = ExtractionConfig.model_validate(numeric)
    with pytest.raises(UnresolvedLabelsError):
        extract(image, numeric, label_result=extract_labels(image, numeric))
    reviewed = ExtractionConfig.model_validate_json((folder / "config-reviewed.json").read_bytes())
    files, _ = load_file_labels(reviewed, folder)
    complete = extract(image, reviewed, file_labels=files)
    assert [s.final_text for s in complete.labels.axes[0].slots] == refs


def test_existing_group_cli_writes_sample_names_and_protects_inputs(tmp_path):
    folder = BASE / "omtn-2021-07-019"
    command = [
        "group-labels",
        str(folder / "figure-3a.jpg"),
        "--request",
        str(folder / "groups-request.json"),
        "--out",
        str(tmp_path),
    ]
    runner = CliRunner()
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "columns.txt").read_text() == (folder / "samples.txt").read_text()
    assert runner.invoke(app, command).exit_code == 2
    assert "publication" not in runner.invoke(app, ["--help"]).output
