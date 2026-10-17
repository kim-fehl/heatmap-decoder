"""Labels io contracts and fixtures."""

import csv
import hashlib
import json

import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder import GridConfig, extract_labels_file, load_raster
from heatmap_decoder.cli import app
from heatmap_decoder.extract import extract_labels
from heatmap_decoder.io import LABEL_OUTPUT_NAMES
from heatmap_decoder.models import OCRRun, OCRToken
from heatmap_decoder.ocr import OCRUnavailableError
from tests.support import ROOT as REPO_ROOT

FIXTURES = REPO_ROOT / "examples/labels"


def read_tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def settings(rows=True, columns=False):
    payload = json.loads((FIXTURES / "config.json").read_text())
    if not rows:
        payload["labels"].pop("rows")
    if not columns:
        payload["labels"].pop("columns")
    return GridConfig.model_validate(payload)


@pytest.fixture
def fake_ocr(monkeypatch):
    seen = []

    def recognize(image, config):
        seen.append(image)
        assert image.mode == "L"
        # Row 0 accepted; row 1 low confidence; row 2 missing; row 3 collision.
        tokens = (
            OCRToken("", (0, 0, 0, 0), -1),
            OCRToken(' MiX_1\t"quoted" ', (30, 96, 90, 144), 91),
            OCRToken("weak", (30, 336, 90, 384), 50),
            OCRToken("one", (30, 816, 90, 864), 99),
            OCRToken("two", (270, 816, 330, 864), 35),
        )
        return OCRRun(tokens, "fake-engine", "fake-wrapper", "eng", ("eng",), "--oem 1 --psm 11")

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    return seen


def test_exports_template_evidence_and_image_hash(tmp_path, fake_ocr):
    out = tmp_path / "review"
    result = extract_labels_file(FIXTURES / "clean.png", settings(), out)
    assert result.unresolved_count == 3
    assert fake_ocr[0].size == (600, 960)
    assert set(p.name for p in out.iterdir()) == set(LABEL_OUTPUT_NAMES) - {
        "labels_columns_enlarged.png"
    }
    rows = read_tsv(out / "labels.tsv")
    assert len(rows) == 7
    assert list(rows[0]) == [
        "axis",
        "index",
        "raw_text",
        "final_text",
        "suggested_text",
        "confidence",
        "bbox",
        "source",
        "status",
        "duplicate",
        "issues",
    ]
    assert rows[0]["final_text"] == 'MiX_1\t"quoted"'
    assert json.loads(rows[0]["bbox"]) == [20, 172, 40, 188]
    assert rows[1]["status"] == "low_confidence" and rows[1]["final_text"] == ""
    assert rows[2]["bbox"] == rows[2]["confidence"] == rows[2]["raw_text"] == ""
    assert rows[3]["raw_text"] == "one\ntwo" and rows[3]["status"] == "ambiguous"
    review = read_tsv(out / "labels_review.tsv")
    assert len(review) == 4
    assert list(review[0]) == ["axis", "index", "text"]
    assert review[0]["text"] == rows[0]["final_text"]
    assert all(r["text"] == "" for r in review[1:])
    metadata = json.loads((out / "provenance.json").read_text())
    assert (
        metadata["input"]["sha256"]
        == hashlib.sha256((FIXTURES / "clean.png").read_bytes()).hexdigest()
    )
    assert metadata["ocr"]["availability"] == "available"
    assert metadata["ocr"]["runs"]["rows"]["tokens"][1]["bbox"] == [30, 96, 90, 144]
    assert metadata["axes"][0]["tokens"][1]["bbox"] == [20, 172, 40, 188]
    assert metadata["unresolved_slots"] == 3
    with Image.open(out / "labels_diagnostic.png") as image:
        assert image.mode == "RGB" and image.width > 1200


def test_unresolved_cli_writes_review_then_exit4(tmp_path, fake_ocr):
    config = tmp_path / "config.json"
    config.write_text(settings().model_dump_json())
    out = tmp_path / "review"
    args = ["labels", str(FIXTURES / "clean.png"), "--config", str(config), "--out", str(out)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 4, result.output
    assert "Unresolved requested slots: 3" in result.output
    assert (out / "labels_review.tsv").exists()
    assert not (out / "matrix.csv").exists() and not (out / "quality.csv").exists()
    first = (out / "labels.tsv").read_bytes()
    again = CliRunner().invoke(app, args)
    assert again.exit_code == 2 and "--overwrite" in again.output
    assert (out / "labels.tsv").read_bytes() == first
    assert CliRunner().invoke(app, args + ["--overwrite"]).exit_code == 4
    assert (out / "labels.tsv").read_bytes() == first


def test_file_sources_do_not_probe_ocr_and_preserve_strings(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("OCR was probed for a file-only request")

    monkeypatch.setattr("heatmap_decoder.ocr._wrapper", forbidden)
    data = " Same-1 \nSame-1\nMiX_2\nName  With  Spaces\n".encode()
    (tmp_path / "genes.txt").write_bytes(data)
    config = settings().model_copy(update={"labels": None}).model_dump()
    config["labels"] = {"rows": {"source": "file", "path": "genes.txt"}}
    out = tmp_path / "review"
    result = extract_labels_file(
        FIXTURES / "clean.png", GridConfig.model_validate(config), out, base_directory=tmp_path
    )
    assert result.unresolved_count == 0
    rows = read_tsv(out / "labels.tsv")
    assert [r["final_text"] for r in rows[:4]] == [
        "Same-1",
        "Same-1",
        "MiX_2",
        "Name  With  Spaces",
    ]
    assert rows[0]["duplicate"] == rows[1]["duplicate"] == "true"
    assert all(r["status"] == "manual" and r["source"] == "file" for r in rows[:4])
    metadata = json.loads((out / "provenance.json").read_text())
    assert metadata["ocr"]["availability"] == "not_requested"
    assert metadata["file_label_sha256"]["rows"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("strings", [["a"], ["a", "b", "", "d"], ["a", "b", "c", "d", "e"]])
def test_file_count_and_empty_strings_fail_before_output(tmp_path, strings):
    (tmp_path / "genes.txt").write_text("\n".join(strings), encoding="utf-8")
    payload = settings().model_dump()
    payload["labels"] = {"rows": {"source": "file", "path": "genes.txt"}}
    out = tmp_path / "review"
    with pytest.raises(ValueError, match="exactly 4 non-empty"):
        extract_labels_file(
            FIXTURES / "clean.png", GridConfig.model_validate(payload), out, base_directory=tmp_path
        )
    assert not out.exists()


def test_missing_ocr_cli_error_and_numeric_still_succeeds(tmp_path, monkeypatch):
    def unavailable():
        raise OCRUnavailableError("Install heatmap-decoder[ocr] and Tesseract on PATH")

    monkeypatch.setattr("heatmap_decoder.ocr._wrapper", unavailable)
    out = tmp_path / "review"
    result = CliRunner().invoke(
        app,
        [
            "labels",
            str(FIXTURES / "clean.png"),
            "--config",
            str(FIXTURES / "config.json"),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 2 and "Tesseract on PATH" in result.output
    assert not out.exists()
    example = FIXTURES.parent
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(example / "figure.png"),
            "--config",
            str(example / "config.json"),
            "--out",
            str(tmp_path / "numeric"),
        ],
    )
    assert result.exit_code == 0, result.output


def test_input_files_are_protected_even_with_overwrite(tmp_path, fake_ocr):
    figure = tmp_path / "labels_diagnostic.png"
    figure.write_bytes((FIXTURES / "clean.png").read_bytes())
    with pytest.raises(ValueError, match="overwrite an input"):
        extract_labels_file(figure, settings(), tmp_path, overwrite=True)
    assert figure.read_bytes() == (FIXTURES / "clean.png").read_bytes()
    assert fake_ocr == []


def test_no_source_and_bad_roi_fail_without_engine(tmp_path, fake_ocr):
    image = load_raster(FIXTURES / "clean.png")
    with pytest.raises(ValueError, match="requires labels"):
        extract_labels(image, settings(rows=False))
    payload = settings().model_dump()
    payload["labels"]["rows"]["bbox"] = [10, 140, 1201, 460]
    with pytest.raises(ValueError, match="inside the image"):
        extract_labels(image, GridConfig.model_validate(payload))
    assert fake_ocr == []
