"""Deterministic M6 publication/review contract; OCR responses are mocked."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from heatmap_decoder import ExtractionConfig, extract, extract_file, load_raster
from heatmap_decoder.cli import app
from heatmap_decoder.extract import UnresolvedLabelsError, extract_labels
from heatmap_decoder.io import load_labels_review, write_csv, write_quality_csv
from heatmap_decoder.labels import CropTransform, apply_review
from heatmap_decoder.models import OCRRun, OCRToken

FIXTURES = Path(__file__).resolve().parents[1] / "examples/labels"
TRUTH = json.loads((FIXTURES / "truth.json").read_text())


def config():
    return ExtractionConfig.model_validate_json((FIXTURES / "extract_config.json").read_text())


def tsv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def review(path, records=None):
    if records is None:
        records = [
            (axis, i, value) for axis, values in TRUTH.items() for i, value in enumerate(values)
        ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["axis", "index", "text"])
        writer.writerows(records)
    return path


@pytest.fixture
def fake_engine(monkeypatch):
    calls = []

    def recognize(image, settings):
        axis = "rows" if image.width == 600 else "columns"
        calls.append(axis)
        bbox = (10, 140, 210, 460) if axis == "rows" else (220, 60, 1000, 125)
        transform = CropTransform(bbox, 3)
        tokens = []
        for i, text in enumerate(TRUTH[axis]):
            box = (
                (50, 172 + i * 80, 170, 188 + i * 80)
                if axis == "rows"
                else (
                    290 + i * 260,
                    80,
                    410 + i * 260,
                    105,
                )
            )
            tokens.append(OCRToken(text, transform.to_crop(box), 95))
        return OCRRun(
            tuple(tokens), "fake-engine", "fake-wrapper", "eng", ("eng",), "--oem 1 --psm 11"
        )

    monkeypatch.setattr("heatmap_decoder.ocr.recognize", recognize)
    return calls


@pytest.fixture
def unresolved_engine(monkeypatch):
    monkeypatch.setattr(
        "heatmap_decoder.ocr.recognize",
        lambda *args: OCRRun(
            (),
            "fake-engine",
            "fake-wrapper",
            "eng",
            ("eng",),
            "--oem 1 --psm 11",
        ),
    )


def test_labelled_values_and_quality_strings_match_numeric_pipeline(tmp_path, fake_engine):
    image = load_raster(FIXTURES / "reconstruction.png")
    baseline = extract(image, config().model_copy(update={"labels": None}))
    result = extract_file(FIXTURES / "reconstruction.png", config(), tmp_path / "out")
    np.testing.assert_array_equal(result.values, baseline.values)
    assert fake_engine == ["rows", "columns"]  # No second OCR pass for reconstruction.
    with (tmp_path / "out/matrix.csv").open(newline="") as stream:
        matrix = list(csv.reader(stream))
    assert matrix[0] == ["gene", *TRUTH["columns"]]
    assert [row[0] for row in matrix[1:]] == TRUTH["rows"]
    np.testing.assert_array_equal(
        np.array([row[1:] for row in matrix[1:]], dtype=float), baseline.values
    )
    for row in csv.DictReader((tmp_path / "out/quality.csv").open()):
        assert row["row_label"] == TRUTH["rows"][int(row["row"])]
        assert row["column_label"] == TRUTH["columns"][int(row["column"])]
    metadata = json.loads((tmp_path / "out/provenance.json").read_text())
    assert metadata["publication"]["numeric_outputs_written"] is True
    assert metadata["labels"]["ocr"]["runs"]["rows"]["engine_version"] == "fake-engine"
    assert metadata["qc_thresholds"]["max_colour_distance"] == config().max_colour_distance


def test_unresolved_evidence_and_manual_review_preserve_raw_results(tmp_path, unresolved_engine):
    out = tmp_path / "blocked"
    with pytest.raises(UnresolvedLabelsError) as caught:
        extract_file(FIXTURES / "reconstruction.png", config(), out)
    assert caught.value.labels.unresolved_count == 7
    assert not (out / "matrix.csv").exists() and not (out / "quality.csv").exists()
    assert set(p.name for p in out.iterdir()) == {
        "labels.tsv",
        "labels_review.tsv",
        "labels_diagnostic.png",
        "labels_rows_enlarged.png",
        "labels_columns_enlarged.png",
        "provenance.json",
    }
    assert all(row["text"] == "" for row in tsv(out / "labels_review.tsv"))
    assert (
        json.loads((out / "provenance.json").read_text())["publication"]["status"]
        == "blocked_labels"
    )
    path = review(tmp_path / "corrected.tsv")
    result = extract_file(
        FIXTURES / "reconstruction.png", config(), tmp_path / "final", labels_review=path
    )
    assert result.labels.unresolved_count == 0
    assert all(
        slot.status == slot.source == "manual" for axis in result.labels.axes for slot in axis.slots
    )
    assert all(slot.raw_text is None for axis in result.labels.axes for slot in axis.slots)
    assert all(
        slot.status == "missing" for axis in result.labels.unreviewed_axes for slot in axis.slots
    )
    labels = json.loads((tmp_path / "final/provenance.json").read_text())["labels"]
    assert labels["review"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert labels["review"]["filename"] == "corrected.tsv"
    assert labels["unreviewed_axes"][0]["slots"][0]["status"] == "missing"


def test_review_can_correct_accepted_text_and_preserve_duplicate_strings(tmp_path, fake_engine):
    records = [(axis, i, value) for axis, values in TRUTH.items() for i, value in enumerate(values)]
    records[0] = ("rows", 0, ' same-0_I1, "é" ')
    records[1] = ("rows", 1, 'same-0_I1, "é"')
    records[4] = ("columns", 0, "Sample  A")
    records[5] = ("columns", 1, "Sample  A")
    path = review(tmp_path / "review.tsv", list(reversed(records)))
    result = extract_file(
        FIXTURES / "reconstruction.png", config(), tmp_path / "out", labels_review=path
    )
    assert result.labels.axes[0].slots[0].raw_text == "TP53"
    assert result.labels.axes[0].slots[0].final_text == 'same-0_I1, "é"'
    assert result.labels.axes[0].slots[0].confidence == 95
    assert result.labels.axes[0].slots[0].duplicate
    assert result.labels.axes[1].slots[0].duplicate
    with (tmp_path / "out/matrix.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    assert rows[0][1:3] == ["Sample  A", "Sample  A"]
    assert rows[1][0] == rows[2][0] == 'same-0_I1, "é"'


@pytest.mark.parametrize(
    "bad",
    [
        "axis\tindex\twrong\nrows\t0\tx\n",
        "axis\tindex\ttext\nrows\t0\tx\nrows\t0\tx\n",
        "axis\tindex\ttext\nrows\t4\tx\n",
        "axis\tindex\ttext\nrows\t-1\tx\n",
        "axis\tindex\ttext\nrows\t1.0\tx\n",
        "axis\tindex\ttext\nrows\tTrue\tx\n",
        "axis\tindex\ttext\nrows\t0\t \n",
        "axis\tindex\ttext\nrows\t0\tx\n",
        "axis\tindex\ttext\ngenes\t0\tx\n",
        "axis\tindex\ttext\nrows\t0\tx\textra\n",
        'axis\tindex\ttext\nrows\t0\t"unterminated\n',
    ],
)
def test_invalid_review_fails_before_ocr_and_any_outputs(tmp_path, bad, fake_engine):
    path = tmp_path / "bad.tsv"
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError):
        extract_file(
            FIXTURES / "reconstruction.png", config(), tmp_path / "out", labels_review=path
        )
    assert not (tmp_path / "out").exists()
    assert fake_engine == []


def test_one_axis_ocr_and_unexpected_review_axis(tmp_path, fake_engine):
    payload = config().model_dump()
    payload["labels"]["columns"] = None
    cfg = ExtractionConfig.model_validate(payload)
    result = extract_file(FIXTURES / "reconstruction.png", cfg, tmp_path / "one-axis")
    assert fake_engine == ["rows"]
    assert [slot.final_text for slot in result.labels.axes[1].slots] == [
        "column_0",
        "column_1",
        "column_2",
    ]
    assert len(tsv(tmp_path / "one-axis/labels_review.tsv")) == 4
    with pytest.raises(ValueError, match="unexpected or out-of-range"):
        load_labels_review(review(tmp_path / "both.tsv"), cfg)


def test_file_sources_publish_without_ocr_and_warn_about_duplicates(tmp_path, monkeypatch):
    def forbidden(*args):
        raise AssertionError("OCR must not be checked for file sources")

    monkeypatch.setattr("heatmap_decoder.ocr._wrapper", forbidden)
    payload = config().model_dump()
    payload["labels"] = {
        "rows": {"source": "file", "path": "genes.txt"},
        "columns": {"source": "file", "path": "samples.txt"},
    }
    (tmp_path / "genes.txt").write_text("same\nsame\nMiX_2\nTP53\n")
    (tmp_path / "samples.txt").write_text("S-1\nName  With  Spaces\nS-1\n")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(FIXTURES / "reconstruction.png"),
            "--config",
            str(path),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "duplicate label text" in result.output
    assert (tmp_path / "out/matrix.csv").exists()
    metadata = json.loads((tmp_path / "out/provenance.json").read_text())
    assert metadata["labels"]["ocr"]["availability"] == "not_requested"
    assert metadata["labels"]["file_label_sha256"]["rows"]


def test_unresolved_cli_never_overwrites_existing_numeric_files(tmp_path, unresolved_engine):
    out = tmp_path / "out"
    out.mkdir()
    for name in ("matrix.csv", "quality.csv"):
        (out / name).write_bytes(b"previous output")
    args = [
        "extract",
        str(FIXTURES / "reconstruction.png"),
        "--config",
        str(FIXTURES / "extract_config.json"),
        "--out",
        str(out),
    ]
    assert CliRunner().invoke(app, args).exit_code == 2
    result = CliRunner().invoke(app, args + ["--overwrite"])
    assert result.exit_code == 4 and "No new matrix/quality outputs" in result.output
    for name in ("matrix.csv", "quality.csv"):
        assert (out / name).read_bytes() == b"previous output"
    metadata = json.loads((out / "provenance.json").read_text())
    assert metadata["publication"]["existing_numeric_outputs"] == ["matrix.csv", "quality.csv"]


def test_review_is_protected_even_with_overwrite_and_numeric_configs_reject_reviews(
    tmp_path, fake_engine
):
    path = review(tmp_path / "labels_review.tsv")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        extract_file(
            FIXTURES / "reconstruction.png", config(), tmp_path, labels_review=path, overwrite=True
        )
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="requires configured"):
        extract_file(
            FIXTURES / "reconstruction.png",
            config().model_copy(update={"labels": None}),
            tmp_path / "out",
            labels_review=path,
        )
    assert fake_engine == []


def test_in_memory_and_writers_enforce_label_gate(tmp_path, unresolved_engine):
    image = load_raster(FIXTURES / "reconstruction.png")
    with pytest.raises(UnresolvedLabelsError):
        extract(image, config())
    numeric = extract(image, config().model_copy(update={"labels": None}))
    evidence = extract_labels(image, config())
    bad = replace(numeric, config=config(), labels=evidence)
    for writer in (write_csv, write_quality_csv):
        with pytest.raises(UnresolvedLabelsError):
            writer(bad, tmp_path / "bad.csv")
        assert not (tmp_path / "bad.csv").exists()
    reviewed = {(axis, i): text for axis, values in TRUTH.items() for i, text in enumerate(values)}
    final = extract(image, config(), reviewed_labels=reviewed)
    np.testing.assert_array_equal(final.values, numeric.values)
    with pytest.raises(ValueError, match="rows then columns"):
        extract(image, config(), label_result=replace(evidence, axes=()))
    applied = apply_review(evidence, reviewed)
    assert evidence.axes[0].slots[0].status == "missing"
    assert applied.axes[0].slots[0].status == "manual"


def test_review_does_not_silently_bypass_missing_ocr(tmp_path, monkeypatch):
    from heatmap_decoder.ocr import OCRUnavailableError

    def unavailable():
        raise OCRUnavailableError("Install heatmap-decoder[ocr] and Tesseract")

    monkeypatch.setattr("heatmap_decoder.ocr._wrapper", unavailable)
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(FIXTURES / "reconstruction.png"),
            "--config",
            str(FIXTURES / "extract_config.json"),
            "--labels-review",
            str(FIXTURES / "reviewed_example.tsv"),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 2 and "Install heatmap-decoder[ocr]" in result.output
    assert not (tmp_path / "out").exists()


def test_labelled_mask_qc_keeps_matching_labels_and_exit3(tmp_path, fake_engine):
    mask = np.zeros((550, 1200), dtype=bool)
    mask[140:220, 220:480] = True
    np.save(tmp_path / "mask.npy", mask)
    result = CliRunner().invoke(
        app,
        [
            "extract",
            str(FIXTURES / "reconstruction.png"),
            "--config",
            str(FIXTURES / "extract_config.json"),
            "--mask",
            str(tmp_path / "mask.npy"),
            "--fail-on-qc",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 3, result.output
    with (tmp_path / "out/matrix.csv").open(newline="") as stream:
        matrix = list(csv.reader(stream))
    assert matrix[1][0] == "TP53" and matrix[1][1] == "nan"
    with (tmp_path / "out/quality.csv").open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["row_label"] == "TP53" and row["column_label"] == "Control-1"
    metadata = json.loads((tmp_path / "out/provenance.json").read_text())
    assert metadata["input"]["mask_sha256"] == hashlib.sha256(mask.tobytes()).hexdigest()
