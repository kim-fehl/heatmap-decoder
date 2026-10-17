"""Cli contracts and fixtures."""

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from typer.testing import CliRunner

from heatmap_decoder import __version__
from heatmap_decoder.cli import app
from tests.support import ROOT as REPO_ROOT

EXAMPLES = REPO_ROOT / "examples"


runner = CliRunner()


def invoke(out, *options):
    return runner.invoke(
        app,
        [
            "extract",
            str(EXAMPLES / "figure.png"),
            "--config",
            str(EXAMPLES / "config.json"),
            "--out",
            str(out),
            *options,
        ],
    )


def test_help_and_version():
    assert "extract" in runner.invoke(app, ["--help"]).output
    assert runner.invoke(app, ["--version"]).output.strip() == f"heatmap-decoder {__version__}"
    assert runner.invoke(app, ["extract", "--help"]).exit_code == 0
    assert runner.invoke(app, []).exit_code == 2


@pytest.mark.parametrize(
    "source",
    [
        {"source": "ocr", "bbox": [0, 32, 24, 272]},
        {"source": "file", "path": "genes.txt"},
    ],
)
def test_label_requests_do_not_silently_publish_numeric_outputs(tmp_path, source, monkeypatch):
    from heatmap_decoder.models import OCRRun

    monkeypatch.setattr(
        "heatmap_decoder.ocr.recognize",
        lambda *args: OCRRun(
            (),
            "fake",
            "fake",
            "eng",
            ("eng",),
            "--oem 1 --psm 11",
        ),
    )
    settings = json.loads((EXAMPLES / "config.json").read_text())
    settings["labels"] = {"rows": source}
    config = tmp_path / "labels.json"
    config.write_text(json.dumps(settings))
    out = tmp_path / "results"
    result = runner.invoke(
        app,
        [
            "extract",
            str(EXAMPLES / "figure.png"),
            "--config",
            str(config),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == (4 if source["source"] == "ocr" else 2)
    assert not (out / "matrix.csv").exists()
    assert not (out / "quality.csv").exists()
    if source["source"] == "ocr":
        assert (out / "labels_review.tsv").exists()


def test_bundled_example_and_pandas_csv(tmp_path):
    out = tmp_path / "results"
    result = invoke(out, "--fail-on-qc")
    assert result.exit_code == 0, result.output
    assert "Flagged cells: 0/400" in result.output
    actual = pd.read_csv(out / "matrix.csv", header=None).to_numpy()
    expected = pd.read_csv(EXAMPLES / "truth.csv", header=None).to_numpy()
    assert actual.shape == (20, 20)
    assert np.abs(actual - expected).mean() < 0.02 * 4
    assert set(pd.read_csv(out / "quality.csv")["flags"]) == {"OK"}
    metadata = json.loads((out / "provenance.json").read_text())
    assert metadata["software"]["version"] == __version__
    assert metadata["input"]["filename"] == "figure.png"
    with Image.open(out / "diagnostic.png") as image:
        assert image.mode == "RGB"


def test_existing_outputs_require_overwrite(tmp_path):
    out = tmp_path / "results"
    assert invoke(out).exit_code == 0
    before = (out / "matrix.csv").read_bytes()
    result = invoke(out)
    assert result.exit_code == 2 and "--overwrite" in result.output
    assert (out / "matrix.csv").read_bytes() == before
    assert invoke(out, "--overwrite").exit_code == 0


@pytest.mark.parametrize("payload", ["{broken", '{"rows": 0}', "null"])
def test_invalid_configuration_is_readable_error(tmp_path, payload):
    path = tmp_path / "config.json"
    path.write_text(payload)
    result = runner.invoke(
        app,
        [
            "extract",
            str(EXAMPLES / "figure.png"),
            "--config",
            str(path),
            "--out",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 2
    assert "Error:" in result.output and "Traceback" not in result.output
    assert not (tmp_path / "results").exists()


def test_missing_input_and_unsupported_raster(tmp_path):
    result = runner.invoke(
        app, ["extract", str(tmp_path / "absent.png"), "--config", str(EXAMPLES / "config.json")]
    )
    assert result.exit_code == 2
    path = tmp_path / "bad.bmp"
    Image.new("RGB", (350, 576)).save(path)
    result = runner.invoke(
        app,
        [
            "extract",
            str(path),
            "--config",
            str(EXAMPLES / "config.json"),
            "--out",
            str(tmp_path / "results"),
        ],
    )
    assert result.exit_code == 2 and "PNG and JPEG" in result.output


def test_mask_qc_exit_code_and_pandas_missing_values(tmp_path):
    mask = np.zeros((576, 350), dtype=bool)
    mask[32:44, 24:36] = True
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    out = tmp_path / "results"
    result = invoke(out, "--mask", str(path), "--fail-on-qc")
    assert result.exit_code == 3
    assert "MISSING=1" in result.output and "Warning:" in result.output
    assert pd.isna(pd.read_csv(out / "matrix.csv", header=None).iloc[0, 0])
    assert json.loads((out / "provenance.json").read_text())["input"]["mask_sha256"]
    assert invoke(out, "--mask", str(path), "--overwrite").exit_code == 0


def test_invalid_masks_and_output_errors(tmp_path):
    mask = tmp_path / "mask.npy"
    np.save(mask, np.zeros((2, 2)))
    result = invoke(tmp_path / "results", "--mask", str(mask))
    assert result.exit_code == 2 and "boolean array" in result.output
    archive = tmp_path / "mask.npz"
    np.savez(archive, mask=np.zeros((2, 2)))
    result = invoke(tmp_path / "results", "--mask", str(archive))
    assert result.exit_code == 2 and "not an archive" in result.output
    out = tmp_path / "not_a_directory"
    out.write_text("keep me")
    result = invoke(out)
    assert result.exit_code == 2 and out.read_text() == "keep me"


def test_configuration_cannot_be_overwritten(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    config = out / "provenance.json"
    config.write_bytes((EXAMPLES / "config.json").read_bytes())
    original = config.read_bytes()
    result = runner.invoke(
        app,
        [
            "extract",
            str(EXAMPLES / "figure.png"),
            "--config",
            str(config),
            "--out",
            str(out),
            "--overwrite",
        ],
    )
    assert result.exit_code == 2 and "overwrite an input" in result.output
    assert config.read_bytes() == original
