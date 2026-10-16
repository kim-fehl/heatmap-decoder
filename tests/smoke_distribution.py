"""CI smoke test of the built wheel in a fresh environment outside the source tree."""

import configparser
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from heatmap_decoder import __version__


def smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    wheel = root / "dist" / f"heatmap_decoder-{__version__}-py3-none-any.whl"
    source = root / "dist" / f"heatmap_decoder-{__version__}.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.startswith("heatmap2matrix/") for name in archive.namelist())
        entry_points_path = next(
            name for name in archive.namelist() if name.endswith(".dist-info/entry_points.txt")
        )
        entry_points = configparser.ConfigParser()
        entry_points.read_string(archive.read(entry_points_path).decode("utf-8"))
        assert dict(entry_points["console_scripts"]) == {
            "heatmap-decoder": "heatmap_decoder.cli:app"
        }
    with tarfile.open(source) as archive:
        names = archive.getnames()
        for suffix in (
            "/examples/figure.png",
            "/examples/m16/config.json",
            "/examples/m16/manifest.json",
            "/docs/progress/M16.md",
            "/docs/nonlinear-calibration.md",
            "/src/heatmap_decoder/gradient.py",
            "/src/heatmap_decoder/ocr_fields.py",
            "/src/heatmap_decoder/rapid_ocr.py",
            "/src/heatmap_decoder/ocr_models/en_PP-OCRv4_rec_infer.onnx",
            "/src/heatmap_decoder/ocr_models/LICENSE",
            "/examples/m16/labels-default.json",
            "/examples/m16/config-padding.json",
            "/examples/m16/labels-padding.json",
            "/docs/images/recognition-ticks-m16.png",
            "/src/heatmap_decoder/diagnostic_style.py",
            "/examples/config.json",
            "/tests/pipeline/test_cli.py",
            "/tests/support/synthetic.py",
            "/tests/conftest.py",
            "/docs/testing.md",
            "/AGENTS.md",
            "/examples/labels/reconstruction.png",
            "/examples/labels/workflow.py",
            "/docs/progress/M6.md",
            "/docs/progress/M9.md",
            "/docs/progress/M10.md",
            "/docs/progress/M11.md",
            "/docs/progress/M12.md",
            "/docs/progress/M13.md",
            "/docs/progress/M14.md",
            "/docs/orientation-proposals.md",
            "/src/heatmap_decoder/orientation.py",
            "/examples/m15/figure.png",
            "/.gitattributes",
            "/docs/palettes.md",
            "/benchmarks/m14/validation.json",
            "/examples/m14/viridis.png",
            "/docs/specs/spec_5.md",
            "/docs/automation.md",
            "/docs/corrections.md",
            "/examples/corrections/figure.png",
            "/examples/corrections/request.json",
            "/src/heatmap_decoder/bar_cores.py",
            "/src/heatmap_decoder/edge_profiles.py",
            "/src/heatmap_decoder/lattice.py",
            "/docs/images/recognition-m12.png",
            "/src/heatmap_decoder/automation.py",
            "/examples/m12/request.json",
            "/benchmarks/m12/test.json",
            "/docs/legend.md",
            "/docs/images/recognition.png",
            "/src/heatmap_decoder/legend.py",
            "/examples/m11/figure.png",
            "/benchmarks/m11/report.json",
            "/src/heatmap_decoder/acceptance.py",
            "/examples/m10/figure.png",
            "/benchmarks/m10/report.json",
            "/benchmarks/m9/manifest.json",
            "/benchmarks/m9/report.json",
        ):
            assert any(name.endswith(suffix) for name in names), suffix
        assert not any("/work/" in name or "/.venv/" in name for name in names)
        assert not any("/src/heatmap2matrix/" in name for name in names)
    with tempfile.TemporaryDirectory(prefix="heatmap_decoder-smoke-") as scratch:
        directory = Path(scratch)
        environment = directory / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        command = scripts / ("heatmap-decoder.exe" if os.name == "nt" else "heatmap-decoder")
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True)
        legacy_command = scripts / ("heatmap2matrix.exe" if os.name == "nt" else "heatmap2matrix")
        assert not legacy_command.exists()
        subprocess.run([str(command), "--version"], check=True, cwd=directory)
        subprocess.run(
            [str(python), "-c", "from heatmap_decoder.ocr_fields import aligned_tick_fields"],
            check=True,
            cwd=directory,
        )
        # The main wheel installs the default label engine/models, without [ocr]
        # or an external Tesseract executable, and its worker runs outside checkout.
        label_command = subprocess.run(
            [
                str(command),
                "labels",
                str(root / "examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg"),
                "--config",
                str(root / "examples/m16/labels-default.json"),
                "--out",
                str(directory / "rapid-default"),
            ],
            check=False,
            cwd=directory,
        )
        assert label_command.returncode == 4  # unresolved identifiers; evidence is written
        default = json.loads((directory / "rapid-default/provenance.json").read_text())
        assert default["config"]["labels"]["ocr"]["engine"] == "rapidocr"
        assert default["axes"][0]["slots"][20]["raw_text"] == "EsGalectin1"
        assert len(default["ocr"]["slot_runs"]) == 41
        assert all(r["run"]["model_sha256"] for r in default["ocr"]["slot_runs"])
        assert all(
            any(name == "en_PP-OCRv4_rec_infer.onnx" for name, _ in r["run"]["model_sha256"])
            for r in default["ocr"]["slot_runs"]
        )
        tick_command = subprocess.run(
            [
                str(command),
                "legend",
                str(root / "examples/diagnostic/figure.png"),
                "--config",
                str(root / "examples/diagnostic/legend.json"),
                "--out",
                str(directory / "rapid-ticks"),
            ],
            check=False,
            cwd=directory,
        )
        assert tick_command.returncode == 6  # proposals require explicit scale review
        ticks = json.loads((directory / "rapid-ticks/legend_provenance.json").read_text())["result"]
        assert [t["parsed_value"] for t in ticks["ticks"] if t["position_source"] == "stroke"] == [
            -3,
            0,
            3,
        ]
        assert all(r["run"]["engine_name"] == "RapidOCR" for r in ticks["ocr_records"])
        subprocess.run(
            [
                str(command),
                "extract",
                str(root / "examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg"),
                "--config",
                str(root / "examples/m16/config.json"),
                "--out",
                str(directory / "m16-partial"),
            ],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [
                str(python),
                "-c",
                "import numpy as np; "
                f"a=np.loadtxt({str(directory / 'm16-partial/matrix.csv')!r},delimiter=','); "
                "assert a.shape==(33,8) and np.isfinite(a).sum()==251",
            ],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [
                str(command),
                "extract",
                str(root / "examples/labels/reconstruction.png"),
                "--config",
                str(root / "examples/labels/file_config.json"),
                "--out",
                str(directory / "file-labels"),
                "--fail-on-qc",
            ],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [
                str(command),
                "extract",
                str(root / "examples/figure.png"),
                "--config",
                str(root / "examples/config.json"),
                "--out",
                str(directory / "results"),
                "--fail-on-qc",
            ],
            check=True,
            cwd=directory,
        )
        roi_figure = root / "benchmarks/m9/figures/grayscale_left-2-clean.png"
        result = subprocess.run(
            [
                str(command),
                "detect",
                str(roi_figure),
                "--config",
                str(root / "benchmarks/m9/review_hints.json"),
                "--out",
                str(directory / "roi"),
            ],
            check=False,
            cwd=directory,
        )
        assert result.returncode == 5
        subprocess.run(
            [
                str(command),
                "extract",
                str(roi_figure),
                "--config",
                str(directory / "roi/config_proposed.json"),
                "--roi-review",
                str(directory / "roi/roi_candidates.json"),
                "--roi-selections",
                str(root / "benchmarks/m9/review_selections.json"),
                "--out",
                str(directory / "reviewed"),
                "--fail-on-qc",
            ],
            check=True,
            cwd=directory,
        )
        subprocess.run(
            [
                str(command),
                "automate",
                str(root / "examples/corrections/figure.png"),
                "--request",
                str(root / "examples/corrections/request.json"),
                "--experimental-export",
                "--out",
                str(directory / "experimental"),
            ],
            check=True,
            cwd=directory,
        )
        check = """
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import heatmap_decoder as h
import importlib.util
assert h.__version__ == sys.argv[1]
assert importlib.util.find_spec('heatmap2matrix') is None
assert importlib.util.find_spec('pytesseract') is None
assert importlib.util.find_spec('matplotlib') is None
experimental = json.loads(Path('experimental/provenance.json').read_text())
assert experimental['publication']['status'] == 'experimental'
assert experimental['automation']['stable_release'] is False
assert pd.read_csv('experimental/matrix.csv', header=None).shape == (20, 6)
config = json.loads(Path(sys.argv[4]).read_text())
hints = h.DetectionHints(rows=config['rows'], columns=config['columns'], requested=(),
    heatmap_bbox=config['heatmap_bbox'], colourbar_bbox=config['colourbar_bbox'],
    colourbar_orientation=config['colourbar_orientation'],
    colourbar_start_value=config['colourbar_start_value'],
    colourbar_end_value=config['colourbar_end_value'])
selective = h.automate(h.load_raster(sys.argv[5]),
    h.AutomationRequest(hints=hints, numeric_only=True))
assert selective.complete.status == 'accepted'
assert not selective.automatic_export_allowed
assert selective.diagnostic_config is not None
from heatmap_decoder.calibration import fit_linear_scale
anchors = (h.ScaleAnchor(position=0, value=2), h.ScaleAnchor(position=100, value=-2))
fit = fit_linear_scale(anchors, (0, 0, 14, 101), "vertical")
assert abs(fit.slope + 4) < 1e-12
assert not fit.tests_linearity
assert Path(h.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
matrix = pd.read_csv('results/matrix.csv', header=None).to_numpy()
truth = pd.read_csv(sys.argv[3], header=None).to_numpy()
assert matrix.shape == (20, 20)
assert np.abs(matrix - truth).mean() < 0.02 * 4
labelled = pd.read_csv('file-labels/matrix.csv', index_col=0)
assert list(labelled.index) == ['TP53', 'HLA-DRA', 'Gapdh', 'BRCA2']
assert list(labelled.columns) == ['Control-1', 'Tumor_2', 'Sample A']
"""
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                check,
                __version__,
                str(environment),
                str(root / "examples/truth.csv"),
                str(root / "examples/config.json"),
                str(root / "examples/figure.png"),
            ],
            check=True,
            cwd=directory,
        )
    print("Source distribution and isolated wheel/CLI smoke checks passed.")


if __name__ == "__main__":
    smoke()
