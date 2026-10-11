"""CI smoke test of the built wheel in a fresh environment outside the source tree."""

import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from heatmap_decoder import __version__


def smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    wheel = root / "dist" / f"heatmap_decoder-{__version__}-py3-none-any.whl"
    source = root / "dist" / f"heatmap_decoder-{__version__}.tar.gz"
    with tarfile.open(source) as archive:
        names = archive.getnames()
        for suffix in ("/examples/figure.png", "/examples/config.json", "/tests/test_cli.py"):
            assert any(name.endswith(suffix) for name in names), suffix
        assert not any("/work/" in name or "/.venv/" in name for name in names)
    with tempfile.TemporaryDirectory(prefix="heatmap_decoder-smoke-") as scratch:
        directory = Path(scratch)
        environment = directory / "venv"
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
        scripts = environment / ("Scripts" if os.name == "nt" else "bin")
        python = scripts / ("python.exe" if os.name == "nt" else "python")
        command = scripts / ("heatmap-decoder.exe" if os.name == "nt" else "heatmap-decoder")
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True)
        subprocess.run([str(command), "--version"], check=True, cwd=directory)
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
        check = """
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import heatmap_decoder as h
assert h.__version__ == sys.argv[1]
assert Path(h.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
matrix = pd.read_csv('results/matrix.csv', header=None).to_numpy()
truth = pd.read_csv(sys.argv[3], header=None).to_numpy()
assert matrix.shape == (20, 20)
assert np.abs(matrix - truth).mean() < 0.02 * 4
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
            ],
            check=True,
            cwd=directory,
        )
    print("Source distribution and isolated wheel/CLI smoke checks passed.")


if __name__ == "__main__":
    smoke()
