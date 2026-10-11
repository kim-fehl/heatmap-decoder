"""Run the documented v0.2 fixture workflow; corrections use supplied known truth.

For real figures, a person must fill/verify the review TSV against the image.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def run(directory: Path) -> None:
    fixture = Path(__file__).resolve().parent

    def command(*args: str, expected: int = 0) -> None:
        result = subprocess.run([sys.executable, "-m", "heatmap_decoder", *args], check=False)
        if result.returncode != expected:
            raise RuntimeError(f"Expected exit {expected}, got {result.returncode}: {args}")

    command(
        "labels",
        str(fixture / "reconstruction.png"),
        "--config",
        str(fixture / "extract_config.json"),
        "--out",
        str(directory / "review"),
    )
    command(
        "extract",
        str(fixture / "reconstruction.png"),
        "--config",
        str(fixture / "numeric_config.json"),
        "--out",
        str(directory / "numeric"),
        "--fail-on-qc",
    )
    command(
        "extract",
        str(fixture / "reconstruction.png"),
        "--config",
        str(fixture / "extract_config.json"),
        "--out",
        str(directory / "clean"),
        "--fail-on-qc",
    )
    command(
        "extract",
        str(fixture / "reconstruction_unreadable.png"),
        "--config",
        str(fixture / "extract_config.json"),
        "--out",
        str(directory / "blocked"),
        expected=4,
    )
    assert not (directory / "blocked/matrix.csv").exists()
    assert not (directory / "blocked/quality.csv").exists()
    command(
        "extract",
        str(fixture / "reconstruction_unreadable.png"),
        "--config",
        str(fixture / "extract_config.json"),
        "--labels-review",
        str(fixture / "reviewed_example.tsv"),
        "--out",
        str(directory / "reviewed"),
        "--fail-on-qc",
    )
    numeric = np.loadtxt(directory / "numeric/matrix.csv", delimiter=",")
    truth = json.loads((fixture / "truth.json").read_text())
    for output in ("clean", "reviewed"):
        with (directory / output / "matrix.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows[0][1:] == truth["columns"]
        assert [row[0] for row in rows[1:]] == truth["rows"]
        np.testing.assert_array_equal(np.array([row[1:] for row in rows[1:]], dtype=float), numeric)
    print(
        "Complete v0.2 workflow passed: exact order, blocked unresolved exports, unchanged values."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("work/v0.2-example"),
        help="New output directory; existing files are not overwritten.",
    )
    run(parser.parse_args().out)
