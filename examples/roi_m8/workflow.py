"""Run the fixed, explicitly reviewed M8 example through the installed CLI."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def command(*args, expected=0):
    result = subprocess.run(["heatmap-decoder", *map(str, args)], check=False)
    if result.returncode != expected:
        raise RuntimeError(f"expected exit {expected}, got {result.returncode}")


def workflow(out):
    figure = HERE.parent / "labels/reconstruction.png"
    command("detect", figure, "--config", HERE / "hints.json", "--out", out / "roi", expected=5)
    saved = json.loads((out / "roi/roi_candidates.json").read_text())
    candidates = {c["id"]: c for c in saved["result"]["candidates"]}
    # These IDs are an explicit fixture review supplied in selections.json,
    # not chosen by score or an automatic detector-to-extraction hand-off.
    assert candidates["colourbar-001"]["bbox"] == [1040, 140, 1060, 460]
    assert candidates["labels_rows-left"]["kind"] == "labels_rows"
    assert candidates["labels_columns-above"]["kind"] == "labels_columns"
    command(
        "extract",
        figure,
        "--config",
        out / "roi/config_proposed.json",
        "--roi-review",
        out / "roi/roi_candidates.json",
        "--roi-selections",
        HERE / "selections.json",
        "--out",
        out / "reviewed",
        "--fail-on-qc",
    )
    command(
        "extract",
        figure,
        "--config",
        HERE.parent / "labels/numeric_config.json",
        "--out",
        out / "manual",
        "--fail-on-qc",
    )
    with (out / "reviewed/matrix.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    truth = json.loads((HERE.parent / "labels/truth.json").read_text())
    assert rows[0][1:] == truth["columns"]
    assert [row[0] for row in rows[1:]] == truth["rows"]
    values = np.asarray([list(map(float, row[1:])) for row in rows[1:]])
    np.testing.assert_array_equal(values, np.loadtxt(out / "manual/matrix.csv", delimiter=","))
    print("M8 reviewed workflow passed: exact 7/7 label order, unchanged displayed-scale values.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workflow(args.out)
