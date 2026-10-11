"""Explicitly reviewed v0.3 CLI example and a blocked uniform-figure example."""

import argparse
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
    figure = HERE / "figures/grayscale_left-2-clean.png"
    command(
        "detect", figure, "--config", HERE / "review_hints.json", "--out", out / "roi", expected=5
    )
    saved = json.loads((out / "roi/roi_candidates.json").read_text())
    candidates = {candidate["id"]: candidate for candidate in saved["result"]["candidates"]}
    # Explicit fixed-fixture review, not rank-based acceptance or an auto policy.
    assert candidates["heatmap-001"]["bbox"] == [160, 100, 424, 220]
    assert candidates["colourbar-001"]["bbox"] == [90, 100, 106, 220]
    assert not saved["automation_policy"]["automatic_roi_enabled"]
    command(
        "extract",
        figure,
        "--config",
        out / "roi/config_proposed.json",
        "--roi-review",
        out / "roi/roi_candidates.json",
        "--roi-selections",
        HERE / "review_selections.json",
        "--out",
        out / "reviewed",
        "--fail-on-qc",
    )
    config = json.loads((out / "roi/config_proposed.json").read_text())
    config.update(
        heatmap_bbox=[160, 100, 424, 220],
        colourbar_bbox=[90, 100, 106, 220],
        colourbar_orientation="vertical",
    )
    manual = out / "manual.json"
    manual.write_text(json.dumps(config), encoding="utf-8")
    command("extract", figure, "--config", manual, "--out", out / "manual", "--fail-on-qc")
    np.testing.assert_array_equal(
        np.loadtxt(out / "reviewed/matrix.csv", delimiter=","),
        np.loadtxt(out / "manual/matrix.csv", delimiter=","),
    )
    command(
        "detect",
        HERE / "figures/challenge-uniform.png",
        "--config",
        HERE / "review_hints.json",
        "--out",
        out / "blocked-roi",
        expected=5,
    )
    command(
        "extract",
        HERE / "figures/challenge-uniform.png",
        "--config",
        out / "blocked-roi/config_proposed.json",
        "--roi-review",
        out / "blocked-roi/roi_candidates.json",
        "--out",
        out / "blocked-matrix",
        expected=2,
    )
    assert (out / "blocked-roi/roi_diagnostic.png").exists()
    assert not (out / "blocked-matrix/matrix.csv").exists()
    assert not (out / "blocked-matrix/quality.csv").exists()
    print(
        "M9 review-only workflow passed: identical numbers; "
        "unresolved geometry retained without matrix publication."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    workflow(args.out)
