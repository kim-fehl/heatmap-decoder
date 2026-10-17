"""CLI diagnostic abstention: assert review artifacts and no new matrix publication."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    directory = parser.parse_args().out
    process = subprocess.run(
        [
            "heatmap-decoder",
            "automate",
            str(ROOT / "examples/m12/figure.png"),
            "--request",
            str(ROOT / "examples/m12/request.json"),
            "--out",
            str(directory),
        ],
        check=False,
    )
    assert process.returncode == 5
    assert (directory / "roi_diagnostic.png").exists()
    assert (directory / "legend_ticks.tsv").exists()
    report = json.loads((directory / "automation.json").read_text())
    assert not report["automatic_export_allowed"]
    assert report["legend"]["fit"]["endpoint_coverage"]
    assert not (directory / "matrix.csv").exists()
    print("M12 selective diagnostics and no-publication workflow passed.")


if __name__ == "__main__":
    main()
