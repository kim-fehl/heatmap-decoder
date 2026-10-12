"""Generate the worked viridis and explicit RGB teal-black-red examples."""

import json
import runpy
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = runpy.run_path(str(ROOT / "benchmarks/m14/fixtures.py"))


def generate():
    directory = Path(__file__).resolve().parent
    for palette in ("viridis", "teal_black_red"):
        case = dict(palette=palette, layout=0, variant="clean", seed=314000, negative=None)
        image, truth = FIXTURES["fixture"](case)
        Image.fromarray(image).save(directory / f"{palette}.png")
        (directory / f"{palette}-config.json").write_text(
            json.dumps(truth["config"], indent=2) + "\n"
        )
        (directory / f"{palette}-truth.json").write_text(json.dumps(truth, indent=2) + "\n")
        c = truth["config"]
        request = {
            "numeric_only": True,
            "hints": {
                "rows": c["rows"],
                "columns": c["columns"],
                "colourbar_start_value": c["colourbar_start_value"],
                "colourbar_end_value": c["colourbar_end_value"],
                "requested": ["heatmap", "colourbar"],
            },
        }
        (directory / f"{palette}-request.json").write_text(json.dumps(request, indent=2) + "\n")


if __name__ == "__main__":
    generate()
