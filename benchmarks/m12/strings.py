"""Development-only string comparison; deliberately has no test-split option."""

import json
import runpy
from pathlib import Path

from heatmap_decoder import LegendConfig, inspect_legend

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/m12/generate.py"))


def main(out):
    attempts = []
    for case in FIXTURE["cases"]("development"):
        image, truth = FIXTURE["make_fixture"](case)
        result = inspect_legend(
            image,
            LegendConfig(
                colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation=truth["orientation"]
            ),
        )
        correct = 0
        for expected in truth["ticks"]:
            matches = [
                t
                for t in result.ticks
                if t.position_source == "stroke" and abs(t.position - expected["position"]) <= 1
            ]
            correct += len(matches) == 1 and matches[0].parsed_value == expected["value"]
        attempts.append(
            dict(
                id=case["id"],
                exact=correct,
                total=len(truth["ticks"]),
                eligible=result.development_eligible,
                strings=[
                    dict(text=t.raw_text, status=t.status, reasons=t.reasons) for t in result.ticks
                ],
            )
        )
        print(case["id"], correct, len(truth["ticks"]), flush=True)
    report = dict(
        attempts=attempts,
        summary=dict(
            exact=sum(a["exact"] for a in attempts),
            total=sum(a["total"] for a in attempts),
            eligible=sum(a["eligible"] for a in attempts),
        ),
        scope="Conditional known-bar development OCR; not release evidence",
    )
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(report["summary"])


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
