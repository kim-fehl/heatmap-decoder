"""Conditional M11 synthetic development evidence, never M12 release validation."""

import argparse
import json
import platform
import runpy
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from heatmap_decoder import ExtractionConfig, LegendConfig, extract, inspect_legend
from heatmap_decoder.legend import legend_policy_snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = runpy.run_path(str(ROOT / "examples/m11/generate.py"))


def cases():
    for orientation in ("vertical", "horizontal"):
        for count in (2, 3, 5, 7):
            for descending in (False, True):
                for side in ("left", "right") if orientation == "vertical" else ("above", "below"):
                    yield dict(
                        id=f"{orientation}-{count}-descending-{descending}-{side}",
                        orientation=orientation,
                        count=count,
                        descending=descending,
                        side=side,
                    )
    for name, options in (
        ("nonuniform", dict(nonuniform=True)),
        ("interior", dict(interior=True)),
        ("nonlinear", dict(nonlinear=True)),
        ("decimal_scientific", dict(count=3, strings=["2e0", "0.003643e0", "-2e0"])),
        ("nonmonotone_sign_labels", dict(strings=["2", "1.002", "0.003643", "1.002", "2"])),
        ("unparsed", dict(strings=["2", "1.002", "O", "-1.002", "-2"])),
    ):
        yield dict(id=name, **options)
    for degradation in ("blur", "jpeg"):
        yield dict(id=degradation, degradation=degradation)
    yield dict(id="misleading_title", title=True)
    yield dict(id="clipped", clipped=True)


def evaluate(case):
    name = case["id"]
    options = {k: v for k, v in case.items() if k not in {"id", "degradation", "title", "clipped"}}
    image, truth = FIXTURE["make_fixture"](**options)
    if case.get("degradation") == "blur":
        image = np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(0.65)))
    elif case.get("degradation") == "jpeg":
        from io import BytesIO

        stream = BytesIO()
        Image.fromarray(image).save(stream, format="JPEG", quality=85)
        stream.seek(0)
        image = np.asarray(Image.open(stream).convert("RGB"))
    if case.get("title"):
        figure = Image.fromarray(image)
        x, y, r, s = truth["colourbar_bbox"]
        ImageDraw.Draw(figure).text((r + 16, y - 38), "z-score", fill="black")
        image = np.asarray(figure)
    if case.get("clipped"):
        x, y, r, s = truth["colourbar_bbox"]
        image = image[:, x:].copy()
        truth["colourbar_bbox"] = (0, y, r - x, s)
    config = LegendConfig(
        colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation=truth["orientation"]
    )
    start = time.perf_counter()
    result = inspect_legend(image, config)  # Runtime decides before consulting truth.
    duration = time.perf_counter() - start
    expected_values = []
    for record in truth["tick_tokens"]:
        try:
            expected_values.append(float(record["text"]))
        except ValueError:
            expected_values.append(None)
    observed = []
    for tick in result.ticks:
        nearest = int(np.argmin(np.abs(np.array(truth["positions"]) - tick.position)))
        associated = (
            tick.position_source == "stroke"
            and abs(truth["positions"][nearest] - tick.position) <= 1
        )
        expected = expected_values[nearest] if associated else None
        observed.append(
            dict(
                index=tick.index,
                position=tick.position,
                raw_text=tick.raw_text,
                parsed_value=tick.parsed_value,
                status=tick.status,
                position_source=tick.position_source,
                associated_correctly=bool(associated),
                expected_value=expected,
                exact_value=bool(
                    associated and expected is not None and tick.parsed_value == expected
                ),
                sign_error=bool(
                    associated
                    and expected is not None
                    and tick.parsed_value is not None
                    and np.sign(tick.parsed_value) != np.sign(expected)
                ),
            )
        )
    complete_exact = len([t for t in result.ticks if t.status != "excluded"]) == len(
        truth["positions"]
    ) and all(t["exact_value"] for t in observed if t["status"] != "excluded")
    numeric = None
    # Diagnostic numerical comparison uses the recognised fit. Manual ROI/count
    # truth is supplied only in this explicitly conditional experiment; no export.
    if result.fit is not None and result.fit.within_residual_limits and not case.get("clipped"):
        cfg = ExtractionConfig(
            heatmap_bbox=truth["heatmap_bbox"],
            colourbar_bbox=truth["colourbar_bbox"],
            rows=truth["rows"],
            columns=truth["columns"],
            colourbar_orientation=truth["orientation"],
            colourbar_start_value=result.fit.intercept,
            colourbar_end_value=result.fit.intercept + result.fit.slope,
        )
        reconstructed = extract(image, cfg)
        expected = np.array(truth["expected_matrix"])
        finite = np.isfinite(reconstructed.values)
        errors = np.abs(reconstructed.values[finite] - expected[finite]) / 4
        numeric = {
            "finite_fraction": float(finite.mean()),
            "mae_fraction_of_true_range": float(errors.mean()) if errors.size else None,
            "maximum_error_fraction_of_true_range": float(errors.max()) if errors.size else None,
            "flagged_fraction": float(np.mean(reconstructed.flags != 0)),
        }
    return {
        "id": name,
        "condition": "manual bar bounds; manual heatmap/counts for numerical diagnostic",
        "expected_ticks": len(truth["positions"]),
        "result": result.model_dump(mode="json"),
        "seconds": duration,
        "observed": observed,
        "complete_exact_ticks": complete_exact,
        "numeric_diagnostic": numeric,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("work/m11-report.json"))
    args = parser.parse_args()
    attempts = []
    for case in cases():
        attempts.append(evaluate(case))
        print(case["id"], attempts[-1]["result"]["status"], flush=True)
    ticks = [t for a in attempts for t in a["observed"] if t["status"] != "excluded"]
    eligible = [a for a in attempts if a["result"]["development_eligible"]]
    report = {
        "purpose": "Synthetic conditional development evidence; no M12 hold-out or real figures",
        "python": platform.python_version(),
        "policy": legend_policy_snapshot(),
        "attempts": attempts,
        "summary": {
            "attempts": len(attempts),
            "ticks": len(ticks),
            "exact_tick_values": sum(t["exact_value"] for t in ticks),
            "sign_errors": sum(t["sign_error"] for t in ticks),
            "complete_exact_tick_cases": sum(a["complete_exact_ticks"] for a in attempts),
            "development_eligible": len(eligible),
            "wrong_development_eligible_tick_cases": sum(
                not a["complete_exact_ticks"] for a in eligible
            ),
            "released_automatic_scale_acceptances": 0,
            "statuses": dict(Counter(a["result"]["status"] for a in attempts)),
            "median_seconds": float(np.median([a["seconds"] for a in attempts])),
            "maximum_seconds": max(a["seconds"] for a in attempts),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
