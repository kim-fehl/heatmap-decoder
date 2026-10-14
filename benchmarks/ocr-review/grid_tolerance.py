"""Trace production rejection separately from exact and tolerant geometry scoring."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.geometry_metrics import compare_grid_geometry  # noqa: E402
from benchmarks.m19.evaluate import measured  # noqa: E402
from benchmarks.m19.fixtures import fixture  # noqa: E402
from heatmap_decoder import AutomationRequest, DetectionHints, automate  # noqa: E402
from heatmap_decoder.blocks import BLOCK_POLICY  # noqa: E402
from heatmap_decoder.models import GridConfig  # noqa: E402

records = []
for family, seed in zip(
    ("none", "horizontal", "vertical", "both"), range(291001, 291005), strict=True
):
    original, expected, truth, _ = fixture(seed=seed, family=family)
    hints = DetectionHints(
        requested=("heatmap", "colourbar"),
        infer_rows=True,
        infer_columns=True,
        colourbar_start_value=expected.colourbar_start_value,
        colourbar_end_value=expected.colourbar_end_value,
    )
    for radius in (0, 0.3, 0.6, 1.0):
        image = np.asarray(Image.fromarray(original).filter(ImageFilter.GaussianBlur(radius)))
        runs = []
        for name, policy in (
            ("production", {}),
            ("width_3px_probe", {"maximum_block_transition_width": 3.0}),
            (
                "width_3px_halo_bypass_probe",
                {
                    "maximum_block_transition_width": 3.0,
                    "minimum_separator_edge_contrast_ratio": 0.0,
                },
            ),
        ):
            with patch.dict(BLOCK_POLICY, policy):
                result = automate(
                    image,
                    AutomationRequest(hints=hints, numeric_only=True),
                    experimental_export=True,
                )
                row = dict(
                    policy=name,
                    numeric_state=result.numeric_state,
                    numeric_accepted=result.numeric_complete.status == "accepted",
                    blockers=[
                        dict(requirement=d.requirement, reasons=d.reasons)
                        for d in result.decisions
                        if d.status != "accepted"
                    ],
                )
                if result.diagnostic_config:
                    row["geometry"] = compare_grid_geometry(result.diagnostic_config, expected)
                    row["numeric"] = measured(image, result.diagnostic_config, truth)
                proposals = []
                for grid in result.detection.block_proposals:
                    axes = {axis.axis: axis for axis in grid.axes}
                    if all(axes[a].segments for a in ("rows", "columns")):
                        config = GridConfig(
                            heatmap_bbox=grid.bbox,
                            rows_segments=axes["rows"].segments,
                            columns_segments=axes["columns"].segments,
                        )
                        metric = compare_grid_geometry(config, expected)
                    else:
                        metric = None
                    proposals.append(
                        dict(
                            bbox=grid.bbox,
                            status=grid.status,
                            reasons=grid.reasons,
                            metric=metric,
                            axes=[
                                dict(
                                    axis=a.axis,
                                    reasons=a.reasons,
                                    selected=a.evidence.get("selected"),
                                )
                                for a in grid.axes
                            ],
                        )
                    )
                row["block_proposals"] = proposals
                if name == "production":
                    row["heatmap_decision"] = result.decisions[0].model_dump(mode="json")
                runs.append(row)
                print(
                    family,
                    radius,
                    name,
                    row["numeric_accepted"],
                    row.get("geometry", {}).get("within_tolerance"),
                    flush=True,
                )
        # Sampling sensitivity at a deliberately shifted but correctly grouped grid.
        shifts = []
        for delta in (1.0, 1.5):
            x, y, r, b = expected.heatmap_bbox
            updates = dict(heatmap_bbox=(x + 1, y + 1, r + 1, b + 1))
            for axis in ("rows", "columns"):
                segments = getattr(expected, axis + "_segments")
                if segments:
                    updates[axis + "_segments"] = tuple(
                        s.model_copy(update={"start": s.start + delta, "end": s.end + delta})
                        for s in segments
                    )
            # Fractional shifts are internal segment edges; integer outer ROI stays valid.
            for axis in ("rows", "columns"):
                segments = updates.get(axis + "_segments")
                if segments:
                    segments = list(segments)
                    segments[0] = segments[0].model_copy(
                        update={"start": getattr(expected, axis + "_segments")[0].start + 1}
                    )
                    segments[-1] = segments[-1].model_copy(
                        update={"end": getattr(expected, axis + "_segments")[-1].end + 1}
                    )
                    updates[axis + "_segments"] = tuple(segments)
            perturbed = expected.model_copy(update=updates)
            shifts.append(
                dict(
                    delta=delta,
                    geometry=compare_grid_geometry(perturbed, expected),
                    numeric=measured(image, perturbed, truth),
                )
            )
        records.append(
            dict(family=family, seed=seed, radius=radius, runs=runs, sampling_shifts=shifts)
        )
(ROOT / "benchmarks/ocr-review/grid-tolerance.json").write_text(
    json.dumps(records, indent=2) + "\n"
)
