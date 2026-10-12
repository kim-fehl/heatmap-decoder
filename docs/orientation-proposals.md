# Reviewing column-label orientations

M15 adds an opt-in proposal search to `heatmap-decoder detect` and `automate`.
It searches above/below a selected or shortlisted grid at exactly 0°, +90° and
−90°. It uses the M13 crop transform, upright OCR grouping and original column
assignment. The matrix is never rotated or transposed.

Enable it in detection hints:

```json
{
  "heatmap_bbox": [40, 300, 1320, 444],
  "rows": 6,
  "columns": 8,
  "requested": ["labels_columns"],
  "labels": {"ocr": {"upscale_factor": 2}},
  "detector": {
    "propose_column_rotations": true,
    "label_search_pixels": 240,
    "max_label_ocr_calls": 12,
    "max_label_ocr_seconds": 60
  }
}
```

The default remains the existing upright search. Explicit OCR sources retain their
configured angle and region; when used for count geometry, they run only that
trial. A reviewed label-region candidate carries its angle into the extraction
configuration. Manual text angles cannot be changed silently by ROI review.

Each orientation trial retains its ID, heatmap association, side, search box,
raw OCR-frame tokens, original-frame tokens, transform, unassigned candidates,
per-slot assignment (when counts are supplied), pixel cost, elapsed seconds,
engine metadata and reasons. Caps and failed trials remain visible. Per-trial
pixels obey the existing option; all shortlisted grids share the 16-million-pixel
budget plus the configured column-search call/time budgets. Individual OCR calls
receive the remaining time, up to the normal 30-second engine timeout. Runtime
setup/preprocessing overhead is recorded but is not a hard process deadline.

An adequate hypothesis must cover at least 75% of supplied columns (at least two
except for a one-column axis), avoid ambiguous assignments and yield text-shaped
boxes matching the trial direction. Unknown counts use regular unassigned text
centres with endpoint support, before slot fitting. No confidence ranking chooses
between competing layout hypotheses. A unique proposal requires a completed
search; capped alternatives cannot be assumed absent.

Opposite signs often both fit the same geometry: an upside-down OCR run can still
produce plausible strings. Both hypotheses remain review-required. A unique
layout proposal establishes neither spelling nor biological identity. Missing
labels retain their original slots; every requested name must be resolved before
a labelled matrix is written.

For count evidence, enable `count_label_geometry` and request `infer_columns`.
Only unique, adequately supported angle/side records contribute original-coordinate
unassigned centres; signal names link their orientation trial. Counting accepted
strings or slot-fitted labels is never an independent observation. All text
hypotheses remain one observation family. The existing independent-support and
extent/interior checks still apply; missing/contradictory text cannot establish a
new count by itself.

[The reviewed example](../examples/m15/README.md) demonstrates separate angle and
name review. Numeric-only workflows do not request this OCR search and remain
independent of label success. Experimental export still permits only numeric-only,
supplied counts and supplied/reviewed linear scale with accepted per-image checks.
Inferred dimensions, OCR-derived scales and identifiers remain outside that scope.
Arbitrary/diagonal angles, nonlinear-scale inference and axis-semantic guessing
remain unsupported.
