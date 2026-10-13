# Block-aware grids (M17)

Use `rows_segments` and/or `columns_segments` in extraction and label configuration:

```json
{
  "heatmap_bbox": [10, 20, 70, 100],
  "columns": 3,
  "rows_segments": [
    {"start": 20, "end": 40, "count": 2},
    {"start": 50, "end": 100, "count": 5}
  ]
}
```

These half-open bounds are original-image axis coordinates. This example has
seven rows, with a gap `[40,50)`, and three regular columns. Totals can be omitted
when segments specify them; supplied totals must agree. The first and last
segment must meet the selected heatmap ROI endpoints. Tighten that ROI to remove
outer whitespace. Segments must be finite, ordered, non-overlapping, in bounds,
and have strictly positive integer counts. Touching segments are allowed and
retain group metadata. Explicit geometry may use unequal pitches.

Both axes may be segmented. Gaps never become matrix cells. Global indices follow
displayed row/column order; provenance retains segment-local indices. The original
image is sampled inside each cell interval, without deleting/rescaling gap pixels.
Each cell must span at least one pixel and retain a pixel centre after the sampling
inset. Insufficient resolution is an error. Default central sampling excludes thin
edge strokes; reducing the inset can admit decoration.

`resolve_grid()` supplies intervals to numerical sampling, OCR ownership and
source overlays. Numerical and label provenance include `resolved_grid` and
`grid_sha256`. The hash includes version, intervals, segments, ownership and source
assumptions; label settings and numeric scale do not change it. `grid_diagnostic.png`
shows these resolved intervals, gaps and indices on the source. Detector overlays
show proposals separately, and never promote unresolved geometry into sampling.

Existing configurations without segments retain the same cell boundaries and
numerical results. Segment configuration is a user assumption, not evidence of
successful automatic detection or identifier verification.

## Automatic proposals

`detect` and `automate` generate bounded unions of aligned foreground blocks,
then test transverse separator profiles and common-pitch segment counts. Evidence
includes observed colour transitions/strokes, per-segment transition coverage,
boundary and endpoint residuals, interior constancy, rejected count models, competing
interpretations and a versioned policy. A supplied count constrains hypotheses; it
is never counted as an independent confirming observation.

A gap needs two measured edges, transverse constancy and compatible lattices in
surrounding blocks. Background colour alone cannot select it. Integer-pitch bands
retain the constant/missing-cell alternative. Palette-compatible strips retain the
annotation/data alternative. Large spaces retain independent-panel interpretations.
Broad block-boundary peaks require review rather than resolving coordinates inside a halo.
Multiple legends remain competing associations. A selected ROI must cover aligned
matrix blocks; it cannot silently export just the largest block.

The added no-gap count path is bounded to at least six cells per inferred axis,
at least 95% measured transition coverage and no unresolved stroke-count model.
The existing independently corroborated count path remains available. Block axes
permit constant neighbouring cells when sufficient lattice transitions remain.
These are model checks and heuristic thresholds, not calibrated probabilities.

Manual explicit geometry overrides proposals. Automatic inferred counts remain
outside experimental numeric export until M19 qualification. No OCR scale,
arbitrary-angle OCR or new identifier acceptance is added in M17. See the
[M17 report](progress/M17.md) for measured support and failed gates.
