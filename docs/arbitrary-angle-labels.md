# Arbitrary-angle sample labels (M18)

Sample names may use a finite numeric `text_rotation_degrees`. The sign is visual
counter-clockwise in the source figure. Angles normalise modulo 360 into
`[-180,180)`: `370` becomes `10`, `180` becomes `-180`, and `90.0` is accepted.
Booleans, strings and non-finite values are rejected. Gene labels remain upright.
Only text crops rotate; matrix cells, blocked intervals and displayed order stay
in original-image coordinates.

```json
{
  "columns": {
    "source": "ocr",
    "bbox": [100, 20, 890, 358],
    "text_rotation_degrees": 30,
    "recognition": "lines",
    "attachment_endpoint": "nearest",
    "max_attachment_distance_pixels": 40
  }
}
```

This object belongs inside `labels`. `recognition: "lines"` selects complete-line
segmentation, including at 0/±90°. Other finite angles automatically use this path
regardless of the legacy recognition setting. Existing 0/±90° sparse/slot paths
retain exact transposes and their existing pixel/OCR behavior. Non-quarter turns
use a recorded affine inverse, an expanded white canvas and bicubic interpolation.
Continuous point/polygon mapping is separate from image interpolation: original
and normalized polygons, inverse matrices, padding and source crop are retained.
Axis-aligned boxes are display bounds, not the association evidence.

## Association and review

The complete selected region is normalized before narrow OCR crops. Dark glyph
components group into oriented lines, preserving word spaces. OCR receives each
complete isolated line, including names extending across neighbouring column bands.
Line polygons and baselines map back to the original image.

Above/below placement determines the relevant matrix boundary. `nearest` associates
the text endpoint closest to that boundary; horizontal centered names use their
centerline midpoint. `start`/`end` explicitly declare which reading endpoint is
attached. A configured endpoint facing away from the matrix requires review.
The endpoint's original x coordinate must fall inside one resolved cell interval
and within the existing axis-offset tolerance. Gaps never own names. The maximum
attachment distance is measured in source pixels, not the enlarged OCR frame.
No spelling, vocabulary or box centroid chooses a diagonal column.

Missing names keep holes, duplicates keep distinct indices, headers beyond the
attachment strip are retained as excluded evidence, and conflicting/overlapping
attachments require review. Truncated source regions and mixed or overlapping line
structure remain unresolved. Dense lines that cannot be separated are outside the
currently validated segmentation scope; use reviewed label files for those cases.

The new line path always marks recognized identifiers **unreviewed**, even at high
engine confidence. `labels.tsv`, raw runs and `labels_review.tsv` remain available.
Fill each requested review slot from the original figure before a labelled numeric
export. Exact repeats are merged only within the same line; conflicting preprocessing
readings remain separate. No name is generated to fill a requested missing slot.

## Bounded automatic proposals

Set `detector.propose_arbitrary_column_angles: true` in detection hints. The new
path widens column search windows so diagonal names can cross matrix boundaries.
It measures an unoriented axis with 12 deterministic coarse trials and at most
8 finer trials; these use glyph/line geometry, not OCR recognition scores. The
180° reading alternative remains explicit.

Glyph baseline alignment may distinguish the two directions when bottom edges
agree and top edges vary. Uniform-height/all-capital glyphs, weak/mixed baselines
and incomplete searches keep competing directions. A declared
`detector.column_attachment_endpoint: "start"` or `"end"` is another explicit
layout assumption; `"nearest"` is the default. An explicit source angle overrides
proposals. These geometric heuristics are not calibrated probabilities or general
font-orientation guarantees.

The selected line region and matrix counts must already be resolved for attachment
ownership. Angle proposals cannot independently supply biological dimensions.
Existing bounded quarter-turn proposals remain available through
`propose_column_rotations`; the new option takes precedence when both are enabled.

Segmentation, angle trials, preprocessing and recognition share pixel/deadline
budgets. Line extraction defaults to 128 calls, 16 million pixels and 60 seconds;
detection reuses its configured call/time budgets and the global pixel budget.
Skipped, exhausted and failed trials are retained. An incomplete search cannot
become unique just because another trial has a high OCR score. Numerical-only
extraction is unchanged by label settings; uncertain orientations do not change
sampling geometry or the scale model.

See the [runnable example](../examples/m18/README.md), [M18 report](progress/M18.md)
and [frozen validation](../benchmarks/m18/README.md). M17 degradation automation
and M19 scale/export qualification remain separate capabilities.
