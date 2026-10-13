# Observed colour and reviewed numeric calibration

The dense observed colourbar supplies colours. A separately chosen numeric model
supplies a value at each bar position. A nonlinear colour trajectory can still
have a linear numeric scale; colour-match distance cannot verify that choice.

`scale.model` defaults to `linear`, preserving existing endpoints and linear tick
fitting. Use `piecewise_linear` only with explicitly supplied/reviewed monotonic
anchors (at least three). Positions use original-image x coordinates for horizontal
bars or y for vertical bars. Supply anchors in strictly increasing spatial order;
values must be strictly increasing or decreasing, finite and within the bar.

```json
{
  "source": "manual_ticks",
  "model": "piecewise_linear",
  "quantity": "displayed z-score",
  "anchors": [
    {"position": 20, "value": -3, "origin": "manual", "status": "reviewed"},
    {"position": 70, "value": -1, "origin": "manual", "status": "reviewed"},
    {"position": 190, "value": 4, "origin": "manual", "status": "reviewed"}
  ]
}
```

That configuration calibrates only positions 20..190. A match or near-match
candidate entering the uncovered span is NaN with `UNCALIBRATED`; its quality
record says `unsupported_calibration_span`. A complete experimental automation
export additionally requires the **entire** physical bar span to be calibrated.
Supply/review actual physical edge values to cover it. Intermediate colour samples
are not new numeric anchors. Linear calibration retains its historical
extrapolation behaviour for compatibility.

Every piecewise provenance record retains knots, segment slopes, positional/value
coverage, unsupported sample count, source/review status and the linear comparison
including failed residual checks. A failed linear comparison never selects a model
or drops an outlier. An ill-conditioned comparison remains explicit evidence and
does not invalidate a separately valid piecewise model. Zero knot residual does
not validate the unknown mapping between ticks.

For image-bound legend review, select `scale.model` in the extraction config or
`scale_model` in the legend-only config. The existing hashed review TSV must review
all detected indices (include or exclude). Reviewed positional values become
manual ticks using the selected model. Extra physical edge anchors can be supplied
through a separate manual-tick configuration; do not invent a detected edge tick.
Unreviewed OCR proposals always require review for piecewise calibration.

`gradient.json` and `colourbar_enlarged.png` show each supplied tick, intermediate
checkpoint and both physical edges. Sampling uses a median over the transverse
central half and **one axial pixel**, preserving narrow transitions and plateaus.
Each checkpoint records original position, rounded sampled position, pixel footprint,
RGB, Lab, transverse dispersion and missing-alpha evidence. Dispersion is evidence,
not a contamination classifier or statistical interval. Measured colours remain
mandatory; endpoint/tick colour interpolation never replaces them.

ROI diagnostics use blue/pink/lilac/violet frames, titled in a white side ledger
with manual/selected/proposed state. Proposed frames are dashed with stable IDs.
OCR frames use H teal (>=90), M yellow (>=configured minimum), L crimson (below
minimum), and ? neutral dashed for unknown confidence. Configure `high_confidence`
and `min_confidence` in OCR settings with high >= minimum. Raw token IDs and
candidate IDs link to JSON record indices; recognition status is printed separately.
Missing slots have cross markers. Manual text never acquires OCR confidence.
Source pixels keep original coordinates; enlarged crops are separate artifacts.
Black ledger text and status letters support dark/bright images and colour-vision
limitations independently of frame colour.

Optional `ocr.retry_variants: ["contrast", "threshold"]` applies only to explicit
slot label OCR. The default is no retries. Each slot retains the original grayscale
bicubic pass, autocontrast (cutoff 0) and fixed luminance threshold 160. All runs
use the same crop/rotation/upscale transform and share the existing 128-call,
16-million-pixel and 60-second per-axis budgets. Skipped/exhausted/failed attempts
are recorded. Matching same-slot exact readings retain minimum confidence and
all raw runs; conflicting readings stay separate review alternatives. The trials
showed fewer false heuristic acceptances but lower accepted coverage, so retries
remain optional. English traineddata is insufficient to verify Greek glyphs;
review the per-slot ledger and original enlarged crop, or explicitly configure
installed language data. Numeric-only reconstruction does not require OCR names.

## Padded OCR fields and regular grid slots

For explicit slot OCR, `crop_padding_pixels` and `white_padding_pixels` are available
on each `labels.rows` / `labels.columns` source, each limited to 0..2 (default 0).
The former adds original neighbouring pixels, clipped to the image; the latter
adds a recorded white border before resizing. The same inverse transform removes
that border before returning boxes to original-image coordinates, including ±90°
columns. Expanded pixels count against the existing shared OCR budgets.

Candidates from an explicit slot remain owned by that slot. They cannot be spread
across neighbours just because a floating grid edge is rounded outwards. Geometry
still checks centre offset and containment with only the recorded source padding
and one-pixel rounding tolerance; failed evidence stays unresolved in its own slot.
Multiple readings remain alternatives. Sparse region OCR retains its original
geometric association policy. Logical field outlines use the uniform grid pitch;
raw token/candidate outlines still show actual observed glyph bounds separately.

Legend options add `align_tick_fields`, `crop_padding_pixels`,
`white_padding_pixels`, and optional `retry_white_padding_pixels` (at most two
extra white-border variants). Alignment is opt-in. It measures one common text
band beyond independently measured strokes: shared top/bottom for a horizontal
bar, shared left/right for a vertical bar. Separate image-ink clusters determine
field widths/heights, keeping observed signs with digits. No expected tick count,
equal spacing, OCR confidence or reference values choose the fields. Competing
bands or clusters require review.

`legend_fields_enlarged.png` draws every attempted tick field, including those with
missing or unparseable text. These field ROI frames are distinct from confidence
frames around actual OCR observations. `numeric_alternatives` preserves every
padding reading and its run/transform link. Conflicting readings retain review
status; contradictory numeric values become unresolved. A recovered retry proposal
also requires review. Dash glyphs such as the observed em dash in `—1` are not
automatically converted to a mathematical minus.

[Measured follow-up](../benchmarks/m16/ocr-padding/README.md) demonstrates improved
raw OCR without relaxing identifier/scale export guards.
