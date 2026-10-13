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
