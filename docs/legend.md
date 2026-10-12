# Numeric legends and scale review (M11)

M11 recognises upright numeric ticks on both sides of a selected/manual continuous
colourbar. It proposes a **linear** positional fit. Every OCR-derived scale still
requires explicit review; M12's independent release gate is not established.
Legend-only inspection needs no heatmap/counts. Label-only work needs no legend.

## Inspect before extraction

```bash
uv run --extra ocr heatmap-decoder legend examples/m11/figure.png \
  --config examples/m11/legend.json --out work/tick-review
# Expected exit 6. Open the overlay and review every proposed tick.
```

A bar-only configuration is:

```json
{
  "colourbar_bbox": [1056, 220, 1070, 770],
  "colourbar_orientation": "vertical"
}
```

A full extraction configuration also works with `legend`. Search strips are
bounded: default 128 px cross-axis for vertical bars, 64 px for horizontal bars,
64 px axial padding, at most 16 tick observations and four million upscaled OCR
pixels per inspection, at most 16 targeted numeric OCR calls, 256 localised text
tokens per strip and a 60-second job deadline (one in-flight call can overrun by
its existing 30-second timeout). `legend` options can override strip widths, language,
upscale/confidence, association offset and endpoint tolerance:

```json
{
  "colourbar_bbox": [1056, 220, 1070, 770],
  "colourbar_orientation": "vertical",
  "legend": {
    "search_pixels": 128,
    "horizontal_search_pixels": 64,
    "max_axis_offset_pixels": 6,
    "endpoint_tolerance_pixels": 2,
    "decimal_separator": ".",
    "ocr": {"language": "eng", "upscale_factor": 3, "min_confidence": 80}
  }
}
```

The OCR adapter uses sparse localisation (PSM 11), then one bounded single-line
pass (PSM 7) per unambiguous measured stroke. Localised text is associated
one-to-one; ambiguous collisions are retained, not resolved by confidence. Only
independently detected tick-stroke pixels are masked in the OCR copy so they do
not become minus signs. Observed glyphs are not corrected. Raw runs, mode, crop
boxes, stroke masks, versions and token membership stay in provenance.

Text centres are a weaker fallback and cannot qualify for development acceptance.
Clearly outside-span titles/captions are recorded as excluded evidence. Unparsed
or conflicting text within the plausible span blocks a complete scale. Text
touching a search edge and numeric-line positions that disagree with their stroke
are also flagged, even when OCR confidence is high. Neighbouring
legends, touching numeric labels or large fonts may need narrower regions or
manual anchors; no unbounded OCR search or automatic outlier removal occurs.

## Review artefacts and strict import

Inspection writes:

- `legend_ticks.tsv`: physical positions, stroke/text boxes, raw/normalised text,
  parsed values, confidence, membership and reasons.
- `legend_review.tsv`: editable `index`, `image_sha256`, `include`, `position`, `value`.
- `legend_diagnostic.png`: bar/search regions, measured strokes and recognised boxes.
- `legend_provenance.json`: raw evidence, fit residuals/span/conditioning and runtime policy.

Copy the template outside the next output directory. Inspect every row, correct
positions/values and explicitly set `include=no` for a title, duplicate legend or
other false proposal. Do not omit rows: every original index must appear exactly
once, with the same image SHA-256. Included anchors require finite values and
positions inside the bar's first/last axial sample, with at least two distinct
positions/values, strict monotonicity and a valid linear fit. Confidence never
chooses which observation to drop. Invalid/mismatched reviews stop before matrices.

```bash
uv run --extra ocr heatmap-decoder legend figure.png --config legend.json \
  --review reviewed_ticks.tsv --out work/confirmed-legend
uv run --extra ocr heatmap-decoder extract figure.png --config extraction.json \
  --legend-review reviewed_ticks.tsv --out results
```

Review reruns bounded OCR to preserve current image evidence. Missing OCR or
truncated resource-limited evidence cannot become an accepted scale through a
review; use explicit manual anchors instead. Existing outputs require
`--overwrite`; no input/review file can be overwritten even with that flag.
Raw ticks and the unreviewed fit/reasons remain separate from reviewed anchors.

## Three explicit scale sources

Legacy endpoint configurations reproduce the existing numeric outputs exactly:

```json
{
  "heatmap_bbox": [320, 220, 896, 770], "rows": 25, "columns": 8,
  "colourbar_bbox": [1056, 220, 1070, 770], "colourbar_orientation": "vertical",
  "colourbar_start_value": 2, "colourbar_end_value": -2,
  "scale": {"source": "manual_endpoints", "quantity": "row z-score"}
}
```

`scale` is optional for legacy inputs; unspecified quantities are explicitly
recorded as displayed values with unknown transformation. For a supplied or
reviewed multi-anchor scale, omit the endpoint fields:

```json
{
  "heatmap_bbox": [320, 220, 896, 770], "rows": 25, "columns": 8,
  "colourbar_bbox": [1056, 220, 1070, 770], "colourbar_orientation": "vertical",
  "scale": {
    "source": "manual_ticks", "quantity": "row z-score",
    "anchors": [
      {"position": 220, "value": 2},
      {"position": 494, "value": 0.003643},
      {"position": 769, "value": -2}
    ]
  }
}
```

Anchors are original-image **axial pixel coordinates** (y for vertical, x for
horizontal). Position 0 is the top/left sampled pixel; position 1 is the last
bottom/right pixel, i.e. `(position - start)/(exclusive_end - start - 1)`.
There is no axial inset. The sampling inset excludes only transverse bar edges.
Use actual positions, not tick indices or equal-spacing assumptions.

For recognition during extraction:

```json
{
  "heatmap_bbox": [320, 220, 896, 770], "rows": 25, "columns": 8,
  "colourbar_bbox": [1056, 220, 1070, 770], "colourbar_orientation": "vertical",
  "scale": {"source": "ocr_ticks", "quantity": "row z-score"}
}
```

`extract` first writes evidence and exits **6** without review, even if the fit
passes development checks. `--legend-review` resolves a concrete manual-tick
config for the ordinary numeric pipeline, while preserving the request, raw OCR,
image/review hashes and fit diagnostics. Endpoint fields with a tick source, OCR
anchors in a manual config, and contradictory source inputs fail validation.
User-entered values remain authoritative; OCR never modifies supplied endpoints.

## Numerical and scientific limits

A fit uses all spatially ordered anchors in `value = a × position + b`, with
ascending or descending direction. Default limits are 1% of the anchor value
range and two pixels of positional residual, conditioning <=1000, and endpoint
coverage within two pixels for development eligibility. Manual tick configs can
set `max_value_residual_fraction` and `max_position_residual_pixels` in `scale`.
No outlier is removed to meet these limits. Residuals remain available when a
monotonic nonlinear-labelled scale fails its linear check.

Two ticks determine a line but cannot test linearity. Three or more are eligible
only with consistent associated values, stroke evidence and endpoint coverage.
Interior ticks can fit a line; extrapolation requires explicit human/manual
confirmation. A shared monotonic OCR mistake can pass a fit, which is why exact
numeric recognition is benchmarked separately and the release gate remains closed.

The parser recognises signed integers/decimals, Unicode minus, `e` notation and
explicit `10^signed_integer`. It preserves raw text and normalization actions.
Separated superscripts/multipliers, missing signs, O/0 substitution and ambiguous
commas are not repaired. A decimal comma requires `decimal_separator=","`; there
is no thousands-separator guessing. Piecewise/log/nonlinear mapping is unsupported.

`matrix.csv` recovers approximate **displayed legend units**. Row z-scores cannot
recover raw counts. `quality.csv` adds `value_kind`: estimates, missing/ambiguous
cells and endpoint colours that may represent upper/lower bounds due to clipping.
The numerical endpoint is not a measured magnitude beyond the displayed range.
CSV decimal digits are serialization precision; ΔE, OCR confidence and fit
residuals are diagnostics, not confidence intervals or biological verification.

## Reproducible example and verification

The reference's reviewed TSV is bound to its exact PNG and explicitly supplied
25 genes/8 samples. It is not transferable to another figure:

```bash
uv run --extra ocr heatmap-decoder extract examples/m11/figure.png \
  --config examples/m11/labelled_config.json \
  --legend-review examples/m11/reviewed_ticks.tsv --out work/m11-reference
HEATMAP_DECODER_OCR_TESTS=1 uv run --extra ocr pytest -q
uv run --extra ocr python benchmarks/m11/evaluate.py --out work/m11-report.json
```

See [M11 results](../M11.md) for actual accuracy/failures and
[detailed manual/label/ROI usage](usage.md) for overrides and review formats.

## M12 association update

Measured tick midpoints now partition text grouping before horizontal words can
merge across adjacent ticks. Tokens spanning a band are retained intact. This
changes recognition geometry, not the numeric grammar or the requirement to
review signs and identities. [Selective automation](automation.md) composes this
scale decision with geometry, supplied counts and numerical quality.
