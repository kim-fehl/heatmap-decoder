# Selective automation (M12)

`automate` runs the same request-specific classifier used by the M12 benchmark.
It composes region, association, count, scale and cell-quality decisions before
checking the release gate. It writes evidence, never a new matrix while the gate
is closed. This makes partial success visible: a readable scale does not imply
that inferred counts or gene identities are trustworthy.

```bash
uv sync --locked --extra ocr
uv run heatmap-decoder automate examples/m12/figure.png \
  --request examples/m12/request.json --out work/automatic
```

Expected exit **5**: inspect `automation.json`, `roi_diagnostic.png`, and, when an
accepted bar permits OCR, `legend_ticks.tsv`, `legend_review.tsv`,
`legend_diagnostic.png` and `legend_provenance.json`. Existing matrices are kept.
Invalid requests/geometry and existing diagnostic outputs exit **2**; use
`--overwrite` only to replace diagnostics. No OCR installation is needed when
endpoints or reviewed anchors are supplied and identifier OCR is not requested.

## Four request modes

Requests wrap the existing `DetectionHints` under `hints`. `numeric_only` is
mandatory: it explicitly opts out of biological identifier verification. It
conflicts with any requested/file/OCR label source; identifiers are never silently
dropped. If identifiers are requested with `numeric_only: false`, their review
requirements participate in the complete decision.

1. **Supplied counts and scale**, automatic regions:

   ```json
   {"numeric_only": true, "hints": {
     "rows": 20, "columns": 6, "requested": ["heatmap", "colourbar"],
     "colourbar_start_value": -4, "colourbar_end_value": 2
   }}
   ```

2. **Inferred counts**, supplied scale: replace the two counts with
   `"infer_rows": true, "infer_columns": true`. The current count classifier
   abstains because uniqueness and outer-grid completeness are unestablished.
   It preserves count hypotheses, independent observation types, conflicts and
   harmonic alternatives. Supplied counts remain authoritative if inference is
   also requested; they do not become evidence of correct automatic inference.

3. **Supplied counts**, tick-derived scale: omit endpoint values and add
   `"scale": {"source": "ocr_ticks", "quantity": "row z-score"}` at the root.
   The bundled request uses this mode. OCR runs only after the runtime selects an
   accepted bar. Three or more parsed anchors must be associated with measured
   strokes, include endpoints, and satisfy residual limits. Every plausible tick
   participates; an outlier is never removed to make the line fit.

4. **Fully requested automatic geometry/counts/scale**: combine modes 2 and 3.
   No truth boxes, dimensions or scale endpoints are supplied. Failed counts keep
   the complete request under review even when the scale succeeds.

The scale directions refer to physical top-to-bottom or left-to-right order.
Both signs and asymmetric ranges are valid; zero need not sit at the palette's
midpoint. Tick values are not inferred from palette colours.

## Compose manual or reviewed capabilities

Manual heatmap and bar boxes are allowed through `hints`; remove that kind from
`requested`. Counts may be supplied independently. A reviewed linear scale can
be supplied without OCR:

```json
{
  "numeric_only": true,
  "hints": {
    "rows": 25, "columns": 8,
    "requested": ["heatmap", "colourbar"]
  },
  "scale": {
    "source": "manual_ticks", "quantity": "row z-score",
    "anchors": [
      {"position": 220, "value": 2, "origin": "reviewed"},
      {"position": 769, "value": -2, "origin": "reviewed"}
    ]
  }
}
```

These example positions belong to the vertical M11 figure, not the horizontal
M12 example. Check them against your own bar. Reviewed/manual ticks can have two
anchors because a person supplies the scale interpretation; extrapolation is
reported. Suggested OCR anchors are rejected as manual evidence.

`diagnostic_config` in the report is a concrete configuration for numerical
assessment, not permission to publish. Its endpoints may come from an unreviewed
OCR fit; the separate scale decision/legend evidence records their origin. To
export, use the existing [ROI review](usage.md) and [scale review](legend.md)
workflow with `extract`. Reviewed outputs retain hashes and raw suggestions.

## Better strings on a known grid

For a supplied/reviewed region and dimensions, select one fixed upright line
recognition pass per slot instead of sparse OCR of the whole strip:

```json
"labels": {
  "rows": {"source": "ocr", "bbox": [233, 222, 305, 768], "recognition": "slots"},
  "columns": {"source": "ocr", "bbox": [335, 789, 882, 810], "recognition": "slots"}
}
```

These regions belong to the bundled M11 figure and require its 25×8 geometry.

The schema defaults to `sparse` for compatibility. `slots` uses PSM 7, grayscale
and bicubic upscale, with a 128-call/16-million-upscaled-pixel/60-second budget per
axis (one in-flight engine call can finish after the deadline). Each crop and raw
run is retained in label provenance; tokens are mapped to original coordinates
before ordinary slot assignment. Full tokens crossing slots remain ambiguous.
The strategy does not establish the counts or invent missing identifiers.

Gene OCR may still confuse HSP70A/B/D, digit/letter pairs, hyphens and underscores.
High confidence does not establish a biological identifier. Review every gene
and sample string; no dictionary substitution, underscore restoration, O/0
conversion or fit-dependent OCR retries are performed.

See [M12 measurements and release decisions](../M12.md), including the fixed
ordinary-image denominator and the synthetic-only scope.
