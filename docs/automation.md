# Selective automation

`automate` composes region, association, count, scale and cell-quality decisions.
Default operation writes review evidence. An explicit runtime flag enables scoped
experimental numeric export after all per-image checks pass. This makes partial success visible: a readable scale does not imply
that inferred counts or gene identities are trustworthy.

```bash
uv sync --locked
uv run heatmap-decoder automate examples/m12/figure.png \
  --request examples/m12/request.json --out work/automatic
```

Expected exit **5**: inspect `automation.json`, `roi_diagnostic.png`, and, when tick OCR is requested, `legend_ticks.tsv`, `legend_review.tsv`,
`legend_diagnostic.png` and `legend_provenance.json`. Unresolved bars are inspected diagnostically in separate candidate directories;
those runs cannot select the legend or enable export. Existing matrices are kept.
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
   `"infer_rows": true, "infer_columns": true`. Conditional acceptance requires
   agreement between independent colour transitions, grid strokes or raw label
   centres, a complete lattice, measured outer closure and consistent interiors.
   Enable `"detector": {"count_label_geometry": true}` to observe label centres
   with OCR independently of identifier assignment. Missing labels retain holes;
   harmonics, competing counts and insufficient support remain reviewable.
   Accepted counts appear in a new resolved config; the request stays immutable. Supplied counts remain authoritative if inference is
   also requested; they do not become evidence of correct automatic inference.

3. **Supplied counts**, tick-derived scale: omit endpoint values and add
   `"scale": {"source": "ocr_ticks", "quantity": "row z-score"}` at the root.
   The bundled M12 request uses this mode. Plausible bars may also receive
   diagnostic OCR before a target is accepted. Three or more parsed anchors must be associated with measured
   strokes and satisfy residual limits. Only positions within the outer tick span
   are calibrated; unsupported cells remain NaN. Every plausible tick
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
assessment; alone it does not permit export. OCR configurations retain their source and runtime anchors; they do not become
manual endpoints. The separate scale decision/legend evidence records the fit. To
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

The schema defaults to `sparse` for compatibility. Gene/sample OCR defaults to
RapidOCR with native-size grayscale input. `slots` recognizes each complete field
without invoking a text detector; `sparse` uses detected text regions. Set
`labels.ocr.engine` to `"tesseract"` for legacy PSM 7 slot OCR. Explicit enlargement
and padding are still supported, with a 128-call/16-million-upscaled-pixel/60-second
budget per axis. RapidOCR inference runs in a reusable worker stopped on timeout.
Each crop and raw
run, model hashes, score semantics and geometry kind are retained in label provenance;
tokens are mapped to original coordinates
before ordinary slot assignment. Full tokens crossing slots remain ambiguous.
The strategy does not establish the counts or invent missing identifiers.

Gene OCR may still confuse HSP70A/B/D, digit/letter pairs, hyphens and underscores.
High confidence does not establish a biological identifier. Review every gene
and sample string; no dictionary substitution, underscore restoration, O/0
conversion or fit-dependent OCR retries are performed.

See [current corrections and validation](corrections.md) and the separate
[historical M12 measurements](progress/M12.md). Synthetic testing does not establish
reliability on unseen publication figures.

## Experimental numeric export

```bash
uv run heatmap-decoder automate examples/corrections/figure.png \
  --request examples/corrections/request.json \
  --experimental-export --out work/experimental
```

Expected exit **0** with `matrix.csv`, `quality.csv`, provenance and recognition
reports. The flag is a runtime control; saved request JSON cannot enable it.
Complete root `matrix.csv` requires numeric-only operation. Numeric capabilities
require supplied or qualified sharp measured counts, a supplied/reviewed scale or
a qualified linear physical-tick OCR scale, evaluated geometry detector settings
and passing numeric decisions. Broad inferred-count/degradation export remains
closed when its registered coverage gate fails. Composed requests can retain
positional numeric results while identifier work remains unfinished; partial
numeric results remain explicitly partial. See [M19 policy](confident-automation.md).

Outputs record the experimental policy version, original request, resolved
configuration, image/report hashes and decisions. Endpoint matches can indicate
clipping; the matrix contains displayed-scale estimates, not original expression
counts. Stable release flags remain false. Any unfinished composed or numeric requirement exits **5**, preserving a root
labelled matrix while qualified retained numbers use a separate positional directory; successful replacement of
existing outputs requires that flag. Inspect diagnostics and use the reviewed
`extract` workflow for other requests.
