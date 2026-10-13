# Numeric tick signs and extra right margin

Later [white-border/crop experiments](../white-padding/README.md) recover the exact
raw `EsGalectin1` on a manually tightened crop at confidence 27. The earlier
failures below remain the results of this configuration; a uniform-row change
regressed recognition and has not replaced these general settings.

This follow-up implements the user's explicit request to constrain tick labels to
numeric syntax and treat dash glyphs in numeric sign positions as minus signs.
Original readings remain available; every normalization records its Unicode code
point in `normalisation_actions`. For example, raw `—1` becomes canonical `-1`
with `observed_dash_to_minus:U+2014`.

Targeted numeric OCR requests digits, sign/decimal punctuation, supported dash
variants and `e/E/^` for scientific/power notation. Gene/sample OCR is unchanged
by that whitelist. The full numeric grammar still rejects letters/units, ranges,
standalone/doubled dashes, missing exponents and invalid decimal conventions. No
missing sign, digit or identifier is invented. Existing Unicode mathematical-minus
normalization remains compatible.

The whitelist alone erased a previously readable zero on this image. The explicit
`retry_unconstrained_numeric` option therefore enables one raw comparison only
when the bounded constrained attempts yield no parsed value. It shares the
existing call/pixel/deadline budgets. Its engine mode and raw reading are retained,
and it still passes the same strict numeric parser. A recovered value remains a
review proposal; conflicting readings do not become a complete automatic scale.

```bash
uv run --extra ocr heatmap-decoder legend \
  examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/config-numeric.json --out work/m16/numeric-signs
uv run --extra ocr heatmap-decoder labels \
  examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/labels-right-padding.json --out work/m16/right-labels
```

The five physical tick proposals now parse exactly as **-2, -1, 0, 1, 2**.
The trial used 13 OCR calls including localisation, compared with 12 previously.
All five fields remain drawn, with canonical and raw text shown together when
normalization occurs. Low confidence, alternatives and spurious localisation
candidates still require review; automatic OCR-scale exports remain disabled.

Rows add **four source pixels on the right**, independently of the existing
one-pixel source padding and one-pixel white margin. `crop_right_padding_pixels`
is bounded to 0..16 and applies only to explicit gene/row slot OCR. Source pixels
are clipped at the image edge, charged to existing budgets and mapped back to
original coordinates. Row heights and slot ownership do not change. The default
is zero extra right pixels.

The wider crop did **not** recover Galectin's trailing `1`: raw `EsGalectin` stays
at confidence 2, unresolved. Expansions of 2,4,6,8,12 pixels also failed, as did
3/4/5× resizing and the existing contrast/threshold variants in the focused probe.
These failures are retained rather than converting the reference transcription
`EsGalectin1` into production OCR output. The right-only change preserves **9/33**
exact raw genes, **8/8** exact samples and **29/41** unresolved requested slots;
one wrong gene reading still passes the heuristic confidence threshold.

Numerical heatmap extraction is bit-identical including NaNs: **251/264 finite
cells**. This remains inspected crop/development evidence, not generalisation.
`right-edge-probe.json` and `galectin-preprocessing-probe.json` retain attempted
variants; `legend.json`, `labels.json`, `summary.json` and `source.json` retain raw
engine/transform evidence, actual outcomes and runtime identity. The TSV never
enters production calibration, region selection or gene spelling.


Verification: **606 regression tests and 18 subtests pass** with real OCR; 134
focused tests also pass (one opt-in test skipped). Ruff, strict mypy with current
and NumPy 1.26.4, source/wheel builds and isolated CLI checks pass. Sign tests cover
supported Unicode variants, scientific exponents, ranges/doubled signs, identifiers
and raw preservation. Margin tests verify row-only bounds and original-coordinate
mapping; comparison tests verify recovered zero under the shared call budget.
