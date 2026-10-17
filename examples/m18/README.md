# M18 example

A synthetic +30° sample region sits above a 4 × 4 matrix with a vertical column gap.
Names extend beyond neighbouring column bands. The fixture, truth and reviewed
labels are generated development/reference evidence, not production OCR input.
RapidOCR is configured to demonstrate its actual underscore/spacing errors.

From the repository root:

```sh
uv run heatmap-decoder extract examples/m18/figure.png --config examples/m18/numeric_config.json --out work/m18/numeric --fail-on-qc
uv run heatmap-decoder labels examples/m18/figure.png --config examples/m18/labels.json --out work/m18/labels
uv run heatmap-decoder extract examples/m18/figure.png --config examples/m18/config.json --labels-review examples/m18/reviewed_labels.tsv --out work/m18/reviewed --fail-on-qc
uv run heatmap-decoder detect examples/m18/figure.png --config examples/m18/hints.json --out work/m18/angles
```

`labels` exits 4 because all four identifiers require review. Inspect the raw strings,
original polygons, normalized crops and attachment positions in `provenance.json`
and `labels_diagnostic.png`. The supplied synthetic review file restores the exact
reference strings and allows the labelled export. Numeric-only and reviewed labelled
matrices are identical. `detect` exits 5 while proposals await review; its geometry
decision and bounded angle/flip evidence are in `roi_candidates.json`.

For a real figure, fill the generated `labels_review.tsv` against your source;
do not reuse these reference names. See [configuration](../../docs/arbitrary-angle-labels.md).
