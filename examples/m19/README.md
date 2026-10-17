# M19: numeric completion and identifier work

`figure.png` is an independently rendered development fixture with three physical
signed ticks, a blocked matrix and supplied geometry/counts. Its `truth.json` is
evaluation evidence and is never an automation input.

```bash
uv sync --locked --extra ocr
uv run heatmap-decoder automate examples/m19/figure.png \
  --request examples/m19/request.json --out work/m19/diagnostic
uv run heatmap-decoder automate examples/m19/figure.png \
  --request examples/m19/request.json --out work/m19/numeric --experimental-export
uv run heatmap-decoder automate examples/m19/figure_partial.png \
  --request examples/m19/request_partial.json --out work/m19/partial --experimental-export
uv run heatmap-decoder automate examples/m18/figure.png \
  --request examples/m19/request_labels.json --out work/m19/composed --experimental-export
```

Tesseract and English data must be installed. Default diagnostics exit 5. A
qualified complete numeric-only export exits 0. The composed run exits 5 while
retaining the successful numeric matrix in
`numeric_positional/numeric_positional.csv`, with explicit `row_0`/`column_0` IDs.
Names remain in `label_evidence.json`; these positional IDs are not gene names.

An incomplete numeric result uses `numeric_partial.csv`, preserves NaN and QC,
and exits 5. No partial result counts as a complete export. Existing outputs
require `--overwrite`; failed runs do not overwrite a root labelled matrix.
