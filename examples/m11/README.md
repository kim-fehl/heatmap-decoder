# M11 expression-scale reference

This original MIT synthetic figure contains **25 genes × 8 samples**, upright
left-side genes, bottom samples and a five-tick vertical diverging legend. Known
numeric truth and raw rendered tick strings/positions are in `truth.json`. The
fixture computes per-row z-scores from seeded synthetic inputs, clips them to
[-2, 2], and retains unclipped scores separately for scientific/QC verification.

```bash
uv run --extra ocr heatmap-decoder legend examples/m11/figure.png \
  --config examples/m11/legend.json --out work/m11-legend
# Expected exit 6. Inspect the overlay and every numeric tick.
uv run --extra ocr heatmap-decoder extract examples/m11/figure.png \
  --config examples/m11/labelled_config.json \
  --legend-review examples/m11/reviewed_ticks.tsv --out work/m11-decoded
```

`config.json` requests OCR ticks without endpoint values. `labelled_config.json`
also provides explicit UTF-8 gene/sample files. The supplied review is a known
synthetic reference, bound to the exact PNG hash. It is not a general review for
other figures or OCR environments; mismatched image/indices must fail.

`hints.json` demonstrates automatic heatmap/label proposals with an explicitly
selected bar; these coordinates remain review-only. `recognition_evidence.json`
retains the actual detector/OCR evidence used for the README image. Its sample
recognition includes low-confidence text, and confidence does not verify strings.

```bash
uv run python examples/m11/generate.py
uv run --extra ocr python examples/m11/recognition.py
```

The generator supports both orientations/text sides, 2/3/5/7 ticks, ascending and
descending values, nonuniform positions, interior ticks and nonlinear challenges.
It uses DejaVu Sans 20 for ticks when available; Pillow's bundled default is a
fallback. Use `quantity: row z-score` to describe the recovered scale. Original
expression counts cannot be reconstructed from this image.
