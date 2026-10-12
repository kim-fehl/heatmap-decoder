# M10 realistic-scale synthetic examples

`figure.png` is an original MIT **30-gene × 10-sample** diverging heatmap with
right-side upright genes, bottom samples and a vertical colourbar above. The
ordinary manual `config.json` uses exact user-supplied labels in UTF-8 files.
`truth.json` retains the known plotted matrix, label boxes and displayed order.

```bash
uv run heatmap-decoder extract examples/m10/figure.png \
  --config examples/m10/config.json --out work/m10-example
uv run heatmap-decoder detect examples/m10/figure.png \
  --config examples/m10/hints.json --out work/m10-roi
# Expected detect exit: 5. Review its overlay and decision evidence.
```

`generate.py` creates 32 combinations of gene side (left/right), sample side
(top/bottom), bar side (left/right/top/bottom) and physical orientation
(horizontal/vertical), with 20×6, 25×8 and 30×10 shapes. Each has clean, Gaussian
blur 0.65 px and JPEG quality 85 variants. The 32 layouts reuse three seeded
numeric matrices; these variants are dependent development data, not 32
independent experimental sources. Use Pillow's DejaVu Sans 14 if available;
otherwise its bundled default font. Only upright, horizontal text is supported.

```bash
uv run python examples/m10/generate.py --out work/m10-fixtures
uv run --extra ocr python benchmarks/m10/evaluate.py --ocr --out work/m10-report.json
```

Tests reconstruct values with the existing ambiguity/missing QC intact, test
mock-token assignment on every layout, export reviewed/file labels on the large
matrix, and run optional real Tesseract checks on 30 genes and 10 samples. Clean
sequential colours can still be ambiguous. OCR can lose underscores or misread
symbols while retaining high confidence. These remain review issues.
