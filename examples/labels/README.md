# Label-only OCR example

Install local Tesseract/English data and run from the repository root:

```bash
uv sync --locked --extra ocr
uv run --extra ocr heatmap-decoder labels examples/labels/clean.png \
  --config examples/labels/config.json --out review/
```

Inspect `labels.tsv`, `labels_diagnostic.png`, `provenance.json`, and the editable
`labels_review.tsv` template. The config requires no colourbar. The command never
writes numeric matrices. Review import is scheduled for M6.

`truth.json` preserves the displayed order. `clean.png` uses upright horizontal
DejaVu Sans 30 px; `blur.png` applies Gaussian radius 2; `compressed.jpg` uses
quality 35/subsampling 2; `unreadable.png` applies Gaussian radius 12. Tests use
these fixed images. To regenerate, install fonts-dejavu-core and run
`uv run python examples/labels/generate.py`.

The clean image recovered 7/7 exact labels locally with default settings.
Blur recovered 6/7 despite zero unresolved slots: accepted OCR can be wrong.
JPEG left 1/7 unresolved; the unreadable image left 7/7 unresolved. See the
[M5 report](../../M5.md) for environment versions, measurements and limitations.
