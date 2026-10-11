# Label extraction and reviewed reconstruction examples

Install local Tesseract/English data and run from the repository root:

```bash
uv sync --locked --extra ocr
uv run --extra ocr heatmap-decoder labels examples/labels/clean.png \
  --config examples/labels/config.json --out review/
```

Inspect `labels.tsv`, `labels_diagnostic.png`, `provenance.json`, and the editable
`labels_review.tsv` template. The config requires no colourbar. The command never
writes numeric matrices. For the full v0.2 numeric/review workflow:

```bash
uv run --extra ocr python examples/labels/workflow.py --out work/v0.2-example
```

`reconstruction.png` adds an independent continuous colourbar and 4 × 3 numerical
truth; `reconstruction_unreadable.png` blurs only label regions so numeric pixels
stay identical. `extract_config.json` requests OCR, `numeric_config.json` omits
labels, and `file_config.json` uses UTF-8 `genes.txt`/`samples.txt` without OCR.
`reviewed_example.tsv` supplies the known fixture labels for the correction demo.
For real images, verify and fill the review TSV manually. See [M6.md](../../M6.md).

`truth.json` preserves the displayed order. `clean.png` uses upright horizontal
DejaVu Sans 30 px; `blur.png` applies Gaussian radius 2; `compressed.jpg` uses
quality 35/subsampling 2; `unreadable.png` applies Gaussian radius 12. Tests use
these fixed images. To regenerate, install fonts-dejavu-core and run
`uv run python examples/labels/generate.py`.

The clean image recovered 7/7 exact labels locally with default settings.
Blur recovered 6/7 despite zero unresolved slots: accepted OCR can be wrong.
JPEG left 1/7 unresolved; the unreadable image left 7/7 unresolved. See the
[M5 report](../../M5.md) for environment versions, measurements and limitations.
