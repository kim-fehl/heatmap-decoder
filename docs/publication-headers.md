# Experimental publication header extraction

Extract regular raster heatmap grids, unnamed sample columns, line-based annotation tracks, colorbar calibration, and approximate cell values. Also detect colored header panels with text inside them, without a separate legend.

This implementation was exercised on all four images in [real-world examples](real-world-examples.md). It does not read reference annotations during extraction.

## Install and run

Python 3.11+ and the **Tesseract executable with English language data** are required. On Debian/Ubuntu, install `tesseract-ocr` and `tesseract-ocr-eng` first.

```sh
uv sync --extra publications
uv run --extra publications heatmap-decoder publication examples/real_world/publication_headers/plos-pone-0244065/figure-7c.png --out artifacts/figure-7c.json
uv run --extra publications heatmap-decoder publication examples/real_world/publication_headers/plos-ppat-1012368/figure-1c.png --out artifacts/figure-1c.json
uv run --extra publications heatmap-decoder publication examples/real_world/publication_headers/iscience-108897/figure-4a.jpg --expected-columns 1237 --out artifacts/figure-4a.json
uv run --extra publications heatmap-decoder publication examples/real_world/publication_headers/plos-pone-0155215/figure-4a.png --out artifacts/figure-4a-0155215.json
HEATMAP_DECODER_OCR_TESTS=1 uv run --extra publications pytest tests/test_publication_headers.py tests/test_publication_geometry.py -q
```

The Python API is `heatmap_decoder.publication.decode(image_path, expected_columns=None)` (the second parameter is keyword-only). JSON boxes use `[left, top, right, bottom]` in original image pixels. Column spans are zero-based and end-exclusive.

## Implemented behavior

- Infer regular cell spacing from image transitions, using support and coverage to avoid half-count and double-count harmonics.
- Keep unnamed sample labels and GEO IDs null, with stable positional IDs.
- Recover independent rows of group annotations. Infer shared span boundaries from the gaps between short rules and snap to the grid, rather than using label width or requiring full line coverage. Ambiguous gaps raise `DecodeError`.
- Calibrate colorbars from detected ticks where available; otherwise use centers of numeric label boxes. Gradient endpoints are not assigned outer label values automatically.
- Estimate displayed values by matching cell-interior colors to the colorbar, with RGB mismatch diagnostics and an explicit count of values outside the labeled interval.
- Read light or dark text within colored header panels and retain the panels' colors and matrix-block geometry. Recover stacked filled and outlined rectangle tracks over a regular grid. Separate white gaps between blocks from the data, and avoid treating rectangle borders as additional grouping rules.
- Return `resolution_limited` for the dense colored-panel case, with null sample columns and numeric values. External expected sample counts are recorded as metadata; they never silently replace an image-derived count.

## Verification and limits

The regression suite covers the four original images, their reference spans and labels, tick/label-center selection, short rules, unequal group sizes, ambiguous inputs, an external-count conflict, row-annotation/colorbar separation, and numeric recovery from a synthetic matrix with known values. Controlled OCR cases cover white text on black/red/gray/blue and black text on white/green. English OCR reads Greek beta as `B` in the miR-29b image; outputs retain the raw text and boxes.

There are 23 publication regression tests, including the installed application's CLI and overwrite protection. The four-image extraction summary is checked in as [evaluation.json](../examples/real_world/publication_headers/evaluation.json).

PLOS outputs recover grids of 46 × 31 and 100 × 18, respectively. Row OCR currently populates 46/46 and 90/100 label slots; populated labels can still contain OCR errors. The iScience image recovers 65 rows and three named panels, but cannot resolve individual samples at this raster resolution. Its expected total of 1,237 is an [article-derived inference](../examples/real_world/publication_headers/iscience-108897/figure-4a.annotations.json), not a pixel measurement.

The additional palmoplantar figure recovers a 75 × 50 grid and both rectangle tracks, including the two distinct `Normal` spans. The visible clipped label `PP` is preserved; the fixture stores the intended biological label separately.

This is an initial implementation for the documented layouts, not a universal plot parser. Numeric colorbar OCR currently supports right-side vertical labels and below-bar horizontal labels with ordinary decimal numbers and a linear displayed scale. Row gene OCR can miss or substitute characters. Rectangle OCR supports high-contrast light/dark text and flat fills; the separate-block dense-panel path conservatively leaves individual columns unresolved. Color-derived numbers have not been validated against GEO expression data and must not be presented as biological ground truth.

Run `uv run --extra publications python scripts/check_real_examples.py` to regenerate extraction JSON and a concise report under `artifacts/`.
