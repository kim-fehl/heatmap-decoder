# Heatmap2Matrix — M1 core

Reconstruct an **approximate** numerical matrix from an opaque PNG/JPEG heatmap
and its observed continuous, linear colourbar. Python 3.11+.

This implements milestone M1 only: configuration, raster loading, regular-grid
geometry, median sampling, CIELAB nearest-colour calibration and CSV export.
M2 QC/ambiguity/missing-data handling, provenance and diagnostics, and the M3 CLI,
release examples and CI are not implemented. This core is intended for clean,
unambiguous reference images; it does not yet determine reliability on real figures.

## Install and use

```bash
uv sync
```

```python
from heatmap2matrix import ExtractionConfig, extract, load_raster, write_csv

config = ExtractionConfig.model_validate_json(open("config.json").read())
result = extract(load_raster("figure.png"), config)
write_csv(result, "matrix.csv")
```

Configuration follows this contract:

```json
{
  "heatmap_bbox": [100, 80, 500, 480],
  "colourbar_bbox": [550, 80, 575, 480],
  "rows": 20,
  "columns": 20,
  "colourbar_orientation": "vertical",
  "colourbar_start_value": 2.0,
  "colourbar_end_value": -2.0,
  "sampling_inset_fraction": 0.25
}
```

Bounds are `[left, top, right, bottom]`, right/bottom exclusive, within the image.
Rows run top to bottom; columns left to right. Bar start means top for vertical
bars and left for horizontal bars, regardless of which endpoint is larger.
The bar ROI must exclude ticks, text and axial borders. Its first and last axial
pixel samples represent the stated endpoint values. All axial pixels are sampled;
the central half across the bar width is reduced by median. Cell sampling uses
pixel centres inside the inset interior, also reduced by median RGB. Empty
interiors fail clearly; lower the inset or use a finer raster.

Core `extract(image, config)` accepts a uint8 RGB NumPy array and performs no I/O.
The result contains `values` with shape `(rows, columns)`. CSV has no header or
index; read with `pandas.read_csv("matrix.csv", header=None)` when using pandas.
Pandas and Typer are deferred until their functionality is needed.

## Validation

```bash
uv run pytest -q -s
uv run ruff check .
uv run mypy src
```

Synthetic tests generate seeded 20 × 20 references independently of extraction,
using sequential and diverging palettes, both orientations, ascending/descending
endpoints and 4/9/16-pixel cells. They exercise PNG loading through CSV round-trip,
assert MAE <2% of the numerical range, and report MAE and maximum error. Flag
proportions are deferred to M2. Focused tests cover configuration, geometry,
median sampling, positional interpolation, perceptual matching and raster loading.

## Assumptions and limitations

Use a regular rectangular grid without perspective distortion or text in sampled
interiors. The colourbar must span the intended limits with a linear value scale.
Repeated/saturated colours may correspond to several values; M1 takes the first
nearest sample and does not yet flag ambiguity or poor matches. Transparent images
are rejected; in-memory RGB arrays have no missing-value mask. JPEG is readable,
but degraded-image accuracy is not established in M1.

These values are approximations of the figure's displayed scale, not exact
measurements or original pre-transformation biological values. No OCR, automated
ROI detection, PDF support, web UI or inference of RNA-seq counts is included.

MIT licensed. Input figures remain copyrighted by their respective owners.
