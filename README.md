# heatmap-decoder

Recover approximate tables from raster heatmaps when the original matrix is
unavailable. The decoder matches cell colours against the observed colourbar and
recovers values on the **displayed scale**, such as a z-score—not original
expression counts.

It supports explicitly configured regular or blocked PNG/JPEG heatmaps, continuous sequential or diverging
colourbars, reviewed linear or piecewise-linear scales, and gene/sample label OCR.
RapidOCR is the default label engine; sample labels support explicit quarter turns.
Automatic region and count detection is experimental and produces review evidence.

## Quick start

From a checkout, install with [uv](https://docs.astral.sh/uv/) and run the bundled
synthetic example:

```bash
uv sync --locked
uv run heatmap-decoder extract examples/figure.png \
  --config examples/config.json --out work/example --fail-on-qc
```

Outputs include `matrix.csv`, per-cell `quality.csv`, diagnostic images and
`provenance.json`. For your own figure, supply or review the grid, colourbar,
scale and label regions using the [usage guide](docs/usage.md).

## Review the result

Compression, repeated colours, saturation and unreadable labels can make recovery
ambiguous. Unsupported or unreliable values receive flags or NaN. Inspect the
quality report and every identifier; OCR confidence does not verify gene identity.
Numeric tick OCR uses Tesseract and requires its additional installation.

## Documentation

- [Usage and installation](docs/usage.md)
- [Automation and review](docs/automation.md)
- [Blocked grid configuration](docs/blocked-grids.md)
- [Scale calibration](docs/nonlinear-calibration.md)
- [Worked examples](examples/README.md)
- [Validation evidence and limitations](benchmarks/README.md)
- [Documentation index](docs/README.md)

MIT-licensed code. Input figures retain their owners' copyright.
