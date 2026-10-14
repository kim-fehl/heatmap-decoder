# Worked examples

Run examples from the repository root after `uv sync --locked`. Output directories
should be new or explicitly overwritten with the CLI's overwrite option.

| Example | Purpose |
|---|---|
| [`figure.png` and `config.json`](config.json) | Basic numeric extraction with supplied geometry and a linear colourbar |
| [Labels and review](labels/) | Label evidence, supplied text files and reviewed reconstruction; historical Tesseract configurations are explicitly selected |
| [Automation](../docs/automation.md#experimental-numeric-export) | Review-oriented proposals and the scoped experimental numeric export |
| [Rotated labels](../docs/rotated-labels.md) | Explicit sample-label quarter turns and reviewed names |
| [Published heatmaps and source truth](real_world/README.md) | Aligned article source tables, image estimates and categorical group naming |
| [Publication crop](m16/README.md) | Observed gradients, reviewed multi-anchor calibration and default RapidOCR labels |

## Basic numeric extraction

```bash
uv run heatmap-decoder extract examples/figure.png \
  --config examples/config.json --out work/example --fail-on-qc
```

The synthetic reference is a 20×20 heatmap with a continuous diverging colourbar.
`truth.csv` is independently generated and headerless; `generate.py` recreates it
with seed 2026 and an analytic blue–white–red palette. This example does not
establish accuracy for published figures.

Other milestone directories retain reproducible development fixtures. Prefer the
[usage guide](../docs/usage.md) when adapting a configuration to your own image.

[M19](m19/README.md) demonstrates separate numeric, partial and identifier completion states.
