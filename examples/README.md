# Synthetic reference

`figure.png` contains a 20 × 20 heatmap and a vertical continuous diverging
colourbar. `config.json` gives its exclusive pixel ROIs and endpoint values;
`truth.csv` is the independently generated numerical matrix, without headers.

```bash
uv run heatmap2matrix extract examples/figure.png --config examples/config.json --out results/ --fail-on-qc
```

`generate.py` recreates all three files using seed 2026 and an analytic blue–white–red
palette. It does not import the extraction library. Run it from the repository:

```bash
uv run python examples/generate.py
```

Read both matrices with `pandas.read_csv(path, header=None)`. The numerical range
is 4, so the acceptance bound for MAE is 0.08. Tests additionally require no flags
on this clean reference. This synthetic example does not establish accuracy for
real published figures.
