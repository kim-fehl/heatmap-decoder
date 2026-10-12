# M11 numeric tick development evidence

`report.json` records every attempted figure and raw run from 42 synthetic
conditional experiments. The bar is manually annotated; downstream comparisons
also use supplied heatmap bounds and true counts. This measures M11 tick/scale
recognition, not fully automatic layout understanding or M12 release coverage.

Cases include both orientations/text sides, 2/3/5/7 ticks, both directions,
nonuniform spacing, interior ticks, nonlinear/nonmonotone labels, scientific
notation, unparsed text, titles, clipping and blur/JPEG. Sources reuse the same
25×8 matrix, so examples and derivatives are dependent. No published figures or
untouched validation/test split is claimed. M9/M10 reports remain historical.

Runtime decides first. Only then does evaluation compare exact parsed tick values,
physical association, signs, fit checks and numerical error against known truth.
A development-eligible fit is not an automatically accepted scale. All released
automatic-scale acceptances remain zero. Numerical comparisons use the recognised
fit without writing a matrix or treating observed residuals as uncertainty intervals.

```bash
uv run --extra ocr python benchmarks/m11/evaluate.py --out work/m11-report.json
```

OCR outcomes depend on engine/language/font versions. Raw metadata and every
failed/unresolved case are retained, including dense seven-tick horizontal labels.
See [M11](../../M11.md) for observed results and limitations.
