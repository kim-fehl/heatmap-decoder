# M13 reviewed rotated-label example

See [configuration and commands](../../docs/rotated-labels.md).
Regenerate with: uv run python examples/m13/generate.py. Generation uses Pillow,
NumPy and DejaVu fonts, without OCR or Matplotlib. The figure is synthetic.

The +90° column names include underscores and hyphens. OCR failures remain in
the [report](../../benchmarks/m13/report.json). The supplied reviewed_labels.tsv
records exact displayed synthetic names for the manual review path. End-to-end
tests check correct reviewed columns and identical numeric-only decoded values.

[Reviewed CSV](reviewed_matrix.csv) and [numeric evidence](reviewed_evidence.json)
retain the worked result and its input/review/output hashes.
