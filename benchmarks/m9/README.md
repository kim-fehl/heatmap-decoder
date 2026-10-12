# Frozen M9 hold-out corpus

See [docs/progress/M9.md](../../docs/progress/M9.md) for results, gates, commands and supported limits.

- `manifest.json` / `manifest.sha256`: frozen base IDs, split, annotations, numeric
  truth, image sources/licences and hashes. All 36 figures/code are original MIT
  material. There are no third-party published figure assets in this corpus.
- `figures/`: immutable PNG/JPEG inputs; transformations stay with their base ID.
- `freeze.py`: independent PIL/NumPy generator, never imports the detector; requires
  DejaVu Sans only when creating a new corpus. It refuses an existing manifest.
- `evaluate.py`: verifies every hash and reports every attempt. Run with `--ocr` for
  conditional OCR; otherwise the omission is explicit. Ground-truth coordinate
  selection in comparisons is an evaluation oracle, not automatic hand-off.
- `report.json`: actual local results, thresholds, environment and failed automatic
  acceptance gates. It includes OCR false acceptance and all failure/negative cases.
- `review_hints.json`, `review_selections.json`, `workflow.py`: a fixed explicitly
  reviewed CLI example and a uniform-input example blocked before matrix output.

Do not tune detection on this corpus or overwrite its report to claim automation.
A future policy requires new held-out inputs and independent practical annotation.
Saved benchmark/detection files cannot enable runtime acceptance. v0.3 is review-only.
