# Changelog

## 0.2.0.dev5 — M5

- Add optional `pytesseract` extra and one fixed local Tesseract OCR pass with
  grayscale/upscaling, lazy engine checks, actionable installation errors and
  raw token/version evidence.
- Add label-only library/CLI, quoted label TSV, editable review template,
  diagnostic overlay and provenance. Support UTF-8 file sources without OCR;
  unresolved slots write review artifacts and return CLI exit 4.
- Add fixed clean/blur/JPEG/unreadable fixtures, mocked core tests and an opt-in
  OCR integration CI job. Report accuracy and unresolved proportions in `M5.md`.
- Keep numeric reconstruction unchanged; labelled CSVs and review import remain M6.

## 0.2.0.dev4 — M4

- Rename distribution/command to `heatmap-decoder` and implementation imports to
  `heatmap_decoder`; preserve deprecated, thin `heatmap2matrix` aliases.
- Validate optional axis label sources and OCR settings against the existing grid.
- Add crop transforms, shared fixed slots, deterministic fake-token grouping,
  conservative assignment and typed evidence/results, with engine-independent tests.
- Preserve numerical reconstruction; explicitly reject label-bearing numeric
  requests until integration. See `MIGRATION.md` and `M4.md`.

## 0.1.0

First release: Python API and Typer CLI for approximate numerical reconstruction
from PNG/JPEG heatmaps with a user-supplied regular grid and continuous linear
colourbar.

- Median interior sampling and CIELAB nearest-colour calibration.
- Colour-matching diagnostics, ambiguity detection, masks, alpha and no-data colours.
- Matrix/QC CSV, input/configuration provenance and diagnostic PNG export.
- Reproducible synthetic example, property tests and cross-platform CI.

Values approximate the displayed scale. Colour distances are diagnostics, not
numerical uncertainty intervals. OCR, automatic detection, nonlinear scales, PDF,
GUI and recovery of original biological measurements are outside this release.
