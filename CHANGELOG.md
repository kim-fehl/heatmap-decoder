# Changelog

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
