# Changelog

## v0.3.0 — Reviewed ROI and count proposals

- Freeze a 36-figure M9 benchmark and publish every attempt, including failures and negatives.
- Keep automatic ROI and per-axis count acceptance disabled because the required gates are not established.
- Persist versioned heuristic thresholds and the enforced review-only policy in detection/review provenance.
- Document OCR false acceptance separately from localisation and numeric reconstruction error.
- Add a reviewed v0.3 CLI example, frozen-input integrity checks and installed-wheel validation.

### M8

- Add continuous horizontal/vertical colourbar proposals and explicit heatmap associations.
- Add optional bounded upright OCR label-region searches with raw evidence and unresolved outcomes.
- Add explicit per-axis count proposals with original-resolution signals, conflicts and harmonics.
- Extend saved review with region/count ID selections and preserve M7 schema compatibility.
- Add M8 fixtures, quality report, reviewed CLI example and OCR CI workflow.
- Keep every inferred region/count review-only; numerical endpoints remain manual.

### M7

- Add review-only heatmap candidates, exact resize transforms and original-pixel refinement.
- Export ROI JSON/overlay and incomplete proposed extraction configs with manual overrides.
- Validate saved candidate review and preserve evidence/hashes in extraction provenance.
- Add synthetic regression fixtures and a complete proposal-quality report, including failures.
- Keep colourbar/label localisation, count inference and automatic hand-off out of M7.

## 0.2.0 — M6

- Integrate the shared label pipeline with reconstruction and block matrix/quality
  publication on unresolved slots while exporting review evidence (exit 4).
- Import strict axis/index/text review TSVs, permit correction of accepted strings,
  preserve raw OCR evidence and record review/source hashes and final assignments.
- Export matching gene/sample strings in matrix and long-form quality CSVs, with
  duplicate strings preserved and warned. Numeric-only formats/values remain intact.
- Package the complete clean/unreadable/reviewed example and file-only alternative;
  document engine installation, manual ROI selection, API, outputs and scientific limits.
- Extend mocked/core and real OCR tests, the OCR CI workflow and installed wheel checks.

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
