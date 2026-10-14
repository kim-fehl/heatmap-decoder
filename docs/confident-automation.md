# Capability-specific experimental automation

M19 separates numeric completion, label geometry and identifier review. Inspect
`numeric_complete`, `label_geometry_complete`, `identifiers_complete`,
`numeric_state` and `identifier_state` in `automation.json`. Numeric success does
not certify OCR spelling. Decisions retain their observations, alternative
references, assumptions, support scope and policy version. Heuristic scores are
not probabilities. Contradictions and incomplete evidence are recorded separately.

The default command remains diagnostic. `--experimental-export` permits only
qualified capabilities with passing per-image evidence. Corpus qualification is
reported independently; its coverage does not prove an individual crop correct.
Supplied counts remain overrides. Measured inferred counts can pass a per-image
assessment while the broad export gate remains closed when corpus coverage fails. The passing
sharp-boundary scope has its own qualification and per-image tests.

## Linear OCR calibration

A scale needs at least three distinct physical ticks, reliable position/value
associations, strict monotonic order and an all-anchor linear fit. Registered
limits are 1% of the numeric tick range and two position pixels; tighter requested
limits remain effective. Every plausible tick participates. Conflicting readings,
skipped required OCR attempts, unreliable text-centre positions and inconsistent
fits remain blockers. There is no outlier removal, inferred edge value or
automatic nonlinear fallback.

A bounded retry can recover a glyph erased by a numeric whitelist. The retained
alternatives must contain one consistent reading, no skipped attempts, valid
position evidence and sufficient confidence. A conflicting nonempty reading still
requires review. The all-anchor fit provides a separate calibration check.

The bar may extend beyond its outer ticks. Only the observed tick span is
calibrated. Colour matching continues to use the complete observed trajectory;
every near-colour candidate must lie within the calibrated span. Unsupported
cells receive `UNCALIBRATED` and NaN. This protection applies before completion,
including cells whose best match alone would be supported. Runtime OCR anchors
retain `origin: ocr` and `status: suggested`; they are not disguised as manually
reviewed anchors or fitted full-bar endpoints. A saved OCR diagnostic configuration
alone does not bypass the review path in `extract`.

## Retained numeric results

A composed request with successful numeric geometry and scale writes positional
numeric output separately when experimental export is eligible. It leaves the
labelled request unfinished and returns exit 5. `label_evidence.json` retains raw
readings and geometry independently. No confidence score authorizes biological
identifiers. Failed label recognition does not discard successful numbers.

Complete numeric-only runs retain the existing headerless `matrix.csv` format;
provenance explicitly identifies positions. Composed runs use
`numeric_positional/numeric_positional.csv`, with zero-based positional headers.
Partial results use `numeric_positional/numeric_partial.csv`, preserve NaN and
flags, and never satisfy the complete decision. Each directory has numeric QC,
source-grid diagnostics and provenance. Default diagnostic runs write no numeric
matrix. File-based identifiers still need the existing reviewed extraction flow.

See the [example commands](../examples/m19/README.md) and
[measured status and remaining gates](progress/M19.md). The stable automatic
release remains disabled; publication generalisation is not established.
