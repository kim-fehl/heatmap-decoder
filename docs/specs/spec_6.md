# M16 specification — Observed gradients, reviewed nonlinear calibration and readable diagnostics

**Status:** proposed implementation specification  
**Implementation record:** [M16 status](../progress/M16.md) contains current results and remaining limits; this document retains the original requirements.  
**Suggested repository path:** `docs/specs/spec_6.md`  
**Starting point:** M15 on branch `work`, inspected at commit `2ec504d29df73636880540e093521c6fb675035f`.

## 1. Objective

Improve extraction and review of low-resolution publication heatmaps by:

1. Preserving and inspecting the observed colourbar curve.
2. Supporting explicitly supplied or reviewed nonlinear position-to-value calibration.
3. Improving bounded OCR where the new real-world example demonstrates failures.
4. Producing readable diagnostics with distinct visual conventions for ROIs and OCR confidence.

Use `examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg` and its accompanying TSV as the principal development example.

This phase extends the existing modular pipeline. It must not create a separate decoder or imply stable-release maturity from one publication example.

## 2. Reviewed starting point

The decoder already samples every axial pixel of the colourbar using a median across its interior width. It therefore preserves observed nonlinear colour changes.

The current numerical calibration assumes a linear relationship between **bar position and displayed value**. These are separate concerns:

- A nonlinear colour trajectory can represent a linear numeric scale.
- A nonlinear numeric scale requires additional positional calibration.
- Colour-match distance cannot verify the numeric scale model.

M15 demonstrated this limitation: forcing linear endpoints onto its nonlinear challenge produced an unflagged wrong matrix, with validation MAE of 16.38% and maximum error of 25.33% of the displayed range.

The new example contains 33 gene rows and eight sample columns, a horizontal magenta–dark–cyan colourbar, right-side gene labels, bottom sample labels and a left dendrogram. Several identifiers contain Greek characters.

The colourbar extends beyond its outer labelled −2 and +2 ticks. Those tick values must not automatically become the physical edge values.

## 3. Scope and compatibility

M16 adds:

- Observed-gradient inspection and sampling evidence.
- Explicit monotonic piecewise-linear numeric calibration.
- Manual and reviewed multi-anchor workflows.
- Bounded low-resolution OCR improvements.
- Shared diagnostic styling across recognition, legend and label review.
- Synthetic validation and a documented real-world trial.

Existing endpoint configurations must retain their current numerical outputs. Linear calibration remains the default.

Default automation remains review-oriented. Unreviewed OCR scales, inferred dimensions and identifiers retain their existing export restrictions.

Automatic scale-model selection, log/spline fitting, arbitrary-angle OCR, learned super-resolution, identifier databases, GUI work and recovery of original TPM from z-scores are outside scope.

## 4. M16.1 — Real-world baseline and reference annotations

Create a reproducible manifest and audited annotations for the supplied example.

Record:

- Publication identifier and panel/crop identity.
- Image and TSV hashes, image dimensions and any preprocessing.
- Heatmap, colourbar core, gene-label and sample-label ROIs.
- Supplied dimensions, original row/column order and label transcription.
- Physical tick positions, numeric values, edge coverage and unresolved assumptions.
- Annotation history and a second-pass audit.

Document how the accompanying z-score table was reconstructed from TPM, including any log transformation, standardisation axis, sample population, variance convention and clipping.

Do not assume the eight displayed columns constitute the entire standardisation population. Until the relationship to the plotted values is established, describe numerical comparisons as **reconstruction agreement**.

Run and report these workflows separately:

1. Supplied geometry and dimensions with reviewed calibration.
2. Automatic ROI detection with supplied dimensions.
3. Legend localisation and numeric OCR.
4. Gene/sample OCR and slot assignment.
5. The composed requested workflow.

Retain failures and abstentions in the denominator. Evaluate the supplied crop without claiming full-publication automation.

The TSV is evaluation evidence. It must not influence production calibration, OCR spelling or ROI selection.

## 5. M16.2 — Observed-gradient inspection

Preserve dense colourbar sampling as the authoritative colour representation.

Add inspection checkpoints:

- At each labelled tick.
- At least once between each adjacent pair of labelled ticks.
- At both physical bar edges.

For each checkpoint, retain:

- Original axial position.
- Sampling footprint.
- RGB and Lab colour.
- Transverse colour dispersion.
- Relevant contamination or missing-data evidence.

Sampling must avoid transverse borders and decorations. Any new axial aggregation must record its footprint and demonstrate that it does not erase narrow transitions or plateaus.

Provide an enlarged colourbar diagnostic containing labelled ticks, checkpoint markers and sampled colour swatches.

Endpoint or tick-colour interpolation may be shown as a comparison, but must not replace measured colours.

Preserve existing repeated-colour, plateau, disconnected-candidate and poor-match evidence. Report candidate spans in calibrated value units as well as positional units. Do not relax existing ambiguity rejection merely because nonlinear calibration is available.

## 6. M16.3 — Reviewed multi-anchor numeric calibration

### Model selection

Extend scale configuration with an explicit model choice:

- `linear`: existing behaviour and default.
- `piecewise_linear`: interpolation between supplied/reviewed positional anchors.

Keep model choice separate from anchor source and review status.

A failed linear fit must remain visible as diagnostic evidence. It must not automatically select piecewise calibration or discard disagreeing anchors.

### Anchor requirements

Piecewise calibration requires at least three supplied or reviewed anchors.

Anchors use original-image axial coordinates: x for horizontal bars, y for vertical bars. They must have:

- Finite positions and values.
- Distinct, spatially ordered positions.
- Strictly increasing or strictly decreasing values.
- Positions within the selected colourbar.
- Explicit source and review provenance.

Reject duplicate positions, contradictory values, nonmonotonic scales and invalid numerical ranges.

### Interpolation and coverage

Evaluate piecewise-linear values at every densely sampled bar position.

Support both orientations and both numeric directions. Preserve the existing original-coordinate conventions.

Unlabelled intermediate colour samples obtain their values from the chosen calibration model. They are not independent numeric anchors.

Do not extrapolate outside the reviewed anchor span by default. Require separately supplied/reviewed edge values for full-span calibration. Otherwise, matches outside the calibrated span remain unresolved, with explicit reasons, and cannot qualify as a complete numeric export.

Piecewise interpolation is an approximation between anchors. Its zero anchor residual does not validate the unknown mapping between ticks.

### Review and provenance

Extend manual configuration and image-bound legend review to retain:

- Selected model.
- Raw and reviewed anchors.
- Covered positional/value range.
- Linear comparison residuals.
- Piecewise knots and segment slopes.
- Image, configuration and review hashes.
- Unsupported-span and rejection reasons.

Unreviewed OCR proposals remain review-required.

Explicit supplied/reviewed piecewise calibration may enter the existing experimental numeric-export path only after its positive, negative and compatibility checks pass. Other capability restrictions remain unchanged.

## 7. M16.4 — Bounded low-resolution OCR improvements

Measure baseline failures before changing preprocessing.

Prioritise the demonstrated risks: minus signs, Greek glyphs, underscores, tightly spaced gene labels and right-side label placement.

Where evidence warrants it, add a small deterministic retry set using crop padding, resizing and contrast/threshold variants.

Requirements:

- Preserve original pixels and every preprocessing transform.
- Retain raw readings, engine confidence, alternatives and engine metadata.
- Transform all boxes back into original-image coordinates.
- Share existing call, pixel and runtime budgets across retries.
- Record skipped or exhausted attempts explicitly.
- Keep spatial association independent of identifier spelling.
- Preserve conflicting readings as review alternatives.

Do not silently replace glyphs, guess aliases or use the reference TSV as a vocabulary. High OCR confidence cannot settle biological identity or conflicting signs.

Document language/data requirements. Unresolved Greek characters must produce actionable per-slot review evidence.

Numeric-only extraction must remain usable independently of identifier recognition.

## 8. M16.5 — Shared diagnostic visual convention

Use one drawing policy across recognition/ROI, legend and label diagnostics.

### ROI frames

| ROI | Frame family | Short title |
|---|---|---|
| Heatmap | Blue | `Heatmap` |
| Gene labels | Pink | `Gene labels` |
| Sample labels | Lilac | `Sample labels` |
| Colourbar | Violet | `Colourbar` |
| Tick search regions | Related pink/violet hues | `Tick search` |

Every displayed ROI must have a short title and an explicit proposed, selected or manual state. Show dimensions only when supplied or supported.

Alternative proposals use dashed borders and stable IDs.

Reserve teal, yellow and crimson annotation frames for OCR confidence. Preserve the source image’s colours.

### OCR frames

Frame each observed text token/candidate by engine confidence:

| Confidence | Frame |
|---|---|
| High | Teal |
| Medium | Yellow |
| Low | Crimson |
| Unknown | Neutral dashed |

Initial configurable bands are high ≥90, medium ≥ the configured minimum confidence, and low below that minimum. Validate threshold ordering.

Use the existing conservative confidence aggregation for grouped candidates and retain individual token confidence values.

Confidence and recognition status are distinct:

- Ambiguous association, parse failure and conflicts require explicit markers or text.
- Manual text must not acquire fabricated OCR confidence.
- Missing text receives a slot marker and `missing` status, not an invented text box.
- A high-confidence conflicting reading may retain its teal confidence frame while displaying an ambiguity marker.

### Readability

Place ROI titles outside content with contrasting backgrounds and collision-aware positioning. Use margins or callouts where titles cannot fit.

Provide:

- A visible confidence legend.
- Stable text IDs linked to review records.
- Enlarged colourbar and label crops.
- A ledger containing raw readings, confidence and review reasons.

Verify original-coordinate alignment, including rotated OCR.

Use status words, symbols or line styles alongside colour. Check contrast on dark/bright patches and under colour-vision simulations.

Refresh `docs/images/recognition.png` and the README’s currently embedded illustration, keeping documentation consistent with runtime diagnostics.

## 9. M16.6 — Validation and acceptance

Generate synthetic fixtures that vary colour trajectories and numeric mappings independently.

Cover:

- Nonlinear colours with linear numeric ticks.
- Monotonic nonlinear numeric scales.
- Asymmetric diverging gradients.
- Unequal tick spacing and reversed directions.
- Horizontal and vertical bars.
- Unlabelled edge extensions.
- JPEG compression, blur and low-resolution text.
- Missing signs and conflicting OCR readings.
- Repeated colours and saturated plateaus.

Keep derivatives with their base figure. Freeze a fresh validation partition and acceptance thresholds before final evaluation. The inspected publication example is development evidence, not held-out generalisation evidence.

Report:

- MAE and maximum error normalised by displayed range.
- Finite-cell coverage and clipping/bound status.
- Exact numeric tick recognition, including sign errors.
- Raw and accepted text accuracy.
- False text acceptance and slot/order accuracy.
- Per-capability coverage, abstention and runtime cost.
- Publication TSV agreement with its documented limitations.

M16 is complete when:

1. Legacy linear behaviour and export guards remain compatible.
2. Reviewed nonlinear calibration meets frozen numerical targets without hiding failures through reduced coverage.
3. Contradictory anchors, unsupported extrapolation and unresolved required evidence cannot produce misleading complete exports.
4. Every displayed ROI is titled and every OCR observation has a confidence frame.
5. Missing, ambiguous and manual states remain distinguishable at low resolution.
6. Focused tests, regression tests, lint, typing and applicable packaging/CLI checks pass.
7. Reviewed example commands, benchmark evidence, updated limitations and `docs/progress/M16.md` are delivered.

## 10. Implementation sequence

Implement **M16 only**, in this order:

1. Audited real-world annotations and baseline.
2. Observed-gradient evidence and shared diagnostic rendering.
3. Explicit reviewed piecewise calibration.
4. Evidence-driven bounded OCR improvements.
5. Frozen validation, examples and documentation.

Read repository instructions, earlier specifications and current correction reports before implementation. Preserve historical evidence and record actual outcomes, including failed trials.
