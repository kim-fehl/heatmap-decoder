# heatmap-decoder — selective automation and colourbar tick recognition

## Grounded review and objective

Reviewed `kim-fehl/heatmap-decoder`, branch `work`, commit `2131083da6db9364f0c83befb4f1c70b9a653e8c` (2026-10-12), including M8.md, M9.md, frozen reports, detection, calibration, OCR, extraction, CLI and policy code. Continue with M10–M12; no implementation is authorised by this planning document alone.

The current v0.3 implementation is genuinely review-only: policy flags are false, there is no automatic acceptance evaluator, and `--auto-roi` is absent. M9 succeeds as a transparent evaluation/release milestone, not an automation milestone.

Evidence from the saved M9 report:

- 36 synthetic figures, no published independently annotated real figures. ROI attempts use true supplied counts; count experiments use annotated heatmap boxes; OCR uses reviewed boxes and true counts. These are separate conditional experiments, not end-to-end automation.
- Heatmap top-three recall 24/27; usable bar recall 18/26. All 12 clean base figures and four downscaled variants have correct first-ranked heatmap and bar boxes. Mild blur/JPEG cases largely lose bar proposals. Multipanel examples still offer competing associations.
- 18 oracle-reviewed numerical comparisons match manual extraction with MAE 0. This is not a runtime selection guarantee.
- First-ranked count proposals are exact in 28/31 row and 24/31 column attempts, but harmonic alternatives remain. M8's merged-colour fixture ranks half the actual dimension first.
- OCR exact strings 24/56; 24 of 48 strings marked accepted are wrong. Lost underscores are a concrete example. Geometric slot association and high engine confidence do not validate identifiers.

Code findings that drive this phase:

- `detection_policy.py` is currently a snapshot of disabled flags, not an acceptance classifier. Add evaluated decisions; do not just flip flags.
- `count_proposals()` scores with the maximum signal alignment × coverage. This can conceal disagreement; most integer multiples survive alignment. Gridline and transition signals come from related image observations and cannot be treated as independent solely because their names differ.
- Label localisation requires a fixed/reviewed heatmap and known counts. Explicit internal staging is needed for end-to-end detection, rather than requiring the user to approve every intermediate box.
- `calibrate_values()` uses a two-endpoint `linspace`. OCR always uses one sparse-text mode. Tick recognition needs a targeted bounded mode and a position-aware scale model, while reusing the OCR adapter.

Local verification of the reviewed commit: `uv run --extra ocr pytest -q` passed 223 tests plus 18 subtests (6 opt-in OCR tests skipped); rerunning the six engine-sensitive tests with `HEATMAP_DECODER_OCR_TESTS=1` passed all six. Ruff and mypy passed. The frozen M9 evaluator was inspected and its saved report analysed; this review did not regenerate the full benchmark or claim independent real-figure validation.

Objective: automatically extract numbers for supported ordinary layouts, infer counts where evidence supports them, and recognise two or more numeric colourbar ticks with explicit validation. Preserve manual inputs and review paths. Treat numeric extraction and identifier acceptance as separate capabilities.

## Scope and priority

Support isolated axis-aligned single heatmaps, separate continuous bars on any side, sequential/diverging/grayscale palettes, visible cells with ordinary gridlines or no lines, modest JPEG compression/blur, and upright numeric tick text. No GUI, learned detector, PDF, rotated text, biological alias inference or categorical legend mapping.

Ship capability tiers independently:

1. Automatic ROI selection with supplied counts and supplied numerical scale.
2. Automatic per-axis counts when supported; manual count overrides otherwise.
3. Automatic tick-derived **linear** numerical scale once its separate gate passes.

All three need not succeed before tier 1 can be useful. If the user requests all three, every requested capability must resolve; never silently drop a requirement. Ordinary failures are development priorities, not occasions to broaden unsupported edge-case scope.

Identifier text stays review-required or explicitly user-supplied until independently validated. Existing label `accepted` means heuristic recognition only; add a separate verification/automation state without falsely marking old evidence verified. Numeric-only automation may use generated positional IDs when that mode was explicitly requested, exporting OCR suggestions separately. Requested gene/sample identities cannot silently become positional IDs. No underscore guessing or identifier dictionary correction.

## M10 — Acceptance decisions and ordinary-image robustness

### Runtime design

Add a small `acceptance.py` using typed evidence and decisions (`accepted`, `review_required`, `unresolved`, with reason codes). Retain candidate generation in `detection.py`, core numerical extraction in `extract.py`, and current threshold snapshots. Decisions are per ROI, association, count axis and scale, followed by a request-specific overall decision. A heuristic score is not a probability.

Only versioned runtime policy may enable a capability. Saved JSON, benchmark reports and user-edited confidence fields cannot enable it. Preserve manual overrides, image hashes, current overwrite protections, evidence provenance and old schema readability; bump schema for new evidence. Extraction receives a concrete validated config, not mutable detector state.

For supplied-count ROI acceptance require original-resolution grid/boundary consistency, usable cell interiors, full-grid bounds evidence, stable refined coordinates under a small preregistered perturbation set, and a unique plausible heatmap–bar association. Near-identical proposals should be deduplicated before evaluating competition. Reject plausible different panels, categorical strips, partial bars, clipping and unstable boxes. Stability is supporting evidence, not an independent proof or permission to average incompatible boxes.

Address ordinary bar failures with robust along-axis continuity and across-axis uniformity measurements that tolerate measured compression/blur noise. Diagnose examples before changing thresholds; normalise to noise/scale where justified. Keep real palette steps distinct from JPEG noise; do not bridge a categorical strip into a continuous bar. Validate border refinement and full-length bar completeness separately from palette similarity. Add performance/resource limits.

### Count decisions

Replace max-of-signals acceptance logic with explicit agreement, conflicts, fundamental/harmonic evidence, phase, coverage and outer completeness checks. Ranking can remain useful for review, but cannot be the acceptance rule.

Record supporting observation provenance. Exclude gridline pixels from the colour-transition channel when claiming separate evidence. Complete line locations and unassigned label positions can support exact dimensions; missing OCR strings must not shrink dimensions. Do not require two distinct OCR engines: repeated OCR/preprocessing is correlated support.

Accept inferred counts only when at least two distinct observations agree and plausible different dimensions are excluded under the supported layout model. Weak single-channel or hidden-cell evidence remains review-only. Some identical adjacent cells cannot be uniquely resolved: do not invent a mathematical proof of uniqueness. Supplied counts allow ROI automation to proceed even when inferred counts remain unresolved, provided inference was not a required capability.

Internally stage shortlisted ROI selection, raw text localisation, count validation and final label-region assignment with bounded candidate combinations. Do not feed assigned label count back as independent evidence. Retain losing plausible hypotheses and stop on disagreement. No unbounded optimisation loop.

### Acceptance and deliverables

Deliver decisions/diagnostics, improved detector fixtures, CLI opt-in for validated tiers and unit tests with adversarial evidence. During development expose decisions in review outputs; enable released automation only after M12's gate. Ensure count inference failures do not prevent testing supplied-count acceptance. Keep M9 frozen as historical regression data; it is now inspected development evidence, never a new hold-out set.

## M11 — Numeric colourbar ticks and multi-anchor calibration

### Localisation and recognition

Add focused `legend.py` for tick localisation/association and numeric parsing. Extend the current OCR adapter with a small documented mode choice for localisation versus single numeric line; keep the existing gene/sample path unchanged. Search narrow bounded strips adjacent to the selected/manual bar, on both possible text sides. Tick strokes and positions are image evidence; text centres are a weaker fallback and must be marked as such.

Associate text to tick positions along the bar with one-to-one geometry, offset limits and ambiguity reasons. Support horizontal/vertical bars but upright text only. Titles, units, nearby row labels, panel letters and a second legend must not become ticks. Retain raw tokens, crops/boxes, original-image tick positions, parsed values, engine confidence and associations.

Parse signed integers/decimals, Unicode minus and standard `e` scientific notation deterministically, preserving raw strings and normalisation actions. Support formatted powers such as `10^3` only through an explicit tested grammar; separated superscripts/multipliers remain review-only until geometry/grammar tests support them. No lost-minus correction, O-to-zero substitution or choosing a parse because it makes the scale look linear. Locale separators must be explicit; ambiguous commas are unresolved.

### Scale contract

Represent anchors as physical bar positions and numerical values, plus origin/status. Define normalised position 0 at top/left and 1 at bottom/right using the exact existing colourbar sample-position convention. Ensure tick-to-sample transforms agree with `sampling.py`, including any inset. Keep observed endpoints optional while scale recognition is unresolved. Resolve to a valid concrete extraction config only after review/acceptance.

Support any number of anchors >=2. Legacy manual start/end values remain equivalent to a two-anchor linear scale and must reproduce existing outputs. Configuration selects a scale source (`manual_endpoints`, `manual_ticks`, `ocr_ticks`); reject contradictory inputs. A separate diagnostic comparison may check supplied endpoints against OCR without changing them. Manual values remain authoritative.

For linear calibration fit `value = a * position + b` to spatially ordered ticks, allowing either direction. Use actual positions, not tick indices or assumed equal spacing. Require distinct positions/values, strict monotonicity and bounded position/value residuals; report residuals, recognised span and fit conditioning. Reject zero range, conflicting duplicates, reversals and grossly ill-conditioned fits.

- **Two ticks:** produce a reviewable fit. They determine a line but cannot test linearity; require manual confirmation for a newly OCR-derived scale in this phase.
- **Three or more ticks:** eligible for automatic linear-scale acceptance only when all reliable associated ticks agree with one line, no unparsed/conflicting plausible tick is silently discarded, and numeric OCR/position gates pass. Minimum three correctly associated anchors does not alone establish correct recognition.
- For automatic full-bar extraction require endpoint coverage within a preregistered pixel tolerance. Interior ticks can fit a line, but extrapolation past their recognised interval stays review-only. No pretending that the outermost ticks are bar endpoints.
- Do not automatically remove outliers to manufacture a passing scale; show suspect anchors and request review. A monotonic OCR mistake shared by all ticks can still pass a linear fit: report this risk and benchmark exact tick values separately from residuals.

If >=3 anchors contradict a linear scale, show the mismatch rather than defaulting to endpoint interpolation. Do not choose piecewise-linear interpolation merely because it passes through the labels. Unequal values at equally spaced ticks may indicate a nonlinear scale, misrecognition or incorrect associations.

Provide **explicit reviewed** `piecewise_linear` or `log10` mapping only if requested/configured and covered by their own tests; these are optional follow-ups, not prerequisites for M11. For log10 require positive values, fit in log space and retain units/transformation provenance; recover plotted legend values without claiming raw biological counts. No automatic nonlinear model selection this phase. Categorical or discontinuous legends remain unsupported.

### Review and verification

Write `legend_ticks.tsv`, editable review TSV, `legend_diagnostic.png` and scale diagnostics in provenance. Review validates indices/positions, bounds, non-empty finite values, duplicates and source-image identity. Label-only commands must not require a legend. Legend-only inspection should work with a manual bar without a heatmap.

Tests cover 2/3/5/7 ticks, negative/decimal/scientific values, ascending/descending scales, horizontal/vertical bars, nonuniform tick placement, interior ticks, minus-loss failures, unparsed tokens, misleading titles, ambiguous associations, clipped bars and nonlinear-labelled scales. Verify reconstructed values against known matrices using the inferred fit, not only against manual extraction. Keep colour-matching ambiguity and numeric-scale QC separate.

## M12 — End-to-end selective automation and practical validation

Build a new corpus split before policy tuning: development, validation and untouched test by base figure/source/layout. Derivative blur/JPEG/resizing variants stay with their base. Independently annotate region bounds, exact dimensions, tick positions/strings/values and scale type; label identities separately. Include licensed/redistributable published examples where possible; if publication images cannot be redistributed, document provenance and retain lawful evaluation-only assets. Do not call synthetic plots a real-figure benchmark.

Preregister supported ordinary layout families and quality envelope without using detector success to define eligibility. Include clean and ordinary raster degradation, no-grid/gridline cases, bar placements, 3–7 ticks and all relevant confounders. Keep dense/touching annotations, multiple panels, absent/categorical legends, missing ticks, hidden cell boundaries and incorrect supplied hints as challenge/negative cases. Both helpful abstention and unnecessary abstention matter.

Evaluate three separate modes plus the full requested pipeline:

1. ROI acceptance with supplied dimensions/scale.
2. ROI plus inferred dimensions, with supplied scale.
3. ROI plus tick-derived scale, with supplied dimensions.
4. Fully requested automatic geometry/counts/linear scale, with no ground-truth config injected.

The runtime chooses/abstains first; evaluation truth is used only afterwards. No oracle top-three selection, manually annotated boxes or exact counts may enter fully automatic attempts. Apply the same deployed classifier in the evaluator; do not hardcode false/true decisions in summary code.

Release engineering gates per enabled mode:

- At least 30 accepted **distinct base figures** across >=3 layout families for ROI automation; >=10 exact accepted examples per inferred axis across >=3 families retains the earlier count minimum. Aim for larger samples and report dependence among derivative images.
- On the preregistered ordinary supported test subset, target >=70% complete acceptance coverage. If missed, report the achieved rate and keep the capability experimental/review-only; do not redefine ordinary cases after seeing results. Counts and OCR scale can release independently of ROI acceptance.
- No observed wrong accepted complete layout, incorrect accepted dimension, wrong accepted tick value/scale direction or automatic acceptance on challenge cases with ambiguous target/unsupported scale. Report denominators and error uncertainty; finite zero errors are not a population guarantee. Optional positional numeric-only mode does not certify OCR identifiers.
- Accepted heatmap edges <=0.1 true cell pitch; bar edges within independently annotated tolerance; accepted inferred counts exact. Numeric MAE <2% of true supported scale range, plus maximum error, finite coverage and per-cell QC. For manual/noisy inputs, retain existing behaviour; automatic poor-match/missing/ambiguous fractions must satisfy preregistered quality limits.
- Tick exact-value accuracy, sign errors, association correctness, scale residuals, range coverage and resulting numerical errors are reported separately. Automatic tick-derived fits use >=3 validated anchors and recognised endpoint coverage; two-tick examples require review.

Minimum supported runtime decisions should be composable, e.g. supplied counts + accepted ROIs + reviewed scale. A failed required capability writes diagnostic/review artefacts and a distinct exit status before new matrix outputs. Document partial-capability success rather than presenting everything as failure. Version all policies; imported old evidence remains review-only unless re-evaluated from the actual image by the current policy.

Document complete CLI examples using the implemented schema, manual overrides and text-review requirements. Run core/OCR tests, lint/type checks, existing workflows and installed-wheel smoke checks. Report performance and OCR environment; no claim that high confidence verifies biology or recovers exact original measurements.

## Implementation order for Codex

Read repository instructions and M8/M9 evidence, then implement **M10 only**. First add measurable acceptance decisions for supplied-count numeric workflows and fix ordinary bar localisation failures. Count acceptance may remain off if independent support cannot be established. Implement M11 and M12 only when instructed. Keep old benchmark/report immutable; report actual coverage and failures. Never unlock automation by editing policy booleans alone.
