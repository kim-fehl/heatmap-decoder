# Archived M16 implementation and follow-up log

This is the historical narrative through commit `2d93ddb`. References to
“current” inside this snapshot describe that stage, not the latest implementation.
The authoritative status and validation results are in [M16.md](../M16.md).

# M16 implementation and first real-world trial

RapidOCR is now the production **default gene/sample OCR engine**, included in the
main package, with native-size label defaults and bounded local-worker execution.
The [normal CLI trial](../../../benchmarks/m16/rapidocr-default/README.md) recovers
`EsGalectin1`, 26/33 exact genes and 8/8 samples. Existing heuristic confidence
acceptance remains subject to identity review; six incorrect names pass it.
Numeric tick OCR retains its Tesseract constraints.
Verification: 627 tests and 18 subtests pass with real OCR; typing, lint,
source/wheel builds and installed-wheel default label extraction also pass.

The preceding [independent OCR comparison](../../../benchmarks/m16/rapidocr/README.md):
RapidOCR recovers `EsGalectin1` from the existing equal-row field. Native-size
fields yield 26/33 exact genes and 8/8 samples, versus Tesseract's 9/33 and 8/8.
The optional standalone candidate generator retains raw evidence and marks every
candidate for review; seven gene errors remain. Earlier
[white-border experiments](../../../benchmarks/m16/white-padding/README.md) are also
retained, including manually tightened Tesseract recovery and failed variants.

**Status:** implementation and first crop trial delivered. Full publication
provenance acceptance remains open: the supplied TSV's TPM-to-z-score recipe and
physical edge values are unestablished. This is experimental development evidence,
not a stable release or unseen-publication generalisation result.

Starting point: `work` at `2ec504d29df73636880540e093521c6fb675035f`.
Implementation is on the local `m16` branch. The attached M16 specification was
treated as the requested implementation target; historical scope exclusions were
read as context, not as requests to abandon the user's M16 instruction.

## Delivered behaviour

- Explicit `linear` / `piecewise_linear` model selection, separate from source and
  review status. Linear endpoints retain their previous outputs. Monotonic
  piecewise models require at least three ordered finite positional anchors.
- No piecewise extrapolation: unsupported matches/candidate components are NaN
  with `UNCALIBRATED` and explicit quality reasons. Partial calibration cannot
  qualify as a complete experimental numeric automation export. Reviewed physical
  edge anchors enable full-span calibration; OCR/inferred counts/identifiers keep
  their previous export restrictions.
- Failed linear comparison remains visible alongside piecewise knots, segment
  slopes, positional/value coverage, configuration/image/review hashes and limits.
  Zero knot residual does not establish the mapping between ticks.
- Dense observed colourbar sampling is unchanged. Tick, midpoint and physical-edge
  checkpoints retain single-pixel axial footprints, RGB/Lab, transverse dispersion
  and missing-alpha evidence, with an enlarged bar and colour swatches.
- Shared titled ROI/state frames and separate OCR/confidence frames; original
  coordinates, stable observation IDs, slot ledgers, missing crosses and enlarged
  label crops. QC shows uncovered scale values with U. README illustration refreshed.
- Optional contrast/threshold slot OCR retries share original call/pixel/time
  budgets and retain raw runs, alternatives and skipped attempts. Pixel budgets
  are checked before resizing. The default remains one pass. A near-edge axial
  border is excluded from stroke measurement only when it is dark across >=80%
  of the bar, preventing several tick strokes from merging into one run.

## Real-world example and annotation audit

The unchanged supplied JPEG is the **Symbiotic crop** of Figure 4, DOI
[10.1038/s41598-024-53477-3](https://www.nature.com/articles/s41598-024-53477-3).
Credit: Duscher, Vroom and Foster, Scientific Reports 14, 2912 (2024), CC BY 4.0
(article image policy, subject to third-party credit exceptions).

It is 1043 × 969 pixels, with 33 clustered gene rows and eight time columns
2, 6, 8, 10, 12, 16, 18, 24. Manual heatmap coordinates are
[335,189,854,861]; bar core [18,39,217,103]; right-side labels
[858,189,980,861]; bottom labels [335,868,854,904]. Five reviewed tick anchors
are x=39,78,117,156,195 with values -2,-1,0,1,2 and about one-pixel positional
uncertainty. They do not cover physical edges x=18..216.

`examples/m16/manifest.json` records input/TSV hashes, transcription, displayed
order, preprocessing, both annotation passes and unresolved assumptions. The
second pass was an enlarged-crop audit by the same analyst, not an independent
annotation. One Greek suffix remains visually uncertain; no TSV or biological
vocabulary was used to select regions, calibrate values or correct OCR spelling.
The five approximately uniform ticks do not demonstrate a nonlinear numeric
scale; the piecewise workflow is explicit and approximate.

## First-try outcomes, including failure

The initial M16 run failed during label diagnostic rendering: an OCR punctuation
box narrower than one pixel produced an inverted Pillow rectangle. The partial
outputs and failure record are retained. Clamping only drawing extents fixed it;
OCR boxes and numeric matching were unchanged. The first completed trial is
retained separately from the subsequent border-stroke correction.

| Workflow | M15 baseline | First completed M16 | Current M16 trial |
| --- | --- | --- | --- |
| Supplied geometry/dimensions/calibration | 259/264 finite; linear extrapolation | 251/264 finite; no extrapolation | 251/264 finite |
| Automatic ROI, supplied dimensions | Abstained | Abstained | Abstained |
| Conditional legend localisation | 3/5 physical ticks | 3/5 | 5/5 |
| Exact numeric tick values | 1/5 | 1/5 | 2/5 |
| Label raw exact strings | 10/41 | 10/41 | 10/41 |
| Label heuristic accepted / exact accepted | 10 / 8 | 10 / 8 | 10 / 8 |
| Composed requested workflow | Abstained | Abstained | Abstained |

The 13 unresolved numeric cells include 12 with unsupported calibration spans
and five with colour ambiguity, overlapping in four cells. No physical-edge bound
was assigned. The manual CSV is explicitly a partial reconstruction; no complete
automatic export was produced. ROI failure reasons include `no_validated_grid`,
`no_continuous_bar` and nonunique association. Identifier requests retain review
restrictions even when geometry would otherwise resolve.

On the 251 finite cells, TSV **reconstruction agreement** is MAE 0.02228
(0.55697% of the labelled -2..2 range) and maximum 0.18636 (4.65896%). The full
264-cell denominator is retained; these errors do not describe the 13 abstentions.
Eleven TSV values lie outside the reviewed tick values. Source TPM, log transform,
standardisation axis/population, variance convention and clipping were not supplied.
The paper describes NanoString; a TPM provenance for this TSV is unestablished.
Across the displayed eight columns, row means range -0.638..0.773 and population
SDs 0.432..1.208. These summaries do not establish a standardisation recipe or
justify treating those eight columns as the whole population.

The label trial attempted **all 33 row and 8 column slots**; every slot had a
raw reading. Rows had 4 heuristic acceptances (2 exact), 23 ambiguities and 6
low-confidence results. Columns had 6 heuristic acceptances (all exact) and 2
low-confidence readings; all 8 raw column strings were exact. The initial README
illustration selected only three observations per axis for display; it has been
replaced with all 41 slot records and expected/observed/unresolved counts. This
was a display omission, not a three-label OCR limit. The existing grid-slot
association and unresolved-export guards already enforce the requested dimensions.

Current tick readings at -2,-1,0,1,2 are respectively `7`, `—`, `0`, `a`, `2`.
All five physical ticks were located and attempted; three readings parse as numbers
but only two are exact. The erroneous `7` is retained as unresolved, not accepted.
The M15/first-completed trial localised only three physical strokes because an
axial border merged others; the border-stroke correction recovered all five.

The optional retry trial made 123 calls versus 41, changing unresolved labels
31→34 and false heuristic acceptances 2→0. Raw exact strings changed 10→9;
accepted exact strings 8→7. This is a conservative tradeoff, not improved OCR
coverage. The first completed run took about 7.5 s for labels and 23.0 s for
retries; the current run under concurrent test/validation load took 12.9 s and
36.1 s. No retries were skipped in this crop. Greek characters, minus signs and
low-resolution glyphs still require per-slot review. High confidence cannot verify
biological identity or resolve conflicting signs.

## Frozen synthetic validation

The initial freeze and 42/48 result are retained. Before final evaluation, a
separate fresh partition was frozen after the rendering corrections and
pre-allocation pixel guard. The final partition has 16 new bases, 48 correlated
clean/JPEG85/blur0.65 attempts and six ambiguity/edge negatives. Colour trajectories
and numeric mappings vary independently across asymmetric/diverging colours,
unequal knot spacing, both numeric directions and both bar orientations.

Final numeric result: **43/48 complete and accurate**, zero wrong complete,
**0/6 negative completions**. All five ordinary abstentions remain in the
denominator. Worst finite-cell MAE/max across every attempt are 0.52911% / 3.58696%
of the eight-unit synthetic range; finite coverage is separately recorded per
attempt. Thresholds were frozen at MAE <2%, max <=5%, full finite coverage and
<=1% flags for completion.

Small underscore-label OCR records 288/720 raw exact strings and 256/451 exact
heuristic acceptances, leaving 195 false heuristic acceptances and **zero identifier
exports**. These results reinforce the existing identifier-review restriction.
They are synthetic conditional-crop evidence, not real-world identity accuracy.
Missing signs/conflicting readings and pixel/call exhaustion are covered by
focused tests and retained publication observations. Approximate colour-vision
projections preserve status letters/dashes; black ledger text has 21:1 contrast
against white. Original-coordinate quarter-turn checks remain in the regression
suite. Simulations are illustrative, not clinical vision modelling.

## Verification and remaining acceptance limits

The final-source full suite passes **571 tests and 18 subtests** with real OCR.
Focused checks are recorded in `benchmarks/m16/checks.json`. Ruff and
strict typing pass; typing also passes with NumPy 1.26.4. Source/wheel builds and
isolated CLI installation/export checks pass. Historical benchmark evidence and
legacy numerical thresholds were not weakened.

Remaining publication acceptance limits: missing TSV reconstruction recipe,
unreviewed physical edge values, ambiguous Greek transcription, poor raw OCR and
automatic ROI abstention on this supplied crop. No whole-publication automation,
original TPM recovery or stable maturity is claimed. The initial delivery was local; subsequent follow-ups are published to remote `work`.

[Reproduce the trial](../../../examples/m16/README.md) ·
[Benchmark evidence](../../../benchmarks/m16/README.md) ·
[Calibration and diagnostic usage](../../nonlinear-calibration.md)

## Follow-up: 1–2 pixel padding and complete field diagnostics

The user identified tight OCR crops and variable-height boxes. The latter were
ambiguous candidate unions rather than equal row fields. Explicit slot OCR now
keeps candidates in their originating slot and draws logical equal-pitch fields
separately from actual raw glyph boxes. Rounding and recorded padding are bounded;
wrong-centre or oversized candidates remain unresolved, without spreading them to
neighbouring rows. This supports regular grids; blocked/gapped layouts are not added.

Measured one source pixel plus one white margin pixel improves raw exact gene
strings **2→9 of 33**. All **8/8** sample strings are exact and heuristically
accepted. Four gene readings pass the confidence threshold, three exactly and one
incorrectly (`EsPGRPS` versus the image transcription `EsPGRP5`). Requested unresolved
slots fall **31→29 of 41**. No biological alias guessing or reference-vocabulary
correction was introduced. One missing row slot remains visible rather than shifting
subsequent rows.

Horizontal tick fields now share an image-measured top/bottom text band; vertical
fields share a left/right band. Signed text clusters determine the other bounds.
Two source pixels plus two white pixels recover the raw `-2`; a shared-budget
zero-white-border retry recovers `1`. All **five** actual OCR field rectangles are
visible in a dedicated enlarged legend. Proposals are `-2`, `—1`, `0`, `1`, `2`:
**4/5** parse exactly, and the em dash remains unparsed. Conflicting attempts and
low confidence remain review evidence, so no OCR-derived scale export is enabled.

The numeric matrix is bit-identical, including NaNs: **251/264 finite cells**.
Calibration, dense colour sampling, ambiguity thresholds and original-expression
limits remain unchanged. The final full regression run passes **588 tests and 18
subtests**; 107 focused tests cover padding, locale and vertical-minus safeguards.
Strict typing passes with current and minimum NumPy; packaging/CLI evidence is
recorded with this follow-up. The previous synthetic/publication evidence is retained.

[Follow-up evidence and commands](../../../benchmarks/m16/ocr-padding/README.md)

## Follow-up: numeric-only tick grammar and right-side gene margin

At the user's request, targeted numeric OCR now uses a numeric/punctuation
whitelist (including exponent syntax and observed sign glyphs). The parser maps
observed dash variants to `-` only at numeric sign positions, retaining raw text
and explicit Unicode normalization actions. Raw `—1` becomes canonical `-1`.
Standalone/range dashes and identifier strings remain invalid; missing signs and
character substitutions are not guessed.

The whitelist suppressed the previously readable `0` on this crop. The explicit,
budgeted raw-comparison retry recovers its actual unconstrained reading, passes it
through the same numeric grammar and retains the entire comparison. All **five
canonical tick proposals** are now exact: **-2,-1,0,1,2**. Low confidence and
conflicts remain review evidence; numerical scale export restrictions are unchanged.

Gene fields add **four right-only source pixels**, preserving row heights and
original-coordinate ownership. The trailing `1` in `EsGalectin1` was **not recovered**:
raw `EsGalectin` has confidence 2 and remains unresolved. The 2..12 pixel expansion
probes and resize/contrast/threshold attempts are retained as failures. Whole-crop
raw exact genes/samples remain **9/33** and **8/8**; 29/41 requested slots remain
unresolved. No reference-vocabulary or manual suffix correction was injected.
Numeric heatmap values are bit-identical, including their existing NaNs.

[Numeric signs/right-margin evidence](../../../benchmarks/m16/numeric-signs/README.md)


This follow-up passes **606 tests and 18 subtests**, plus lint, strict typing with
current/minimum NumPy and source/wheel/isolated CLI checks. The focused sign/margin
suite passes 134 tests (one opt-in skip). The matrix comparison and Galectin failure
are retained in the follow-up evidence.
