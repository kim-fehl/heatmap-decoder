# Current-phase review corrections

These corrections address the review of M12 commit `9efb175`. They do not declare
M13, a stable capability release or recovery of original expression counts.
Historical M9–M12 benchmark files remain unchanged. Inspected M12 fixtures are
regression/development evidence, not a fresh hold-out.

## Recognition and acceptance

- **Colourbar components are search windows.** Original-pixel transverse coherence
  and smooth longitudinal variation identify cores in both orientations, including
  grayscale. Moving tick text cannot move the core by a midpoint heuristic.
  Thin black borders and legitimate white palette centres are handled separately.
  Multiple cores, rejected appearances and exhausted search budgets remain evidence.
- **Competition blocks selection.** Both labelled and unlabelled copies survive
  proposal generation. Text presence and candidate rank cannot choose the target.
  Search truncation blocks unsupported uniqueness claims.
- **Blur/JPEG edges use profiles.** Robust samples beyond the halo estimate the
  transition, background noise, cross-axis support and location uncertainty.
  The background-spread limit remains 8 RGB units. Aggregated gradient peaks test
  phase against observed boundaries; every possible boundary need not be visible.
  These uncertainty bounds are engineering diagnostics, not calibrated intervals.
- **Counts have a conditional positive path.** Independent colour transitions
  measured away from strokes, grid strokes or raw unassigned label centres must
  agree. At least one sensor must support a complete lattice. Missing labels retain
  holes; ambiguous harmonics, competing counts and cap-hidden alternatives abstain.
  The model assumes a complete regular rectangular grid with at least three cells
  per inferred axis and usable interiors. Raster data cannot prove absence of all
  invisible subdivisions. Selected counts resolve a new config without changing
  supplied counts or the original request.
- **OCR availability and accuracy are separate.** Plausible bars receive bounded
  diagnostic tick OCR even when geometry remains unresolved. Conditional annotated-
  bar and uniquely detected-bar results measure strokes, text, exact values, signs
  and fits separately from end-to-end coverage. They never enable extraction.
  Two OCR ticks, disagreeing values and unsupported nonlinear scales need review.

## Experimental export

`--experimental-export` is an explicit runtime control, excluded from request JSON.
It permits numeric-only tables with supplied counts and supplied/reviewed linear
scales, default detector settings and every per-image decision accepted. Inferred
counts, OCR scales and identifiers remain outside this export scope. Default
operation produces review diagnostics. Failed requests preserve existing matrices.

Outputs retain the original request, resolved config, decisions, hashes, per-cell
quality and experimental policy/maturity. Stable capability flags remain false.
A 70% coverage target measures usefulness; it cannot authorise a particular export.
See the [worked command and overrides](automation.md#experimental-numeric-export).

## Validation design and limits

The [new preregistration](../benchmarks/corrections/preregister.json) defines
12 development and 24 validation bases, each with clean, blur sigma 0.65 and
JPEG quality 85 derivatives kept in the same partition. All figures pass text/text
and text/core separation, margin, box-extent and numeric-precision assertions
before recognition. Pillow and Matplotlib renderers use three font families,
20/25/30 genes, 6/8/10 samples, all bar-side/orientation combinations, left/right
row labels and above/below sample labels. Five symmetric/asymmetric ranges,
both directions, grid/no-grid and 3/5/7 ticks are covered.

Eleven named challenges cover absent/categorical/competing bars, multiple panels,
hidden boundaries, partial crops, wrong supplied counts, missing text, two ticks,
nonlinear ticks and touching core/text. Applicability differs by request mode:
missing tick text does not invalidate a scale deliberately supplied by a person.
Every ordinary input stays in the coverage denominator.

Accuracy checks require heatmap edges within 0.1 pitch, bar edges within 1 pixel,
MAE below 2% of displayed range, maximum cell error at most 5%, complete finite
output and at most 1% flagged cells. Wrong complete selections and negative exports
are reported explicitly. Conditional OCR is diagnostic evidence only.

The fresh split changes seeds and varies layouts/renderers/fonts, but shares
renderer code and font families across partitions. It is **synthetic validation,
not unseen-source generalisation**. No independently annotated lawful publication
figures were supplied, so reliability on real expression heatmaps is unvalidated.
Gene identities have no database validation and require human review. Quantised
colours, clipping and non-invertible transformations limit the decoded table.

Measured results and reproducibility hashes accompany the benchmark reports.

## Inspected M12 regression

With supplied counts and scale, the corrected classifier accepts **92/135**
ordinary inputs (68.1%), compared with historical M12's 11/135 (8.1%). All 92
accepted selections pass the numerical/geometry checks; all six applicable
negative cases abstain, including competing bars. This is inspected regression
evidence, not a fresh hold-out or a real-publication reliability estimate.
The [retained report](../benchmarks/corrections/m12-regression.json) contains every
input, decision and error measurement.

The complete regression suite passes 462 tests plus 18 subtests with real OCR.
Mypy passes with NumPy 1.26.4 and 2.5.4; CI now checks both minimum/current
NumPy typing. The foreground mask explicitly returns a boolean ndarray, fixing
version-dependent `no-any-return`. The built wheel also exports the worked
20 × 6 example in an isolated environment without OCR or Matplotlib installed.

## Frozen validation results

[Validation report](../benchmarks/corrections/validation.json) ·
[Compact results](../benchmarks/corrections/summary.json) ·
[Source/evaluator freeze](../benchmarks/corrections/policy-freeze.sha256) ·
[Engine, font and dependency environment](../benchmarks/corrections/environment.json)

| Request mode | Complete / ordinary | Coverage | Wrong complete | Experimental exports |
| --- | ---: | ---: | ---: | ---: |
| Supplied counts and scale | 33/72 | 45.8% | 0 | 33 |
| Inferred counts, supplied scale | 3/72 | 4.2% | 0 | 0 |
| Supplied counts, OCR scale | 16/72 | 22.2% | 0 | 0 |
| Inferred counts and OCR scale | 2/72 | 2.8% | 0 | 0 |

All applicable negative requests abstained (8/7/11/10 per mode), with zero
negative exports. Supplied-count heatmap bounds passed 72/72, and bar checks
passed 66/72. Remaining supplied-scale failures primarily involve excessive
colour flags or ambiguous/missing cells; geometry recovery alone cannot ensure
a reliable table. Worst accepted MAE was 0.34% and maximum cell error 1.86% of
the displayed range across all modes. No thresholds were tuned after validation.

Conditional annotated-bar and uniquely detected-bar OCR each found/localised
**342/342** ticks and recognised **306/342** values exactly (89.5%); ten sign
errors remain in the raw evidence. Linear-scale checks passed 34/72 conditional
images. These experiments do not contribute to automatic coverage or authorise
OCR-scale export. Count coverage remains low, despite a working positive path.

The 70% usefulness target remains unmet. Zero observed errors in this small,
clustered synthetic sample does not establish a production error rate; real
publication images and biological identifiers remain unvalidated.
