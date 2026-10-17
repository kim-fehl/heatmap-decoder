# M14 grid geometry repair

The coverage investigation was followed by a bounded repair of terminal gridlines,
blurred strokes and JPEG grid evidence before M15. Numeric ambiguity thresholds,
bar endpoint verification and export scope are unchanged. Acceptance evidence
changed, so the detector is now `grid-strokes.1` and policy `review-acceptance.3`.

## Changes

- Keep a thin terminal dark row/column when it continues at least three equal
  spacings and has the same width as the internal strokes. Apply this to right
  and bottom terminal strokes; continue trimming separate exterior frames.
- Find narrow dark troughs supported across transverse samples. Use bounded
  quadratic trough localisation and per-sample dispersion for centre uncertainty.
  Preserve the observed gradient width separately in evidence. Group ringing
  around a verified trough into that one stroke. Non-stroke broad transitions
  retain their original width-based uncertainty.
- Estimate local raster noise from independent one-pixel interior differences.
  Subtract at most half of the measured interior residual for equal-grid scoring.
  Preserve the original piecewise-constant score, acceptance thresholds, residual,
  local noise and remaining structural residual in the evidence.

The new noise estimate makes the grid evidence less sensitive to JPEG artifacts;
this is a change to the score definition, not a claim that the same numeric score
has identical meaning across policy versions. All matching and QC thresholds are
unchanged. Broad uncertain boundaries, low-contrast bar endpoints and unresolved
numeric cells continue to require review.

## Results

| Synthetic partition | Before | After | Wrong complete | Lost complete |
| --- | ---: | ---: | ---: | ---: |
| Original development | 34/84 (40.5%) | 49/84 (58.3%) | 0 | 0 |
| Inspected original validation | 33/84 (39.3%) | 50/84 (59.5%) | 0 | 0 |
| Frozen new matrix seeds | — | 51/84 (60.7%) | 0 | — |

The original validation is now development regression evidence. It was inspected
before these repairs. A complete source/evaluator freeze and new-case manifest
were recorded before recognition on new seeds. The additional seed partition
uses the same generator and layouts; it does not establish publication-image
or independent layout generalisation.

All three runs reject all seven negative families, with zero negative exports.
The broad dark-band challenge accepts correctly; the touching colourbar frame
remains review-required. Each report includes all 84 ordinary attempts and nine
challenges, with truth and diagnostics retained. Original reports are untouched.

On the inspected original validation, heatmap verification improves from 46 to
69 accepted figures. All 17 newly complete matrices satisfy truth error limits.
Accepted clean/JPEG/blur counts improve from 14/12/7 to 20/18/12 out of 28 each.
Worst accepted mean error remains 0.4253% and maximum cell error 2.1792% of scale.
For the new seed partition, worst accepted mean error is 0.4693% and maximum
cell error 2.0842% of scale.

New-seed acceptance by palette (12 attempts each): viridis 12, magma 5,
cividis 11, grayscale 8, blue-white-red 10, teal-black-red 5, asymmetric 0.
Coverage remains below the M14 target. Remaining losses include weak boundary
contrast, genuinely cropped dark endpoints, uncertain row transitions and colour
ambiguity. The repairs do not justify filling ambiguous numeric cells.

## Verification and evidence

Ten focused tests exercise final strokes versus external frames, measured blur
centres, JPEG reconstruction, irregular spacing, wide strokes, strong blur,
wrong counts and smooth within-cell structure. The full suite passed: **534 tests
and 18 subtests**, including real OCR integration. Ruff and strict mypy passed
with current NumPy and minimum supported NumPy 1.26.4. Source/wheel builds and
the isolated wheel installation smoke check passed.

The historical supplied-count/scale regression retains 34/72 complete matrices,
with zero incorrect complete decisions, zero lost historical acceptances and
zero exports across eight negative cases. This matches the pre-repair M14
regression result.

[Evidence and reproduction](../../benchmarks/m14/repair/README.md) include frozen
sources, cases and complete reports. `runtime.patch` records the exact four-file
change from the pre-repair M14 runtime; reversing it reconstructs that runtime.
Original M14 report checksum manifests remain valid; their historical runtime
freeze is not a manifest of the newly repaired source.

M15 has not been started.
