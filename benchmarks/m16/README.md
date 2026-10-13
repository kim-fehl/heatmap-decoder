# M16 evidence

Historical M15 and corrections evidence is unchanged. `baseline.json` is the M15
runtime at `2ec504d29df73636880540e093521c6fb675035f`, run on the supplied crop.
To reproduce it, archive that commit's `src` into `work/m16/baseline-source`, then
run `baseline.py` with `PYTHONPATH=work/m16/baseline-source/src`. It writes scratch
outputs to `work/m16/baseline`. The baseline intentionally measures the historical
linear extrapolation and does not certify physical edge values.

`first-trial-failure.json` retains the initial diagnostic failure from a subpixel
punctuation box. `first-completed-trial.json` retains the corrected first full run.
`first-trial-source.json` describes the initial source; the drawing correction
clamped boxes below one pixel without changing OCR geometry or numerical results.
`current-trial.json` includes the subsequent narrow border-stroke correction.
All attempts and abstentions remain visible; no held-out-publication claim applies.

```bash
uv run python benchmarks/m16/real_trial.py --out work/m16/new-trial
uv run python benchmarks/m16/validate.py --split development --out work/m16/new-dev.json
uv run python benchmarks/m16/validate.py --split validation --out work/m16/new-validation.json
```

All runners refuse report overwrite. Validation verifies `freeze-validation.json`
before recognition. `freeze-validation-first.json` / `validation-first-freeze.json`
retain the earlier evaluation. Before the **final fresh** validation partition,
QC rendering was corrected to show uncovered cells with U, ledger font readability
was improved, and the shared pixel budget was moved ahead of crop allocation.
The final freeze includes all current source/evaluator hashes and thresholds.
There are 16 fresh validation bases, 48 correlated clean/JPEG85/blur0.65 attempts,
plus six calibration/ambiguity negatives. Numeric mappings vary separately from
colour trajectories: both bar orientations and directions, unequal knot spacing,
asymmetric/diverging palettes, linear and monotonic nonlinear maps. Negatives
cover physical edge extensions, repeated colours and saturated plateaus. Small
underscore-bearing labels are evaluated independently using explicit crops.
Greek and conflicting/sign OCR evidence is in the publication report and focused
tests. Numeric checks require MAE <2%, max <=5%, full finite coverage and <=1%
flagged cells. Every failed derivative remains in the denominator. Text heuristics
are reported; they never enable automatic identifier export.

`development.json` is the initial numeric-only development trial;
`development-with-text.json` is the expanded inspected development trial. The
validation shares renderer/palettes/fonts/layout with development but changes base
seeds; it is synthetic evidence with limited distribution diversity. A finite set
with zero wrong complete numeric decisions cannot establish a production error rate.
