# M17 frozen geometry evaluation

`baseline.json` captures the M16 whole-image and supplied-hint workflows at
`452c8c0` before detector changes. `baseline-after.json` repeats those workflows;
all six completion decisions are unchanged. Earlier milestone reports are untouched.

Four sequential freezes use disjoint seeds. Keep failures visible:

| Report | Clean no-gap / horizontal / vertical / both | Negatives accepted | Strict wrong complete |
|---|---|---:|---:|
| `validation.json` (first freeze) | 10 / 10 / 10 / 6 out of 10 | 1/12 | 14 |
| `validation-repair.json` | 10 / 10 / 10 / 10 out of 10 | 1/12 | 1 |
| `validation-final.json` | 9 / 10 / 10 / 10 out of 10 | 0/12 | 0 |
| `validation-verified.json` (test assertion migration only) | 10 / 10 / 10 / 10 out of 10 | 0/12 | 0 |

“Strict wrong complete” includes any accepted case with mismatched original cell
intervals, even if all numerical values meet the error limits. The earlier
coordinate mismatches were numerically accurate; the negative acceptances were
partial-matrix errors. They triggered repairs and new freezes, not denominator
changes. Earlier freezes retain source fingerprints but their intermediate source
snapshots were not archived; the third and fourth snapshots are archived and reproducible here.

The final partition has **40 bases × clean/JPEG/blur = 120 correlated attempts**,
plus **12 negative bases**. It balances all four gap families, varies dimensions,
pitches, gap widths, unequal group sizes, palette and legend orientation/direction.
Explicit clean extraction passes 40/40. Automatic geometry/counts with supplied
reviewed scale passes 40/40 clean and 57/120 overall. The 70% ordinary-derivative
gate fails; JPEG/blur automation is incomplete. No cases were excluded after OCR
or evaluation; no OCR is used in this numerical M17 benchmark. Identifier geometry
is covered by deterministic slot-ownership and existing rotation regressions.

`summary.json` includes each family/degradation and earliest blockers; full reports
retain intervals, alternatives, rejection evidence, coverage, flags, dimensions,
numerical errors, decisions, runtime and supplied-hint provenance. Targets and
eligibility were frozen before each run. Ten clean bases per family provide limited
evidence: 10/10 gives a Wilson 95% interval of about 72–100%; 9/10 about 60–98%.
Derivatives must not be treated as independent population trials.

The frozen generator is evaluation code: truth never enters production detection.
Only the reviewed scale is supplied to automatic workflows. Explicit extraction
is a separate reconstruction ceiling. Publication baseline trials remain
supplied-hint/development evidence, not held-out publication qualification. Six
new publication figures, arbitrary angles, automatic OCR scales and composed label
workflows are later-phase work.

Reproduce the final partition from the repository root and the locked dependencies:

```sh
uv sync --locked --group dev --group benchmark --extra ocr
mkdir -p work/m17
uv run python benchmarks/m17/evaluate.py evaluate --freeze benchmarks/m17/freeze-verified.json --out work/m17/reproduced.json --seed-start 210001 --negative-seed-start 211001
```

The evaluator verifies all source/runtime/plan hashes and refuses overwrite.
`frozen-source-verified.tar.gz` archives the exact verified source, tests, generator and
lockfile referenced by that freeze. The first two reports are regression/development
evidence after inspection; do not use them as fresh validation of repairs.

See [M17 status](../../docs/progress/M17.md) and [configuration](../../docs/blocked-grids.md).
