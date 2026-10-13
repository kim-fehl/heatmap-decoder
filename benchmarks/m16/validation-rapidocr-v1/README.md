# M16 current acceptance: RapidOCR default

**Acceptance passed** for the current implementation under the registered numeric
scope. Source/runtime/model/input hashes and thresholds were committed in
`6bddb93` **before evaluation**, using fresh positive seeds 176301..176316 and
negative seeds 176317..176322. Existing completed freezes remain unchanged.

## Protocol and thresholds

The 16 bases vary colour trajectory and numeric mapping independently: two colour
families, linear/nonlinear numeric mappings, both bar orientations and both numeric
directions. Each has clean, JPEG85 and blur0.65 derivatives, kept with its base.
Six negatives cover repeated colours, plateaus and unsupported edge extensions.
The renderer is shared with earlier synthetic studies; only the base seeds are
fresh. This is not held-out publication evidence.

The existing per-case numerical targets remain MAE <2%, maximum ≤5%, full finite
coverage and ≤1% flagged cells, normalized by displayed range. Registration also
requires ≥80% complete-and-accurate positive attempts, zero wrong complete/exported
cases, zero negative exports and no identifier export authorization. No threshold
or partition was changed after observing results.

## Results

| Numeric result | Outcome |
|---|---:|
| Complete and accurate / all ordinary attempts | **43/48 (89.58%)** |
| Wrong complete or exported numeric cases | **0** |
| Negative completions / exports | **0/6** |
| Worst finite-cell MAE / maximum, normalized | **0.514% / 3.587%** |

Five JPEG attempts abstain. Each remains in the 48-attempt denominator with
coverage and flags; low error among finite cells does not count as completion.
Actual experimental numeric-export decisions are exercised for every attempt.

| Label strategy | Raw exact | Accepted exact | False acceptances | Unresolved | Calls | OCR wall time |
|---|---:|---:|---:|---:|---:|---:|
| Sparse | 512/720 | 512/624 | 112 | 96/720 | 96 | 194.33s |
| Per-slot | 672/720 | 592/592 | 0 | 128/720 | 720 | 24.67s |

Both use the production RapidOCR/native-size defaults. The 720 label slots per
strategy include all 48 attempts. Source-field/detected-text geometry, raw tokens,
confidence, slot statuses and skipped attempts are retained. No
identifier exports were authorized; high confidence alone does not certify names.
The zero false acceptances in per-slot synthetic labels does not extend to the
publication trial, which still has six wrong accepted gene names.

Total evaluator wall time was 229.31s. Model warm-up is included in the first OCR
call. The publication workflow ran concurrently, so these are observed runtime
costs rather than isolated performance claims. Hardware/load and engine internals
can change timing.

## Publication rerun

[`publication.json`](publication.json) repeats all five workflows with current
settings, consulting reference strings/values only for evaluation:

| Workflow | Result | Wall time |
|---|---|---:|
| Supplied geometry and reviewed calibration | 251/264 finite cells; partial | 0.16s |
| Automatic ROI, supplied dimensions | Abstains | 5.82s |
| Numeric tick localisation/OCR | 5/5 exact physical ticks; unresolved scale | 5.68s |
| Default gene/sample labels | 34/41 raw exact; six wrong accepted; one unresolved | 5.37s |
| Composed workflow | Abstains | 1.05s |

The exact label total is 26/33 genes plus 8/8 samples, including `EsGalectin1`.
TSV agreement is MAE 0.02228 and maximum 0.18636 on 251 finite cells, with all 264
cells in the coverage denominator. It is reconstruction agreement, not validation
of an unknown source-data recipe. Physical-edge values and one Greek transcription
remain unreviewed. No automatic publication export is authorized.

## Artifacts and reproduction

- [`freeze.json`](freeze.json): source, runtime, model, font, input hashes, case plan and policy.
- [`results.json`](results.json): numerical/label aggregates, negatives and acceptance result.
- [`results.jsonl`](results.jsonl): all 48 attempt records, raw OCR, coverage, flags and timing; its hash is bound into the summary.
- [`publication.json`](publication.json): complete current five-workflow report.

From the repository root, with the exact versions recorded in the freeze and local
DejaVuSans font available:

```bash
uv sync --locked --extra ocr --group benchmark
uv run python benchmarks/m16/validate_current.py run \
  --freeze benchmarks/m16/validation-rapidocr-v1/freeze.json \
  --out work/m16/reproduced-validation.json
uv run python benchmarks/m16/current_trial.py \
  --freeze benchmarks/m16/validation-rapidocr-v1/freeze.json \
  --out work/m16/reproduced-publication
```

The publication rerun additionally requires local Tesseract/English data. The
model hashes in this freeze cover RapidOCR; the publication report separately
retains Tesseract engine/wrapper metadata. Source
and runtime/model mismatches are rejected before generating any validation image;
source/runtime checks repeat after evaluation. Runners refuse to overwrite evidence.
A new implementation requires a separately registered new partition; do not refresh
this completed freeze. The original registration command is:

```bash
uv run python benchmarks/m16/validate_current.py freeze \
  --seed-start 176300 --out PATH_TO_A_NEW_FREEZE.json
```

That command records a plan only. Do not reuse these evaluated seeds as a new
unseen validation partition. [Historical evidence](../README.md) is retained.
