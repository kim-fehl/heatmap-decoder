# M12 evidence

These are new MIT synthetic biological-layout figures, not a published-figure
benchmark. `preregister.json`, `challenge_scope.json` and `corpus.sha256` fix split,
ordinary quality envelope, derivative grouping and gate criteria before end-to-end
evaluation. `policy-freeze.sha256` fixes classifier/scorer code before validation
and test. No policy tuning follows test inspection.

- `development.json`, `validation.json`, `test.json`: deployed choices for four
  modes, with subsequent truth scoring, failures and diagnostic quality retained.
- `*-metrics.json`: additional truth metrics from frozen choices, including the
  expected-tick denominator and numeric errors; never selects another candidate.
- `strings-before.json`, `strings-after.json`: conditional development-only OCR
  comparison on known bars; not automatic pipeline coverage.
- `identifiers-development.json`: sparse/slot comparison on known grids, including
  high-confidence errors and preserved raw strings.
- `renderer-audit.json`: independently detects text overlap in the annotated
  renderer; no cases are removed from the preregistered denominator.
- `environment.json`: observed engine/language/data/font environment and timing
  limits. `m10-historical.sha256`/`m11-historical.sha256` verify older files unchanged.
- `release-decision.json`: measured per-mode gate failures; no released exports.

Each split has 10 challenge bases; ordinary bases number 15/15/45 with three
versions each. Different seeded matrices define base figures. Layout templates,
font, gene list and renderer are shared across splits, limiting independence;
this is not validation on unseen publication sources. See [docs/progress/M12.md](../../docs/progress/M12.md)
for gates, results, commands and limits. Running an inspected test again is a
regression run, not a new held-out evaluation.

```bash
uv run --extra ocr python benchmarks/m12/evaluate.py \
  --split development --out work/m12-development.json
uv run python benchmarks/m12/analyse.py work/m12-development.json \
  --out work/m12-development-metrics.json
```

Report commands refuse existing output paths. CI runs the development benchmark
and the abstention workflow; it does not rewrite frozen release evidence.
