# Review correction evidence

This directory is separate from the frozen M12 evidence. `corpus.sha256` freezes
renderer and preregistration after layout QA, before recognition. Development
permits correction; validation runs after the source/evaluator policy freeze.
Reports refuse overwrite. Derivatives remain with their base.

```bash
uv sync --locked --extra ocr --group benchmark
sha256sum -c benchmarks/corrections/corpus.sha256
uv run --extra ocr --group benchmark python benchmarks/corrections/evaluate.py \
  --split development --conditional-ocr --out work/development.json
uv run --extra ocr --group benchmark python benchmarks/corrections/evaluate.py \
  --split validation --conditional-ocr --out work/validation.json
```

Install Tesseract English data and DejaVu fonts separately. Matplotlib is a
benchmark dependency; the installed decoder does not require it. `development.json`
and `validation.json` retain all request-specific decisions, geometry, count,
numeric and OCR evidence. Conditional annotated/detected-bar OCR is never counted
as automatic coverage. The runtime receives requested counts/scales only in their
respective supplied modes; no truth boxes or count answers enter inferred modes.
Inferred requests uniformly enable optional raw label geometry, independent of
fixture layout. See [scope and limitations](../../docs/corrections.md).

Validation used two disjoint base shards on a two-CPU environment:

```bash
# Run index 0 and index 1 independently, retaining all derivatives with each base.
uv run --extra ocr --group benchmark python benchmarks/corrections/evaluate.py \
  --split validation --shards 2 --shard-index 0 --conditional-ocr --out work/part0.json
uv run --extra ocr --group benchmark python benchmarks/corrections/evaluate.py \
  --split validation --shards 2 --shard-index 1 --conditional-ocr --out work/part1.json
uv run --extra ocr --group benchmark python benchmarks/corrections/merge.py \
  work/part0.json work/part1.json --out work/validation.json
```

The merger verifies disjointness, complete input coverage and matching policy;
`validation.json` retains both shard hashes and every original decision. Timing
is per request under this concurrency, not a serial throughput estimate.
`summary.json` presents compact results. `m12-regression.json` is separately
labelled inspected development regression (92/135 mode-1 complete, zero wrong).
Historical M12 source hashes belong to its original commit, not corrected source.
