# M8 reviewed regions and count alternatives

See [docs/progress/M8.md](../../docs/progress/M8.md) for the full schema, commands, measured quality and limits.
`hints.json` requests bar/label regions around the fixed M6 labelled figure.
`selections.json` is its explicit fixture review; inspect the diagnostic before
using it. `workflow.py` runs that reviewed CLI example and verifies exact labels
and unchanged numerical values. It does not select candidates by score.

`count_hints.json` explicitly requests independent row/column count proposals on
`count_complete.png`. Expected exit 5 leaves counts absent in proposed config;
true counts and competing multiples remain in JSON. Supply/review counts before
requesting final label assignment or numerical reconstruction.

`generate.py` recreates MIT synthetic figures and annotations; base layouts/splits
are declared before evaluation. `benchmark.py --ocr` records all bar/count/label
attempts in the same format as `benchmark.json`, retaining failures. Omit `--ocr`
for a report requiring no OCR extra/engine. Every proposal remains review-only.
