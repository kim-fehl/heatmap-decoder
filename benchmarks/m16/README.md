# M16 evidence index

## Current implementation

[Validation with the RapidOCR default](validation-rapidocr-v1/README.md) is the
current acceptance record. Its source/runtime/model hashes and fresh base seeds
were registered at commit `6bddb93` before evaluation. It includes both label
strategies, numerical export gates, negative cases, runtime and an updated
five-workflow publication trial. See [M16 status](../../docs/progress/M16.md) for
conclusions and remaining source-data limits.

## Historical evidence

| Record | Stage |
|---|---|
| `baseline.json` | M15 at `2ec504d`; historical linear extrapolation, not certified physical edge values |
| `first-trial-failure.json` | Initial M16 diagnostic rendering failure |
| `first-completed-trial.json` | First completed run after the subpixel-frame fix |
| `current-trial.json` | Subsequent border-stroke correction; “current” is its original historical filename |
| `freeze-validation-first.json`, `validation-first-freeze.json` | First frozen evaluation |
| `freeze-validation.json`, `validation.json` | Earlier final partition, before the later OCR changes |
| [Padding](ocr-padding/README.md) | Field ownership, source/white padding and shared tick bands |
| [Numeric signs and margins](numeric-signs/README.md) | Audited dash normalization and the retained Tesseract suffix failure |
| [White-border probes](white-padding/README.md) | Manually tightened Tesseract recovery and failed generic variants |
| [Independent OCR comparison](rapidocr/README.md) | Comparison that motivated the default switch |
| [Default integration](rapidocr-default/README.md) | Production adapter, provenance and packaging verification |

Old `validate.py` and `real_trial.py` runners reproduce their original protocols,
not the latest default-engine acceptance. Frozen historical validation must run
with its matching archived source; a hash mismatch is an integrity check, not a
reason to rewrite an evaluated freeze. Earlier source snapshots, derivatives,
failures and abstentions remain retained.

All publication comparisons are reconstruction agreement against an undocumented
TSV recipe. Synthetic and inspected-crop results do not establish a production
error rate or held-out publication accuracy. Thresholds, complete-case coverage,
false text acceptances and abstentions must be read together.
