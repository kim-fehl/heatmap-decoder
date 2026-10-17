# M10 development evidence

`evaluate.py` runs the deployed development classifier before consulting truth.
It evaluates 96 larger synthetic layout variants, eight new challenges and 32
historical M9 ROI inputs. It also records conditional count experiments on manual
bounds and optional OCR on manually annotated label regions. These experiments
are separate; no complete identity/count automation is claimed.

`report.json` includes every attempt and all decision reasons. Geometry tolerances
are 0.1 **true** cell pitch and one pixel for bar edges. Numeric errors use the
known plotted matrix, with finite coverage, maximum error and flag fraction
reported separately. Recognition uses exact strings, with per-slot wrong
suggestions and unresolved slots retained. Performance is wall-clock detector
and classifier time; it excludes file I/O and optional OCR.

All data is original MIT synthetic imagery. M9 files are read-only historical
regression data, already inspected during development. New layouts reuse three
seeded matrices and derivative images remain dependent. No data here is an
untouched M12 hold-out or independently annotated published figure. Inferred
counts and released automatic exports remain disabled. The current thresholds
must be frozen and tested against a separately split corpus in M12.

```bash
uv run --extra ocr python benchmarks/m10/evaluate.py --ocr --out work/m10-report.json
```
