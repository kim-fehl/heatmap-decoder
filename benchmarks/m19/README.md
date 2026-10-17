# M19 registered validation

Run from the repository root after installing OCR/benchmark extras and Tesseract:

```bash
uv run python benchmarks/m19/evaluate.py freeze --freeze NEW_FREEZE.json
uv run python benchmarks/m19/evaluate.py evaluate --freeze NEW_FREEZE.json --output NEW_REPORT.json
```

The evaluator refuses to overwrite evidence and checks source/runtime/model/font
fingerprints before and after evaluation. `freeze.json` records the initial failed partition. `freeze-final.json` registers
40 fresh bases,
120 correlated clean/JPEG/blur attempts, 12 negative bases, thresholds and hint
provenance. Each corresponding `.tar.gz` preserves the evaluated code. Previously inspected
seeds 290001–290004 and development seeds 291001–291008 are not validation bases.

Every base measures explicit extraction, automatic geometry/counts with reviewed
scale, fully requested OCR scale, and supplied-geometry/count OCR scale. Paired
independent wide-pitch label figures measure explicit and automatic angles and
composed numeric retention. Those paired label trials supply ROIs/counts; they do
not claim whole-figure label localisation success. No truth text/polygons or tick
values enter detection. Family/degradation denominators include failures.

`validation.json` retains the initial negative failures; `validation-final.json`
retains decisions, blocker chains, calibrated spans, raw tick
alternatives, exact intervals, numeric errors, flags, runtime and actual hints.
`publication.json` records the sole available inspected publication crop and the
unmet six-figure/three-held-out requirement. It preserves the earlier user access
constraint and does not substitute synthetic figures for unavailable publications.

The final partition replaces the interrupted repair partition documented in
`interrupted-repair.json`. No failed or interrupted evidence was overwritten.

The pre-export calibration/count decisions are measured while new qualification
gates are closed. Any subsequent export-policy or adapter changes are recorded
separately from this frozen report; inspected validation does not become a fresh
partition. Older M17/M18 evidence remains untouched.
