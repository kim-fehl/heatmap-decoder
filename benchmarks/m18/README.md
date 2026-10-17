# M18 frozen label geometry and spelling evidence

The source starts at M17 commit `ef8636c`. `baseline.json` preserves numeric value
and flag hashes for a regular synthetic figure, the M17 blocked example and the
M16 publication crop. Earlier reports remain unchanged.

The registered validation has **42 fresh base figures**, each clean/JPEG/blur,
covering 0, ±15, ±30, ±45, ±60, ±75, ±90 and 37.25°. It varies above/below placement,
blocked columns, DejaVu Sans/Mono at 14–16 pixels, numeric seeds and digit strings;
selected regions and matrix counts are supplied. Text polygons/strings in the
reference annotations are evaluator-only. Fonts and model/runtime dependencies
are fingerprinted before evaluation. All 126 correlated attempts stay in their
registered denominators.

Explicit-angle and bounded automatic directed-angle/attachment geometry are correct
on **42/42 clean, 42/42 JPEG and 42/42 blur** cases. Zero wrong accepted associations
were observed. Automatic geometry uses the same production segmentation, attachment,
baseline-direction and geometric angle primitives, with a shared pixel/time budget;
this engine-independent geometry ceiling does not claim whole-image ROI/count/OCR
workflow success. Full detector integration, caps and contradictory flips are covered
by deterministic tests and the runnable real-engine example.

Five challenge bases retain ambiguous or incomplete decisions: uniform-height
all-capital glyphs, overlapping lines, truncated regions, incorrect endpoint
configuration, and a missing name plus group header. Additional engine-independent
regressions cover both signs/directions, non-integer point/polygon round-trips,
missing/duplicate strings, conflicts and shared exhaustion. Negative evidence is
finite and does not prove zero population risk.

Real Tesseract reads **56/56** exact raw strings in the 14-angle subset. Default
RapidOCR reads **2/16** exact raw strings in its four-angle subset, with **14 wrong
strings above the heuristic threshold**. All 72 recognized identifiers remain
unreviewed: no confidence value authorises a labelled export. Exact geometry and
spelling are separate metrics. This is a renderer/font-specific spelling sample,
not publication identifier qualification.

The gates are explicit clean geometry 100%, automatic clean geometry ≥80%, and
zero observed wrong accepted associations. All pass in the preregistered support
scope. Font/foreground/line-spacing generalisation and new publication trials remain
unqualified. Existing M17 JPEG/blur numeric automation remains incomplete.

Reproduce with the frozen source/runtime/inputs from the repository root:

```sh
uv sync --locked --group dev --group benchmark --extra ocr
uv run python benchmarks/m18/evaluate.py evaluate --freeze benchmarks/m18/freeze.json --out work/m18/reproduced-validation.json
```

The evaluator verifies hashes, engine/model/font versions and policy/plan, then
refuses overwrite. `frozen-source.tar.gz` archives the exact source/tests/evaluator
and lockfile. Derivatives are correlated; 42/42 clean bases give a Wilson 95% interval
of about 92–100%. The five challenge bases do not establish a publication release.

See [configuration](../../docs/arbitrary-angle-labels.md), [M18 status](../../docs/progress/M18.md)
and `validation.json` for original polygons, decisions, unresolved slots, coverage,
raw/heuristic spelling errors, runtime and supplied-hint provenance.
