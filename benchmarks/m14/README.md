# M14 palette-diverse reconstruction and QC audit

## Reproduce

    uv sync --locked --extra ocr --group benchmark
    sha256sum -c benchmarks/m14/corpus.sha256
    uv run --extra ocr --group benchmark python benchmarks/m14/evaluate.py --split development --out work/m14-development.json
    uv run --extra ocr --group benchmark python benchmarks/m14/evaluate.py --split validation --out work/m14-validation.json
    uv run --extra ocr --group benchmark python benchmarks/m14/regression.py
    uv run --group benchmark python benchmarks/m14/audit.py

Evaluation refuses to overwrite retained reports. The regression evaluator also
refuses overwrite; copy its retained report aside if intentionally regenerating.

[Preregistration](preregister.json) defines all cases before runtime tuning.
Each split has 28 palette/layout bases and 84 ordinary derivatives, plus seven
ambiguity/error negatives and two structural challenges. Four paired layouts
cover orientation, background, grid and scale direction; these factors are not
fully factorial. Matrix seeds are shared across palettes within layout and differ
between partitions. Derivatives and shared numeric patterns are correlated.

All ordinary attempts remain in the denominator, including failures. Clean,
JPEG quality 85 and blur sigma 0.65 use the same source figure per base. Generator
v1 records 256-entry LUTs, anchors, RGB interpolation and hashes. Matplotlib
generates viridis/magma/cividis LUTs only; the recogniser receives no palette
metadata. Source installation does not require Matplotlib.

## Evidence

- [Before development](before-development.json): original runtime, unchanged numeric thresholds.
- [Development](development.json): updated detector and retained component diagnostics.
- [Frozen validation](validation.json): runtime/evaluator frozen before recognition.
- [31-image corrections audit](corrections-audit.json): every historical mode-1 numeric flag failure.
- [Inspected corrections regression](corrections-regression.json): original validation is now development evidence.

Manual geometry is a separate conditional experiment. Selective automation
receives counts and scale endpoints only, chooses/abstains using the deployed
runtime, and is scored against truth afterwards. The reports separate heatmap,
bar, association, numeric QC, complete acceptance and maximum/mean numeric
errors. They retain all attempts and LUTs.

The initial corpus hash was recorded before tuning. Formatting changed Python
whitespace afterwards; corpus-initial.sha256 retains that record, corpus.sha256
records the final formatted generator, and before/after per-image hashes verify
identical development inputs. The final runtime/evaluator policy freeze precedes
validation. No validation-driven tuning was performed. Source hashes accompany
both development reports.

The asymmetric gradient is self-approaching in colour space and can be ambiguous
under the unchanged tolerance. Repeated colours, plateau, categorical/competing
bars, off-bar colours, severe quantisation and clipped bars are negatives.
A broad dark band is a positive structural challenge; a touching black border
remains review-required. See [M14 report](../../docs/progress/M14.md).

This is synthetic validation, not publication-image generalisation or a claim
that high confidence verifies biology. Automatic counts, OCR scales, identifier
export and M15 orientation proposals are unchanged.

The original validation freeze is retained as policy-freeze-validation.sha256.
A subsequent Optional-value narrowing repair for minimum-NumPy mypy changed
only QC serialization typing. post-validation-typing.patch records the exact
change; reversing it reconstructs the original I/O source hash. Both worked QC
files remained byte-identical. policy-freeze.sha256 checks the final source;
validation source hashes in the report describe the original run.

Post-validation coverage diagnosis is documented in
[M14 investigation](../../docs/progress/M14-investigation.md), with per-case
[investigation evidence](investigation.json) and `investigate.py` reproduction.
These inspected validation cases must be treated as development data for future
changes. The investigation makes no runtime or threshold changes.

The subsequent [grid repair](repair/README.md) supersedes the historical runtime
freeze. policy-freeze.sha256 retains the exact pre-repair source hashes; it is
not a checksum manifest for the current decoder. Original evidence.sha256 still
checks the historical reports and manifests.
