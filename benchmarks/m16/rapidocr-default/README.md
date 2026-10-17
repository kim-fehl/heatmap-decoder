# Default RapidOCR label extraction

RapidOCR is now the **default gene/sample OCR engine** in the normal extraction
pipeline, installed with the main package and bundled PP-OCRv4 models. Label
settings default to native-size grayscale input. Explicit enlargement and padding
remain respected; selecting `labels.ocr.engine="tesseract"` preserves the legacy
engine, with 3× enlargement unless overridden. Numeric colour-bar OCR remains
Tesseract with the existing character constraints and auditable sign normalization.

```bash
uv sync --locked
uv run heatmap-decoder labels \
  examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/labels-default.json --out work/m16/default-labels
```

The example does not select an engine or enlargement factor: it exercises their
new defaults through the normal `labels` command. It retains the same equal slots,
one-pixel source margins and four-pixel right margin as the standalone comparison,
with zero synthetic white padding. All 41 raw readings and source-field boxes
match the prior native-input experiment.
The command returns the existing **exit 4** because one gene slot remains
unresolved; all review artifacts are written before that exit.

| Result | Genes | Samples |
|---|---:|---:|
| Raw exact readings | **26/33** | **8/8** |
| Heuristically accepted | 32/33 | 8/8 |
| Accepted and exact | 26/33 | 8/8 |
| Unresolved | 1/33 | 0/8 |

`EsGalectin1` is recovered with confidence 89.4037 (the recognizer's 0.894037 score
multiplied by 100). Six wrong gene readings pass the unchanged confidence rule;
`IRAK42` is especially misleading at 97.07. Review **all** identifiers. These scores
are explicitly uncalibrated and engine-specific. RapidOCR can also lose internal
spaces in rotated sample names; raw observations are preserved without vocabulary
correction. Legacy Tesseract remains selectable for comparisons and other language
packs; RapidOCR currently supports the `eng` setting with the bundled model.
Automatic geometry decisions also change with the detector: the prior M10
synthetic inferred-count case now requires review under RapidOCR, while explicit
Tesseract still accepts its 20×6 grid. The acceptance thresholds are unchanged;
inadequate independent support produces abstention rather than a guessed count.
Selecting an engine through OCR settings alone does not request identifiers in
an otherwise numeric-only request.

The adapter uses a reusable local worker with one inference thread. Each call's
send/inference/receive runs under a timeout; on timeout the worker is terminated
and can be recreated for a later request. This also works from the library API
without importing or executing the caller's main script in a child process. Model
hashes, engine/runtime versions, geometry and score semantics remain in provenance.

Slot recognition has no detector and reports **source-field boxes**, explicitly
marked `geometry_kind=source_field`. Artificial white margins are removed from
these boxes before mapping to the image. Sparse OCR retains detected-text boxes;
quarter-turn normalization, original-coordinate mapping, fixed ownership, call/
pixel/deadline budgets, ambiguity handling and review/export gates remain in place.
Confidence heuristics are unchanged, including their known false acceptances.

`provenance.json` is the normal CLI evidence, `summary.json` evaluates it against
reference strings only after inference, and `source.json` pins the implementation.
`legend_provenance.json` retains the repeated numeric OCR check; `checks.json`
records verification. **627 tests and 18 subtests passed** with real OCR enabled.
Ruff, strict mypy with current NumPy and NumPy 1.26.4, source/wheel builds and the
isolated installed-wheel smoke test passed. The latter installs the main wheel
without `[ocr]`, runs outside the checkout and verifies all 41 RapidOCR fields,
model hashes, the recovered suffix and the expected unresolved-slot exit.
Numeric values are bit-identical to the preceding reviewed M16 trial, including
NaNs: **251/264 finite**. The five physical numeric tick proposals remain
**-2,-1,0,1,2** under Tesseract; extra localisation candidates/low-confidence
conflicts still prevent automatic scale acceptance. No colour or calibration
threshold changed.

Regression validation covers default dispatch, native sizing, raw suffix/text
preservation, sparse geometry, score validation, model provenance, a genuinely
stalled subprocess timeout, synthetic-border coordinate mapping, the normal
publication CLI, legacy engines, rotated fields and unchanged reviewed numeric
values. This is inspected-image development evidence, not a claim of universal OCR
accuracy. Earlier [standalone comparison](../rapidocr/README.md) and
[white-padding failures](../white-padding/README.md) remain available.
