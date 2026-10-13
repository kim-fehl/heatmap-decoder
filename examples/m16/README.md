# Publication crop: reviewed calibration and labels

This is the unchanged Symbiotic crop of Figure 4 from
[Duscher, Vroom and Foster, Scientific Reports 14, 2912 (2024)](https://www.nature.com/articles/s41598-024-53477-3).
The article image policy is CC BY 4.0, subject to third-party credit exceptions.
`manifest.json` records image/table hashes, the 33×8 grid, supplied regions,
displayed order, tick annotations and a second-pass audit. It is an inspected
**development example**, not a held-out full-publication benchmark.

## Current commands

```bash
uv sync --locked
uv run heatmap-decoder extract examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/config.json --out work/m16/manual
uv run heatmap-decoder labels examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/labels-default.json --out work/m16/labels
```

The label command uses the production RapidOCR defaults: native-size grayscale
and explicit equal slots. It writes all evidence and returns exit 4 if any slot
remains unresolved. Review every name, including those accepted by confidence.

For numeric tick OCR, install local Tesseract and English data as described in the
[usage guide](../../docs/usage.md), then:

```bash
uv sync --locked --extra ocr
uv run heatmap-decoder legend examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/config-numeric.json --out work/m16/legend
```

Legend review returns exit 6 while required evidence remains unresolved. Raw
strings, sign-normalization actions, conflicting alternatives and budgets are
retained; parsed values alone do not certify a scale.

## Interpretation

![Supplied regions, complete label fields and tick evidence](../../docs/images/recognition.png)

The diagnostic distinguishes supplied regions, source-field/detected-text boxes,
confidence and review status. Its colours do not certify identifier correctness.

The reviewed −2..2 ticks do not cover the physical bar edges. Numeric extraction
therefore contains NaNs and `UNCALIBRATED` flags outside the reviewed span.
`--fail-on-qc` rejects QC completion. The approximately uniform visible ticks do
not prove a nonlinear numeric scale: piecewise calibration here demonstrates an
explicit reviewed workflow, while colours are always sampled densely.

The TSV's reconstruction recipe remains unknown. Source counts/TPM, log transform,
standardisation axis and population, variance convention and clipping are not
established; the paper describes NanoString. Comparisons are **reconstruction
agreement**, and eight columns are not assumed to be the complete population.
Transcription, including a visually uncertain Greek suffix, is evaluation-only.

[Current results and acceptance](../../docs/progress/M16.md) ·
[Fresh validation and all five publication workflows](../../benchmarks/m16/validation-rapidocr-v1/README.md)

## Historical configurations

| Configuration | Purpose |
|---|---|
| `config-linear-baseline.json` | Earlier linear extrapolation comparison; unknown physical edges are not certified |
| `labels.json`, `labels-retries.json` | Initial Tesseract slot trial and contrast/threshold retries |
| `labels-padding.json`, `config-padding.json` | Earlier crop-padding and tick-field alignment investigation |
| `labels-right-padding.json`, `config-numeric.json` | Wider Tesseract gene crops and audited numeric sign constraints; the numeric config remains current |
| `rapidocr_labels.py` | Standalone engine comparison that preceded production integration |

Historical label configurations explicitly select Tesseract. Use
`labels-default.json` for the current engine. See the [evidence index](../../benchmarks/m16/README.md)
for failures and measurements at each stage.
