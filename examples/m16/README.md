# M16 publication crop trial

The supplied JPEG is the **Symbiotic crop** of Figure 4, DOI
[10.1038/s41598-024-53477-3](https://www.nature.com/articles/s41598-024-53477-3).
Credit: Duscher, Vroom and Foster, Scientific Reports 14, 2912 (2024).
Article image policy is CC BY 4.0, subject to third-party credit exceptions.
The unchanged repository JPEG is evaluation/development evidence, not a full
publication automation benchmark. `manifest.json` records its hashes, image
coordinates, displayed order, visual second-pass audit and unresolved assumptions.

Install the optional OCR wrapper and local Tesseract with English traineddata.
English data does not establish Greek identifiers. No database or TSV vocabulary
enters recognition. The optional retries are configured in `labels-retries.json`;
they preserve conflicts and do not enable automatic identifier export.

```bash
uv sync --locked --extra ocr --group benchmark
uv run heatmap-decoder extract examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/config.json --out work/m16/manual
uv run heatmap-decoder legend examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/config.json --out work/m16/legend
uv run heatmap-decoder labels examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/labels.json --out work/m16/labels
uv run python benchmarks/m16/real_trial.py --out work/m16/trial
```

Legend exits 6 until reviewed; label evidence is not a verified identity table.
Manual numeric output is a **partial reconstruction**, with NaNs and explicit
`UNCALIBRATED` flags outside the reviewed tick span. Do not present it as a complete
matrix. `--fail-on-qc` makes the extraction command reject QC completion.
The physical bar extends beyond -2 and +2. No physical edge values were supplied;
we do not relabel those edges or extrapolate to fill the missing values.

The five visible ticks are approximately equally spaced. Selecting piecewise
calibration demonstrates its explicit workflow; this image does not prove a
nonlinear numeric scale. Its observed colour curve is sampled densely in either
model. `config-linear-baseline.json` preserves the prior extrapolating linear
workflow as baseline evidence, not the recommended interpretation of unknown edges.

The supplied TSV lacks its reconstruction recipe. Source TPM, log transform,
standardisation axis/population, variance convention and clipping are unresolved.
The article describes NanoString measurements; a TPM provenance for this TSV is
not established. Eight displayed columns must not be assumed to be the complete
standardisation population. Numerical comparisons are **reconstruction agreement**.
The image transcription has a visually uncertain Greek suffix at row 17 and is
used only for evaluation. It never corrects production OCR spelling.

See [results and remaining limits](../../docs/progress/M16.md).


The padded follow-up configurations are `labels-padding.json` and
`config-padding.json`. They preserve equal slot ownership, add 1–2 source pixels
and recorded white margins, and expose all five aligned tick fields in
`legend_fields_enlarged.png`. Use the same commands above with the padded config.
[Measured results](../../benchmarks/m16/ocr-padding/README.md) retain raw conflicts
and unresolved Greek/sign evidence.


`config-numeric.json` adds constrained numeric tick OCR and a bounded raw comparison,
with audited dash-to-minus normalization. `labels-right-padding.json` adds four
right-only gene crop pixels. All five canonical tick proposals are exact; the
Galectin suffix remains unresolved despite wider crops. See
[the actual trial](../../benchmarks/m16/numeric-signs/README.md).
## Default label OCR

RapidOCR is now installed with the main package and is the default gene/sample
engine. Run the normal command with native-size input and the existing equal slots:

```bash
uv run heatmap-decoder labels \
  examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/labels-default.json --out work/m16/default-labels
```

The [normal CLI trial](../../benchmarks/m16/rapidocr-default/README.md) retains raw
readings, model hashes, score semantics and geometry. Set `labels.ocr.engine` to
`"tesseract"` for the legacy engine. Explicit enlargement remains respected;
omitting it gives RapidOCR native-size input. Colour-bar tick OCR still uses
Tesseract with numerical constraints.

## Earlier independent OCR comparison

The earlier `labels.json`, `labels-retries.json`, `labels-padding.json` and
`labels-right-padding.json` configurations now explicitly name Tesseract to
preserve their historical measurements. Use `labels-default.json` for the new
RapidOCR defaults.

[`rapidocr_labels.py`](rapidocr_labels.py) generates review candidates from the
same explicit equal-row/column fields using RapidOCR. The
[reproducible comparison](../../benchmarks/m16/rapidocr/README.md) recovers
`EsGalectin1` and improves exact genes from 9/33 to 26/33 while retaining 8/8 exact
samples. Raw scores are engine-specific; this experiment does not automatically
accept gene names or change the production OCR default.
