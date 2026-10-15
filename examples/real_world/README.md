# Published heatmaps and source tables

These materials were supplied by the user and prepared on `work`. Original JPEGs
and XLSX files are preserved byte-for-byte; manifests record hashes and attribution.

| Folder | Identified panel | Prepared material | Ground-truth status |
|---|---|---|---|
| [`elife-78496-fig5e`](elife-78496-fig5e/) | Li et al., Figure **5E**, skeletal system development | 49 × 14 `truth.tsv`, displayed row/column mappings, configs, decoded estimates, reviewed labels | Author-supplied plotted z-scores; all 49 genes joined to the workbook in the **image's order** |
| [`elife-78496-fig5-supp3a`](elife-78496-fig5-supp3a/) | Li et al., Figure **5—figure supplement 3A**, visible cluster 2/3 region | Original crop, 14 generated column names, two-track categorical evidence | Gene matrix and clustered row order missing; no numeric truth |
| [`elife-97830-fig4e`](elife-97830-fig4e/) | Bæk/Muk et al., Figure **4E**, DEGs–Immune Response | Original crop, 25 reviewed row names, 26 generated column names, configs, decoded estimates | Blood workbook supplied; 25 × 39 source export verified. Image shows 26 samples; selection/order and plotted transformation unresolved |
| [`elife-17290-fig2`](elife-17290-fig2/) | DeJesus et al., Figure 2 source data 2 | All **598** source records as TSV, preserving guide sequence, gene, source row, log2 FC signs | Figure 2G screenshot supplied; 20 × 3 truth matched to exact guide records, reviewed labels, extraction and OCR tests |

The first attached image's filename says `fig5`, but the crop is specifically
**5E**, not all of Figure 5. The second image has a three-cluster legend and shows
the cluster 2/3 region of supplement 3A; it must not be matched to the main Figure
5A's four-cluster analysis. Its gene rows are unlabeled and compressed, so counting
raster scanlines does not establish gene counts. The third image visibly carries
panel **E** and the immune-response title. The CRISPR workbook belongs to a
separate article and contains a single `high versus low` worksheet; its first
20 records match the newly supplied Figure 2G screenshot in displayed order.

Article/caption sources: [Li et al., eLife 11:e78496 (2022)](https://doi.org/10.7554/eLife.78496),
[Li figures](https://journal-symfony7.elifesciences.org/articles/78496/figures),
[Bæk/Muk et al., eLife 13:RP97830](https://doi.org/10.7554/eLife.97830.3),
[Bæk/Muk figures](https://journal-symfony7.elifesciences.org/articles/97830/figures),
[DeJesus et al., eLife 5:e17290 (2016)](https://doi.org/10.7554/eLife.17290).
These articles use CC BY; preserve author and article attribution when reusing
figures/tables. Figure crop identity is based on the visible content plus captions;
no full-resolution uncropped image was supplied for the second or third crop.

## Values and ordering

For 5E, the workbook's `Figure 5E` sheet already supplies the plotted z-scores.
**No additional standardization, averaging, clipping or guide aggregation is
performed.** `row_mapping.tsv` records the original worksheet row for each
visually transcribed gene. Source columns are `C:F`, `H:K`, `M:O`, `Q:S`; empty
spacers are excluded. The groups have 4/4/3/3 columns. `column_mapping.tsv` preserves
the left-to-right source order and genotype/diet metadata.

Names such as `BMAd-Pnpla2+/+__Ad libitum__01` describe group and display position.
The workbook and crop do not expose original sample IDs; the suffix does not
claim biological replicate identity. `truth.tsv` always keeps the original
source precision and values, including values outside visible tick support.
`decoded_estimates.tsv` is a separate image-derived approximation; `NaN` entries
mean ambiguous or unsupported calibration cells.

## Reproduce tables and tests

From the repository root:

```bash
uv sync --locked
uv run --with openpyxl python examples/real_world/build_tables.py
uv run python examples/real_world/evaluate.py
uv run --group dev pytest tests/test_real_world.py tests/test_group_labels.py -q
```

`build_tables.py` rebuilds the source truth, CRISPR source export, blood-expression
source export, and uploaded-workbook inventory (`workbooks.json`).
`evaluate.py` writes `evaluation.json` and the three `decoded_estimates.tsv` files;
it also measures actual RapidOCR row-label accuracy. Tests independently read
cells from the XLSX XML, verify hashes/order, compare decoded colors to the 5E
source values, and check categorical naming on both real images.

For a CLI extraction, which also writes per-cell QC and diagnostic images:

```bash
uv run heatmap-decoder extract examples/real_world/elife-78496-fig5e/figure.jpg \
  --config examples/real_world/elife-78496-fig5e/config.json \
  --out work/real-world/fig5e
```

`config.json` uses manually reviewed visible tick-label centers and piecewise
linear interpolation **only within those ticks**. Physical bar edges extend
beyond them. `config-linear.json` is an explicit affine extrapolation comparison,
not evidence that edge values were independently supplied. All regions use
half-open coordinates in the original attached JPEG. These inspected fixtures
are development examples, not held-out benchmark evidence.

Current measurements are in [`evaluation.json`](evaluation.json): 5E tick-only
extraction gives **618/686 finite cells, RMSE 0.052**, with 13 ambiguous and 55
uncalibrated cells. Affine extrapolation gives **673/686 finite cells, RMSE 0.052**.
OCR is exactly correct for **44/49** 5E genes and **25/25** immune-response genes.
Three wrong 5E names were accepted by OCR confidence; two were unresolved. Use
the reviewed row files for these examples. `--fail-on-qc` intentionally rejects
the numeric run while flags remain.

## Categorical group labels and complex names

The new `group-labels` command accepts reviewed legend text/swatch boxes, a
categorical strip box for each track, and axis extent/count (optional nonuniform
segments). It matches inset strip pixels to the swatches in CIELAB, rejects
mixed/unknown/indistinguishable colors, and generates unique names from the
category combination and ordinal within that combination. It preserves spaces,
underscores, Unicode and genotype punctuation.

```bash
uv run heatmap-decoder group-labels examples/real_world/elife-97830-fig4e/figure.jpg \
  --request examples/real_world/elife-97830-fig4e/groups-request.json \
  --out work/real-world/immune-groups
```

This produces `High_SE__01` … `High_SE__13`, then `Low_SE__01` … `Low_SE__13`.
For the supplement crop, use its `groups-request.json`: it combines AL/CR and
Pnpla2+/+/Pnpla2-/- in the correct 4/4/3/3 order. `groups.json` retains sampled
boxes and category-match fractions; `groups.tsv` separates category metadata
from the generated name. Feed the resulting `columns.txt` to the existing
`labels.columns` file source. Names identify positions, not original sample IDs.

Legend text/regions are supplied and reviewed; automatic legend discovery and
OCR of superscript genotype text are not implemented by this command. An
unresolved assignment returns exit 4 and writes evidence without a complete name
file. Existing outputs require `--overwrite`.

## Remaining source material

Figure 4 source data is now supplied as `elife-97830-fig4-data1-v1.xlsx`.
All 25 displayed genes match uniquely in `DEGSEhighvsLow`; their 39 source
expression columns are preserved in `source_expression.tsv`, with original sample
IDs, group labels and source row/column mappings. There are **13 HIGH_SE and 14
LOW_SE** records, whereas the image shows 13 per group. The omitted Low_SE sample,
within-group display order and plotted transformation remain unverified. This
export is source expression, not plotted z-score truth; no `truth.tsv` is claimed.
The Figure 2 workbook for this article is liver data, not the blood-panel source;
it is inventoried and checked for integrity rather than matched to Figure 4.

Figure 2G now has a byte-preserved user screenshot and **20 × 3** guide-level
truth, using source rows 2–21 with no aggregation or transformation. MAP1LC3B and
MAP1LC3B2 share one guide and identical values in the source; both displayed names
are preserved. The 0–6 legend uses manually estimated bar endpoints, not tick
centers (the two endpoint labels sit outside the bar). All 60 cells are finite;
RMSE is **0.176 log2 FC**, maximum absolute error **0.325**, correlation **0.9994**.
`config.json` and `config-linear.json` use the same two-anchor linear fit; there
are no interior ticks to justify a separate piecewise calibration.

Panel G OCR accepts only **9/20** names at native size, despite **18/20** raw
strings being exact. Eleven slots remain unresolved; `WIPI2` becomes `IPI2` and
`SBDS` becomes `BDS`. No accepted Panel G name is wrong in this run. Upscaling by
2 or 3 accepts **10/20**, with raw accuracy unchanged. This demonstrates a problem
with this label layout, but does not isolate font compression as its cause.
Angled column labels are manually reviewed; automatic column OCR is not validated.
Use `rows.txt` and `columns.txt` for numeric extraction.

Run the actual publication OCR regression tests explicitly:

```bash
HEATMAP_DECODER_OCR_TESTS=1 uv run pytest tests/test_real_world.py -q
```

The numeric/source tests run without OCR and independently read XLSX XML. The
real-engine tests enforce minimum observed accuracy while allowing improvements;
`evaluate.py` records raw/accepted accuracy, accepted wrong names, unresolved slots,
and the panel G upscale comparison. These remain development fixtures, not a
held-out benchmark. See [`PROBLEMS.md`](PROBLEMS.md) for known limitations.

For supplement 3A, its clustered gene-by-sample matrix and gene order are still
missing; the main Figure 5 workbook has sheets 5B/5D/5E/5F only.

The earlier Scientific Reports image/TSV pair remains under this directory; its
existing calibration and provenance notes are in [`../m16/README.md`](../m16/README.md).
`research/` retains the earlier article shortlist and download script. Publisher
access in the managed workspace is still blocked; the supplied attachments and GitHub commits provide
the workbooks used here.
