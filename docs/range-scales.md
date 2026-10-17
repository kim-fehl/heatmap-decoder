# Expression ranges and independent gene categories

Use `extract-ranges` when the image supports **ranges** rather than precise point
values: unequal TPM brackets, a below-cutoff plateau, or an open upper range.
Continuous numeric heatmaps continue to use `extract` with their reviewed scale.

The request contains `position_config`, an ordinary extraction configuration
with endpoints **0 and 1**, no `scale`, and the reviewed grid and observed colorbar
box. These endpoints describe position only; they are never TPM values. Existing
sampling, segmented-grid, missing-color, alpha and label-source options apply.

Supply at least two ordered, nonoverlapping `bands` with `label`, `start`, `end`,
optional numeric `lower` and `upper`, and optional `lower_inclusive` /
`upper_inclusive` (defaults true / false). A null bound is open-ended. Numeric
bounds are reviewed inputs, never inferred from the colors or label text.
`quantity` records the expression units and any display convention.

Band positions use the first sampled axial pixel as 0 and the last as 1.
For a horizontal bar `[left,top,right,bottom]`, convert an original x coordinate
using `(x-left)/(right-left-1)`; vertical bars use y analogously. Spans are
half-open except that a band ending at 1 includes the last sampled pixel. Gaps
between configured bands remain unsupported. The [TPM fixture request](../examples/real_world/elife-87930-response15b/ranges-request.json)
is a complete runnable example.

```sh
uv run heatmap-decoder extract-ranges examples/real_world/elife-87930-response15b/figure.jpg \
  --request examples/real_world/elife-87930-response15b/ranges-request.json \
  --out work/tpm-ranges
```

The Python equivalent is `extract_ranges(image, RangeExtractionRequest(...))`,
exported by `heatmap_decoder`. File-label contents can be supplied through the
same `file_labels` mapping used by `extract`; the CLI loads paths relative to the
request file and protects those inputs from overwrite.

Outputs are `ranges.json` and a cell-indexed `ranges.tsv`, including bounds,
inclusivity and status. No point-expression matrix is emitted. All near-color
candidate positions from the existing matcher must lie in a single supplied
band. A plateau or disconnected near-match within that band can identify its
range even when a point estimate is ambiguous. Cross-band matches, unsupported
spans, poor color matches and missing cells remain unresolved, with candidate
range labels retained. Range bounds are **not statistical confidence intervals**.
Exit code 4 indicates unresolved output requiring review; 2 indicates invalid
input/output paths or configuration. Both output files require `--overwrite`
if they already exist.

## Annotation axis and entity

Categorical legend layout does not determine its annotation axis. A legend can
occupy four printed columns yet describe a single sample strip. Conversely,
separate sample and gene legends must have separate requests:

- `axis: "columns", entity_kind: "sample"` for the `Condition` strip.
- `axis: "rows", entity_kind: "gene"` for the `Category` strip.

`entity_kind` defaults to `sample` for columns and `unspecified` for rows when
omitted. This preserves existing column requests without assuming that all row
axes represent genes. It can be explicitly selected on either axis, including
transposed figures. Outputs include `entity_kind`, `entity_name`,
`entity_status`, and `visible_entity_names_recovered`. Sample-name compatibility
fields are null / not applicable for gene requests. Reviewed visible gene names
remain display names even when gene-category metadata is also present.

The existing default rejects indistinguishable legend swatches. Set
`indistinguishable_swatches: "unresolved"` to retain ambiguous candidate labels
and continue decoding distinguishable categories. This does not relax the color
margin, select a label by legend order, or recover complete identities from
reused colors. `candidate_categories` records categories with sufficient
near-color pixel support, including competing colors. Unresolved requests do
not emit a complete axis name file.

## Publication results

[TPM example](../examples/real_world/elife-87930-response15b/README.md): five genes,
160 image-derived column positions, 784/800 cells assigned broad TPM brackets,
79/160 unique tissue assignments. Exact TPM values, exact biological sample count
and original identities remain unresolved.

[Gene-category example](../examples/real_world/lsa-e202302332-fig7a/README.md):
66 genes, 30 sample columns, all memberships resolved on both independent axes.
Numeric output retains the displayed scale without asserting an unverified
normalization or source matrix.

Reproduce both with `uv run python scripts/check_range_examples.py`. Results
are in [range_examples_evaluation.json](../examples/real_world/range_examples_evaluation.json),
with separate annotation references and image-derived tables. These inspected
cases do not establish held-out automatic count or legend-detection performance.
