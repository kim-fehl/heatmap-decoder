# Life Science Alliance e202302332 — Figure 7A

Vanhaver et al., [Immunosuppressive low-density neutrophils in the blood of cancer
patients display a mature phenotype](https://www.life-science-alliance.org/content/7/1/e202302332),
Life Science Alliance 7(1):e202302332 (2024), doi:10.26508/lsa.202302332, CC BY 4.0.
The [article's Figure 7 caption](https://pmc.ncbi.nlm.nih.gov/articles/PMC10628041/#F7)
identifies the heatmap. The publisher asset filename `F13.large.jpg` is **not the
figure number**. The uploaded 667 × 1280 crop is preserved byte-for-byte;
`manifest.json` records attribution and hash.

Two categorical legends apply to **different axes**:

| Visible title | Axis and entity | Categories |
|---|---|---|
| Condition | columns / samples | CD45high LDN, TAN, CD45low LDN, NDN |
| Category | rows / genes | Chemotaxis, Immune defense, Inflammation, Lung neutrophils, Maturity, Phagocytosis, Primary granules, Secondary granules, Tertiary granules, ROS biosynthesis, S or G2M phases |

The reviewed grid has **66 gene rows × 30 unnamed sample columns**. Eleven
nonuniform row segments exclude the horizontal white gaps: those gaps are not
missing genes or extra rows. `rows.txt` is a visual transcription; the output
uses these visible gene names, not category-plus-ordinal names as replacements.
The gene-category request explicitly uses `axis: "rows", entity_kind: "gene"`.
JSON and TSV report gene names as `entity_name`, with sample-identity fields null.

The column strip resolves nine CD45low LDN, nine CD45high LDN, nine NDN and three
TAN columns. They are interspersed in dendrogram order; they must not be regrouped
by category. Original sample identities and patient matching remain unresolved.
The supplied crop has broad JPEG transitions: the column request explicitly
uses ΔE ≤15 and a **0.65 matching-pixel fraction**, while the row request keeps
the default 0.8 fraction with ΔE ≤15. These are inspected development settings,
not changes to the decoder defaults. All 30 column and 66 row assignments match
the reviewed annotation references.

The colorbar has visible 4, 2, 0, −2, −4 labels and extends beyond them. The config
uses reviewed label centers with piecewise interpolation only inside their
support, never assigning the outer bar edges ±4. The exact plotted normalization
is unresolved, so the output quantity remains **displayed colorbar values**;
do not infer z-scores solely from the diverging colors. `decoded_estimates.tsv`
contains **1,939/1,980** finite color-derived estimates and normal QC NaNs. There
is no independently aligned author-supplied numeric truth matrix.

```sh
uv run heatmap-decoder group-labels examples/real_world/lsa-e202302332-fig7a/figure.jpg \
  --request examples/real_world/lsa-e202302332-fig7a/groups-request.json --out artifacts/lsa-columns
uv run heatmap-decoder group-labels examples/real_world/lsa-e202302332-fig7a/figure.jpg \
  --request examples/real_world/lsa-e202302332-fig7a/row-groups-request.json --out artifacts/lsa-genes
uv run heatmap-decoder extract examples/real_world/lsa-e202302332-fig7a/figure.jpg \
  --config examples/real_world/lsa-e202302332-fig7a/config.json --out artifacts/lsa-numeric
```

Both examples can be reproduced together with
`uv run python scripts/check_range_examples.py`. Generated estimates, annotation
references, count evidence and numeric ground truth have separate artifacts.
