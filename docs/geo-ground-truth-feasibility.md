# GEO reference feasibility: PLOS Pathogens 1012368

Checked 2026-10-15. Suitable for annotation ground truth now; exact numerical reconstruction remains unverified.

The [paper](https://journals.plos.org/plospathogens/article?id=10.1371/journal.ppat.1012368), Fig. 1C and caption, support these blocks (1-based columns):

| Columns | Disease annotation | Cell annotation |
| --- | --- | --- |
| 1–6 | Severe Covid-19 | LDG |
| 7–16 | Severe Covid-19 | PMN |
| 17–24 | Mild Covid-19 | PMN |
| 25–31 | HC | PMN |

The authors applied a neutrophil-purity filter (>0.65). The image uses row Z-scores; its caption specifies complete linkage and Spearman rank ordering.

[GSE272381](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE272381) lists 36 human samples, `GSE272381_merged-gene-counts.txt.gz`, `GSE272381_gene-annotation.tsv.gz`, and `GSE272381_RAW.tar` (per-sample count files). [GSE271808](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE271808) is mouse data.

## Evidence boundary

GEO metadata and file listings were available through indexed search results. Direct pages returned browser verification, and the web reader could not retrieve the supplement directory. The executor network policy excludes NCBI/PLOS downloads. No expression file was downloaded or numerically compared.

The deposited matrix is a candidate reference, not verified plotted values. Establish the retained sample subset, order, and normalization/log transformation before scoring numerical recovery. Do not assume count values equal plotted Z-scores or accession order equals figure order. S1A Fig. (purity) and S1B Fig. (broader heatmap) are next checks for sample identifiers. Expression-to-image matching could then provide an inferred mapping with uncertainty; author-provided plot input would be stronger evidence.

## Decoder representation

The [annotation fixture](../examples/real_world/publication_headers/plos-ppat-1012368/figure-1c.annotations.json) keeps synthetic column IDs, null visible sample labels, and null GEO IDs. Synthetic IDs are indexing aids, not inferred identities.

Use independent annotation tracks of column spans with optional track names. The figure has category labels but no visible track titles; inferred semantic roles are separate metadata. Disease and cell-fraction boundaries cross, so avoid a single parent/child grouping tree.

Keep an array of annotation tracks so additional rows need no schema migration. Score annotation boundaries/category labels separately from sample identity and numerical recovery. The subsequent [decoder implementation](../README.md) now recovers both tracks from this image and tests their spans against the reference; GEO numerical reconstruction remains unverified.
