# eLife 87930 — author response image 15B

Urbanska, Ge et al., [De novo identification of universal cell mechanics gene
signatures](https://elifesciences.org/articles/87930/figures#sa2fig15), eLife
12:RP87930, doi:10.7554/eLife.87930.3. The original uploaded 1280 × 766 JPEG is
preserved unchanged, with its SHA-256 in `manifest.json`. Figure copyright remains
with the authors (CC BY 4.0). This is **author response image 15B**, not main Figure
15. The caption identifies young mice and GXD RNASeq Heatmap, accessed 2 October
2024.

The five reviewed genes are CAV1, FHL2, IGFBP7, TAGLN, THBS1. The inspected matrix
extent is `[48,194,1157,391]`. A **160-column hypothesis**, approximately 6.93
pixels per column, fits the observed boundaries substantially better than nearby
counts. `count_evidence.json` retains measured transitions, competing counts and
residuals. This is an image-derived estimate, not a sample count verified in the
article or an automatically accepted detector result. Only 20 strong transitions
are measured across the five rows; harmonics and unresolved identities remain.
[GXD documentation](https://www.informatics.jax.org/userhelp/EXPRESSION_help.shtml)
explains that heatmap columns represent **bioreplicate sets**, which may aggregate
biological samples. Do not equate 160 columns with 160 individual mice.

The tissue legend has **42 entries arranged in four printed columns**, describing
**one column annotation track**. All swatch boxes and visible labels are reviewed
inputs. They do not require reading order to match the strip's left-to-right
order. Seven pairs of tissues use indistinguishable or almost indistinguishable
swatches at the default ΔE margin: arcuate nucleus/ovary, cerebellum/secondary
oocyte, cerebral cortex/skin, cerebral hemisphere/small intestine,
colon/spermatid, corpus striatum/spinal cord, crypt of lieberkuhn/spleen.

The explicit `indistinguishable_swatches: "unresolved"` policy keeps these
candidates instead of rejecting the entire request. **79/160** positions match a
single category with the default color thresholds; 81 remain unresolved due to
color ambiguity, mixed pixels or poor matches. `column_groups.tsv` preserves
candidates and null categories. No complete generated sample-name file is supplied.

The scale is **TPM ranges**, with unequal numeric intervals, a gray below-cutoff
span, and an open-ended high range. `ranges-request.json` supplies four broad
printed brackets: below cutoff `[0,0.5)`, low `[0.5,11)`, medium `[11,1001)`, high
`[1001,∞)`. These are **display-bracket conventions**. GXD describes low/medium
using rounded labels 0.5–10 / 11–1000 and high >1000. The raster cannot establish
sub-TPM threshold precision. The intermediate printed labels are not treated as
equally spaced numeric ticks. The final `4000+` label is not a finite maximum.

`extract-ranges` reuses the decoder's observed colorbar matching and cell QC.
All near-color candidate positions must belong to the same supplied bracket.
Repeated colors within one bracket can identify that range without identifying
a point value. Cross-bracket and poor matches stay unresolved. The current run
recovers ranges for **784/800** cells; 11 are range-ambiguous and five have poor
color matches. `decoded_ranges.tsv` is image-derived output, **not ground truth**.
It supplies no exact TPM values, z-scores or inferred values above a finite cap.

```sh
uv run heatmap-decoder extract-ranges examples/real_world/elife-87930-response15b/figure.jpg \
  --request examples/real_world/elife-87930-response15b/ranges-request.json --out artifacts/tpm-ranges
uv run heatmap-decoder group-labels examples/real_world/elife-87930-response15b/figure.jpg \
  --request examples/real_world/elife-87930-response15b/groups-request.json --out artifacts/tpm-groups
```

Both commands intentionally return exit code **4** while unresolved cells or
categories remain. Outputs are written for review. Legend detection, legend text
OCR, exact original sample identities and exact TPM reconstruction are not
established by this example. These are inspected development cases, not held-out
automation qualification.
