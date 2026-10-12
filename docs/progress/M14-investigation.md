# M14 acceptance investigation

This is post-validation diagnosis of the retained 84 ordinary validation attempts.
The runtime and acceptance thresholds are unchanged. These inspected cases are now
development evidence for any subsequent fix, not a fresh holdout.

## What the 33/84 result measures

It is complete automatic numeric acceptance with supplied counts and scale endpoints.
It combines geometry verification with cell QC. It does not mean only 33 images
contain recoverable numeric data. All 84 attempts were rerun with identical image
hashes and reproduced their original complete decision.

| First blocking stage (exclusive) | Figures |
| --- | ---: |
| Accepted | 33 |
| Heatmap verification | 38 |
| Colourbar verification, after heatmap passes | 3 |
| Numeric QC, after geometry passes | 10 |

Six bars fail in total; three overlap heatmap failures. Association passes all 84.
The accepted counts are clean 14/28, JPEG 12/28, mild blur 7/28.

With the annotated geometry supplied, **60/84** have fully finite output and pass
numeric QC, including **27 currently rejected figures**. Their worst cell error
is 2.18% of the scale range. This is a conditional manual result, not achieved
automatic coverage or proof that those bounds can always be inferred.

## Geometry failure mechanisms

Heatmap reasons overlap: full bounds 17, broad row peaks 17, grid interior
agreement 10, row phase alignment 10. Every rejected heatmap has one candidate;
14 candidates exactly match truth, and the other 24 differ by at most one pixel
per edge. Candidate localisation is substantially better than its acceptance rate.

1. **Final gridlines are mistaken for an outer border.** All seven clean layout-1
   figures have detected bounds `(80,80,343,463)` versus `(80,80,344,464)`.
   `refine_box` trims the uniform last black row/column, and
   `refine_measured_box` deliberately prevents outward correction across an
   adjacent black stroke. Extent verification then checks a different boundary.
   For viridis, right/bottom fitted offsets are about 1.17 pixels with 0.55-pixel
   uncertainty, exceeding the shared 1.60-pixel tolerance. Re-running the same
   grid verifier at the annotated bounds passes all seven; six have unflagged
   numeric reconstructions, while asymmetric still requires colour review.
   A repair must distinguish a final regular gridline from an external frame.

2. **Blurred stroke width is treated as location uncertainty.** For layout-1
   viridis with blur sigma 0.65, the candidate bounds are exact, equal-grid score
   is 1.0, and every row transition passes phase alignment. Nevertheless the
   four-pixel-wide peaks fail because half their width (2 pixels) exceeds 10%
   of the 16-pixel row pitch (1.6 pixels). Peak centres are approximately half a
   pixel before the expected boundaries. A fitted stroke/transition model could
   measure localisation independently of width; simply ignoring wide peaks
   would also hide genuinely blurred or irregular boundaries.

3. **JPEG ringing contaminates grid evidence.** Layout-1 viridis JPEG has exact
   bounds and a piecewise-constant score of 0.956, but equal-grid score 0.841
   falls below 0.90. This score magnifies interior residuals by four relative to
   between-cell colour changes. The row observer also reports extra transitions
   around real gridlines (e.g. 222 and 224), dropping alignment to 0.848 versus
   required 0.95. Manual extraction has zero flags and maximum error 1.46%.
   Stroke-aware grouping and compression-aware residual estimation are better
   targets than a blanket lower score threshold.

4. **Dark cells reduce usable edge support.** Teal-black-red layout-3 clean has
   exact bounds, perfectly constant interiors and aligned transitions. Its right
   edge support is 0.754, below 0.80: near-black cells resemble the background and
   do not meet the per-sample contrast threshold. All visible crossings agree
   at zero offset with 0.5-pixel uncertainty. Distributed supporting segments
   and corroborating lattice evidence may help, but incomplete panels must
   remain distinguishable from weakly visible complete boundaries.

5. **Dark magma endpoints really are cropped.** All six dark-background magma
   bars fail. Horizontal candidates start 10 pixels late; vertical candidates
   start 12–13 pixels late. The gradient merges into the dark background before
   foreground segmentation sees it. These rejections prevent calibration against
   a shortened bar. Recovering extent needs independent endpoint evidence or
   review; accepting the current candidate would shift the value mapping.

## Numeric rejections

The ten geometry-approved numeric failures are seven asymmetric cases, two
blurred grayscale cases, and one blurred blue-white-red case. All are ambiguity
flags, with 1–12 affected cells each; there are no poor-match flags in these ten.
Even one unresolved cell prevents a complete automatic matrix.

The asymmetric gradient approaches itself in colour space. Blurred ordinary bars
also acquire mixed endpoint colours that resemble interior colours, creating
extra disconnected matches. The grayscale examples have candidate spans up to
about 21.9% of the numeric range. Low colour distance alone does not resolve this.
The earlier corrections audit already found a flagged nearest match wrong by
21.69% of range. Keep these unresolved values rather than silently filling them.

Across all 84 manual reconstructions, 24 retain flags: asymmetric 12,
teal-black-red 5, grayscale 4, blue-white-red 2, cividis 1. Thus fixing geometry
alone cannot produce 84 fully accepted matrices under the present observed-bar
matching method.

## Recommended work order

First repair the final-gridline/border distinction and model blurred/JPEG strokes
as strokes, with focused frame, clipped-panel and irregular-grid counterexamples.
Then improve weak-edge corroboration. Investigate endpoint contamination and dark
bar extent separately; those affect calibration validity. Keep numeric ambiguity
checks intact. Evaluate any revisions on this now-inspected set as development,
then freeze the policy before a new independent validation set.

## Reproduce

`uv run --group benchmark python benchmarks/m14/investigate.py`

The script refuses to overwrite its report. It verifies image hashes and original
complete decisions, and records all stage evidence, candidate coordinate errors,
and seven annotated-bound grid counterfactuals in
[the diagnostic report](../../benchmarks/m14/investigation.json).
The original validation report and frozen evidence are unchanged.

The subsequent [geometry repair](M14-repair.md) changes runtime decisions.
Reproducing this historical diagnosis requires the pre-repair M14 runtime;
reverse benchmarks/m14/repair/runtime.patch in an isolated copy. Its recorded
source hashes describe that historical run.
