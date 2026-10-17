# Observed palettes and numeric ambiguity (M14)

The decoder matches sampled cell colours to the figure's **observed colourbar**
in CIELAB (ΔE76). It requires neither a palette name nor a colour catalogue.
Position along that bar determines the numeric value and its direction; hue and
brightness need not be monotonic. Matplotlib is used only to generate named
benchmark LUTs, and is not an installation dependency.

The reproducible fixtures cover viridis, magma, cividis, grayscale,
blue–white–red, an explicit teal–black–red RGB gradient, and an asymmetric
self-approaching gradient. Both bar orientations, both scale directions,
light/dark backgrounds, grid/no-grid, clean/JPEG/blur are included.
No journal-wide palette is assumed. For a publication, use its observed bar.

The teal–black–red fixture has RGB anchors (0,180,180), (0,0,0), (210,0,0)
at normalised positions 0, 0.5, 1. The generator records RGB interpolation,
its rounded 256-entry LUT and hash, and generator version 1. Named-palette LUTs
and all custom anchors are recorded in each benchmark truth record.

## Interpreting quality evidence

The numerical ambiguity rule is unchanged: retain every bar sample within
best ΔE76 + ambiguity_colour_tolerance (default 1); flag the cell if their total
position span exceeds ambiguity_position_fraction (default 0.05). Ambiguous
matrix entries remain NaN, with their nearest candidate retained separately.
Colour-distance, flagged/missing-fraction and experimental export gates were
not relaxed.

The new colour_candidate_evidence JSON field in quality.csv retains:

- Every connected near-match component, with inclusive first/last sample indices.
- Normalised sample positions: first top/left pixel is 0, last bottom/right is 1.
- Minimum/maximum values mapped through the actual configured calibrated scale.
- Total position/value span and a diagnostic kind.

Broad contiguous near-match intervals describe low-gradient colour uncertainty
or saturation. Disconnected near-matches may describe repeated colours or
self-approaching branches; quantisation/noise can also split a local interval.
A component count alone cannot distinguish these causes. Local components below
the existing span limit remain local; materially broad spans remain ambiguous
regardless of how many components they contain.

These are **engineering tolerance sets, not statistical confidence intervals**.
A low ΔE does not bound numeric error. The benchmark reports maximum accepted
error as well as MAE, finite coverage and QC fractions against plotted truth.

No RGB colour is missing by default. Black and white are valid palette colours.
Structural masking now requires thin width and neighbouring contrast rather than
excluding every broad dark band. Thin complete perimeter strokes can still be
trimmed; a whole dark sample column cannot be eroded as a presumed border.

## Practical limits

A black centre against a dark background or a white centre against a light
background can split foreground components. Smooth, changing core evidence now
joins supported low-contrast gaps. Cropped fragments are suppressed only when a
closed full core continues smoothly through their internal endpoints. Rejected
fragments remain in bar-search evidence; genuinely competing bars still block
selection. Low-contrast endpoints, touching black borders and uncertain grid
extents can still require manual ROI review.

A long saturated plateau or repeated-colour palette cannot recover unique values.
Heavy quantisation, compression and off-bar cell colours can produce uncertainty
or wrong nearest candidates and must retain QC. Endpoint matches may represent
clipped limits; colours cannot recover values beyond those limits. Displayed
transformed/z-scored values do not recover raw RNA-seq counts or original
pre-transformation measurements. Automatic nonlinear scale selection remains
unsupported.

See [M14 results](progress/M14.md), the
[conditional palette benchmark](../benchmarks/m14/README.md), and
[worked numeric examples](../examples/m14/README.md).
