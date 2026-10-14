# Grid coordinate tolerance and mild blur

A **1–1.5 pixel boundary tolerance** is reasonable for evaluating these figures,
provided that cell counts and group membership remain exact and numerical recovery
is checked separately. `benchmarks/geometry_metrics.py` now reports maximum
boundary error for each axis, exact equality, and the 1.5-pixel result. It never
forgives a missing/extra row, a different group split, or reordered cells.

The original M19 evaluator uses exact `AxisSlot` equality. This is overly precise
as a practical geometry metric, but it did **not** cause the published failures:
no accepted M19 geometry/full-OCR/explicit-OCR result had accurate numbers and
failed solely on exact intervals. Production rejected the unsuccessful candidates
before that comparison ran. The archived reports retain their original scoring.

The new [development trace](../benchmarks/ocr-review/grid-tolerance.json) examines
four layouts and paired blur radii, with reviewed scale and no OCR:

| Gaussian radius | Proposal with exact counts/groups, edges within 1.5 px | Production complete numeric result |
|---|---:|---:|
| 0 | 4/4 | 4/4 |
| 0.3 | 4/4 | 2/4 |
| 0.6 | 4/4 | 0/4 |
| 1.0 | 2/4 | 0/4 |

At 0.6, all four correctly grouped proposals are within **one pixel** on every
cell edge. Thus “detection struggles” describes overly conservative acceptance
more than an inability to find the grid on these particular cases.

The lattice fitter already permits **1.5 px** residuals. Other checks are stricter:

- Block acceptance rejects a transition wider than **one pixel**, even when its
  measured centre fits the lattice and uncertainty is 1.5 pixels. A blurred step
  can span three pixels without shifting its centre by three pixels.
- Separator acceptance rejects the inner uniform plateau of a blurred gap when
  its edge contrast is too weak. Here the resulting endpoint shifts are only one
  pixel; the check does not currently distinguish that bounded error from a more
  serious misplaced separator.
- Outer-bound checks combine measured offset with uncertainty. A one-pixel ROI
  shift plus a half-pixel raster bound can fail a one-pixel closure threshold.

The trace includes isolated width and halo-bypass experiments. Raising the width
limit alone recovers no complete cases at 0.6; disabling both checks recovers one
of four. Other cases still hit outer-bound or numeric QC checks. These experiments
are confined to the probe and do not change runtime policy.

Sampling was also tested with a one-pixel ROI translation and up to 1.5-pixel
internal segment-edge shifts. Clean figures stay complete in 4/4 cases; at 0.6,
3/4 stay numerically complete, matching extraction with supplied true geometry.
The remaining case already has a flagged/NaN cell with true geometry. Coordinate
tolerance must not conceal that numeric failure.

The detector should judge **edge location and its measured uncertainty**, rather
than equating visible transition width with coordinate error. Missing-cell bands,
wrong counts, palette-coloured annotations and independent panels still need
separate checks; a coordinate tolerance cannot resolve those interpretations.
These four inspected development bases establish the diagnosis, not a population
success rate or a new automatic-export qualification.

Reproduce with `uv run python benchmarks/ocr-review/grid_tolerance.py`. The probe
uses the production detector, retains proposal/rejection evidence, and evaluates
truth only after detection. Frozen M17–M19 reports remain unchanged.
