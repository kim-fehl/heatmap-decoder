# M16 follow-up: padded, owned fields and aligned legend text

This is inspected development evidence on the same supplied publication crop.
It is not a fresh publication validation set. Historical M16 reports and freezes
are unchanged; the original numeric validation can be reproduced at commit
`378f870` using its frozen source/evaluator. Current runtime source has deliberately
changed OCR field geometry, so the historical hash check must not be weakened.

The trial uses `examples/m16/labels-padding.json` (one original source pixel of
padding and one white pixel before upscale) and `config-padding.json` (shared
text-band localisation, two original source pixels, two white pixels, and one
bounded retry with no white border). The reference TSV is not accessed by OCR,
field selection, numeric calibration or identifier correction.

```bash
uv run --extra ocr heatmap-decoder labels \
  examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/labels-padding.json --out work/m16/padded-labels
uv run --extra ocr heatmap-decoder legend \
  examples/real_world/10.1038__s41598-024-53477-3__fig4.jpg \
  --config examples/m16/config-padding.json --out work/m16/padded-legend
```

Label/legend commands retain their review-required exit codes. Numeric-only
extraction with `config-padding.json` is bit-identical, including NaNs, to the
previous 251/264-cell reconstruction. Neither the colourbar calibration, colour
sampling nor numerical QC thresholds changed.

| Capability | Previous reported trial | Padded/owned-field trial |
|---|---:|---:|
| Attempted gene/sample slots | 33 / 8 | 33 / 8 |
| Exact raw gene strings | 2/33 | 9/33 |
| Exact raw sample strings | 8/8 | 8/8 |
| Heuristic accepted genes / exact | 4 / 2 | 4 / 3 |
| Heuristic accepted samples / exact | 6 / 6 | 8 / 8 |
| Unresolved requested slots | 31/41 | 29/41 |
| Physical tick fields | 5 | 5, all drawn separately |
| Exact parsed tick proposals | 2/5 | 4/5 |
| Automatic OCR-scale / identifier exports | 0 | 0 |

New raw tick proposals are `-2`, `—1`, `0`, `1`, `2`. An observed em dash is
**not silently replaced with a minus**, so `—1` stays unparsed. Raw alternative
readings remain in `numeric_alternatives`, linked to original OCR records.
Conflicting nonempty readings require review even if only one parses numerically;
contradictory parsed numbers clear the value. A uniquely parsed retry can be shown
as a proposal but remains review-required. No engine confidence resolves identity
or sign conflicts.

Equal slot fields come from the supplied regular heatmap grid, not the size of an
OCR glyph box. Each slot crop owns its candidates; a fractional raster-edge overlap
cannot leak candidates into adjacent slots. Off-centre or oversized observations
remain unresolved in their originating slot. The original pixels, padded footprint,
white-border size, quarter-turn transform and raw token boxes remain recorded.
Rounded raster crops can differ by one pixel; the logical field pitch is constant.

Aligned legend fields measure a common perpendicular ink band, then separate axial
ink clusters with bounded glyph-sized gaps to keep an observed sign with its digits.
Horizontal fields share top/bottom coordinates; vertical fields share left/right
coordinates. This assumes ordinary aligned tick text, not equal tick spacing or a
known tick count. Missing/competing bands and merged/missing clusters retain explicit
unresolved evidence. The feature is opt-in and does not guess missing ticks/values.

Padding and retry calls share the existing pixel/call/deadline budgets, including
white pixels after upscale. Coordinate and sign tests cover both orientations,
all four bar sides, 0/±90° column transforms, overlapping/competing fields,
conflicting numbers, decimal locale and exhausted call budgets.

`owned-without-padding.json` is the controlled ownership-only run: 4/33 raw
gene strings are exact before padding, compared with the previous 2/33. Adding
one source and one white pixel yields 9/33. This separates the association fix
from the crop change.

`source-and-white-probe.json` retains the initial 0/1/2-pixel comparison.
`ink-band-probe.json` retains the signed-text crop investigation.
`labels.json` and `legend.json` retain complete engine/transform/raw evidence.
`summary.json` records outcomes and the bit-identical numerical comparison.
`source.json` records the trial runtime; `post-review-source.json` records the
subsequent locale-preservation and vertical-minus safeguards, covered by dedicated
focused tests.
