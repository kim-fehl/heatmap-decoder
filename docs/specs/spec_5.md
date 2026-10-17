# heatmap-decoder — quarter-turn sample names, palette diversity and real-figure trials

## Reviewed starting point

Continue M13–M15 from branch `work`, commit `505f24e` (reviewed 2026-10-12). Read repository instructions, earlier specifications and `docs/corrections.md` before implementation. Do not infer stable release maturity from passing synthetic examples.

The correction commit materially improves the previous blockers:

- Inspected M12 regression, supplied counts/scale: 92/135 complete decisions versus 11/135 previously; all accepted selections pass recorded accuracy checks. Competing bars now abstain. This is regression evidence, not fresh validation.
- Fresh corrections validation: supplied counts/scale 33/72 (45.8%), inferred counts/supplied scale 3/72, supplied counts/OCR scale 16/72, fully inferred 2/72. No wrong complete decisions or applicable negative exports were observed on this finite synthetic set.
- Supplied-count heatmaps pass 72/72; bars pass 66/72. Remaining numeric failures include 23 excessive-colour-flag decisions and eight missing/ambiguous-cell decisions; overlapping reasons are not additive counts.
- Conditional legend localisation recovers 342/342 ticks; exact numeric recognition is 306/342 (89.5%), including ten recorded sign errors. OCR-derived scale and inferred dimensions still cannot use experimental export.
- Experimental exports now work for numeric-only, supplied counts and supplied/reviewed linear scale with accepted per-image decisions. Default behaviour remains review-oriented.

Local checks: 65 focused correction/reconstruction/QC tests passed, one skipped. Additional clean synthetic 20×20 manual-geometry probes detected bars and produced fully finite unflagged matrices for viridis, magma, cividis and a specified teal–black–red RGB gradient. MAE fractions of scale range were approximately 0.00093, 0.00090, 0.00091 and 0.000019 respectively. These small probes confirm basic palette compatibility, not practical coverage or rotated-text support.

## Goals and scope changes

1. Support sample names printed at exactly 0°, +90° or −90°, above or below the matrix. Genes remain rows and samples remain columns; rotating text must never rotate or transpose the matrix.
2. Validate observed-bar reconstruction across multiple palettes, especially viridis-like sequential colours and diverging teal–black–red with a dark centre. Fix palette-dependent detection/QC failures where evidence identifies them.
3. Make manual/reviewed real-publication trials practical and measure the same figures through selective automation without ground-truth injection.

This supersedes the horizontal-only restriction for **column/sample labels only**. Other label limitations stay: no arbitrary angles, perspective correction, multiline text, biological alias guessing or axis-semantic inference. No new ML, GUI, gene database/API integration or automatic nonlinear scale selection this phase. Row-label rotation can be a later extension after the column path works.

Do not invent a named journal-wide palette. The exact teal–black–red colours of a publication come from its observed colourbar or a supplied reference. Synthetic palette fixtures must name their actual RGB anchors, interpolation space and generator version.

## M13 — Explicit quarter-turn sample-name OCR

### Configuration and compatibility

Extend the existing OCR label source with an optional field such as `text_rotation_degrees`, accepting only integer `0`, `90`, `-90`; default 0 preserves existing configurations. The field describes the visual counter-clockwise text rotation from upright, in the original figure. Normalisation applies the inverse rotation to the crop. Reject unsupported values and bools with a clear message. Include a diagram/example of each sign in documentation rather than requiring users to guess.

Example addition to an existing label configuration:

```json
{
  "columns": {
    "source": "ocr",
    "bbox": [100, 20, 500, 80],
    "text_rotation_degrees": 90
  }
}
```

The box is still `[left, top, right, bottom]` in original-image pixels; the example illustrates schema only, not a universal crop height. A column-label region may be tall and narrow enough for long vertical names. File/manual labels do not need a text angle. Legend tick text stays upright under the current legend scope.

### Geometry and orchestration

Extend the crop transform to compose original crop origin, exact quarter turn and upscale. Use Pillow transpose operations for ±90° rather than interpolating a general rotation. Preserve the existing upscale/preprocessing policy and pixel caps, accounting for swapped dimensions.

Group recognised words/lines in the upright OCR coordinate frame. Then map candidate boxes and token geometry back to original coordinates and assign by original **x-coordinate** to column slots. Reusing horizontal line grouping after inverse-transforming tokens would incorrectly group vertical glyphs; make grouping frame explicit.

Transform all four box corners through the inverse map, retaining valid half-open extents. Keep raw OCR-frame and original-frame boxes, angle, transform, crop and engine run linked in provenance. Preserve raw character spelling, token membership, spaces, underscores, punctuation, case and confidence.

For slot OCR, slice each column band in original coordinates, normalise that patch, OCR the single upright line and map its evidence back to the same slot. For whole-region OCR, transformed order may reverse along the upright frame; sort the final labels by original column index, never OCR reading order. Missing/duplicate labels retain their existing slot behaviour.

### Acceptance

- Geometry tests cover both signs, non-square crops, nonzero origins, upscale, border-touching boxes and exact inverse/forward round trips. No lost pixels, slot reversal or off-by-one extents.
- OCR-enabled tests include 6–10 long sample strings, ±90°, above/below, several fonts and clean/JPEG/blur examples. Report exact text, assignment correctness and unresolved proportions separately; do not assert noisy-image universal accuracy.
- At least one clean supported fixture per sign and placement recovers every exact string in displayed order. Include punctuation and underscores to expose existing OCR weaknesses; failures remain visible and require review.
- Original 0° paths, row labels, manual/file labels and numeric reconstruction regress unchanged. One reviewed rotated-label end-to-end example produces a matrix with correct column names and unchanged values.

Explicit angles ship first. M15 adds bounded optional angle proposals; do not add arbitrary orientation detection to M13.

## M14 — Palette-diverse reconstruction and numeric-QC audit

### Preserve palette-independent design

`calibration.py` already maps cell colours to the observed bar in CIELAB. Keep that principle. Do not require a palette name or switch to a blue/red classifier, RGB intensity mapping, preset matching or colour sorting. Bar position carries numeric direction; luminance and hue need not be monotonic. Named palettes belong to fixtures/documentation, not required runtime inputs.

Add reproducible sequential fixtures (viridis, magma, cividis), grayscale, blue–white–red, teal–black–red and at least one asymmetric/custom gradient. Include both scale directions, horizontal/vertical bars, dark/light plot backgrounds, cell grid/no-grid, quantisation, neutral/dark centres, JPEG/blur and touching black borders as supported or explicitly challenged cases. Include a repeated-colour/non-injective palette and long saturation plateau as ambiguity negatives.

Use benchmark-only dependencies for fixture generation; installing the decoder must not require Matplotlib or a palette catalogue. Record LUT/anchors/interpolation. Use fixed reviewed numeric scales initially to isolate palette behaviour from tick OCR. Standard named-palette values can come from the benchmark dependency; a publication-specific palette comes from its actual bar.

### Diagnose flags before changing defaults

Analyse the corrections validation's excessive-colour-flag and missing/ambiguous cases individually. Separate cell sampling contamination, mismatched colourbar sampling/interpolation, bad ROI bounds, genuine off-bar colours, repeated positions, endpoint clipping and low-gradient colour uncertainty. Do not blindly raise colour thresholds or disable NaNs to achieve a coverage target.

Current ambiguity uses `best distance + colour tolerance` and total candidate position span. A broad contiguous near-match interval on a low-gradient palette and two distant equally coloured branches are different failure modes; report them separately. A long flat interval can still be numerically unresolved, so separation is not permission to accept it.

If changing QC, retain candidate-position intervals/components and map them through the actual calibrated scale. Use measured rendering/noise behaviour to justify engineering tolerances; do not call them calibrated statistical confidence intervals. Validate maximum accepted numeric error as well as MAE, finite coverage and missing/flagged fractions. Keep repeated/non-injective branches and materially broad ranges review-required.

Check that black-centred palette regions are not erased as presumed gridlines, missing-data colour or borders. Structural line evidence must depend on geometry/width/context, not darkness alone. Conversely, a categorical stripe must not pass because its colours happen to fit part of a continuous bar. Grayscale/dark backgrounds cannot depend on saturation.

### Acceptance

- Core manual-geometry clean unambiguous fixtures achieve MAE <2% of scale range, with maximum error and finite/QC coverage reported; compare against plotted truth, not only manual decoder output.
- Per-palette detector, association, numeric flags and complete acceptance are measured separately. Recover clean bar cores across supported schemes without introducing false competing-bar/categorical acceptance.
- Any new QC policy demonstrates improved useful acceptance and no newly accepted materially incorrect values on development/validation negatives. Version changed policy and retain previous evidence. No palette-specific truth supplied to the recogniser.
- Document dark-centre/saturated-range ambiguity, clipped limits and transformed-scale limits. Palette compatibility does not imply raw RNA-seq measurements are reconstructed.

## M15 — Bounded orientation proposals and publication-oriented validation

### Automatic sample-label region/orientation proposals

For a selected/reviewed grid, search above/below regions with a bounded set of rotations `{0,90,-90}`. Each trial runs the same transform/OCR/grouping/assignment path and retains raw evidence, slot coverage, geometric alignment, text conflicts and resource cost. Explicit user angles are authoritative.

Choose an angle only when supported by layout consistency and a unique adequate hypothesis; OCR confidence alone cannot choose it. Correct slot association and repeated recognition do not verify spelling or biological identity. Keep contradictory angles or mirrored/reversed-looking interpretations review-required. Sample names and their orientation can be reviewed independently of numbers.

Use unassigned original-coordinate candidate centres as optional count evidence, with provenance linking the angle proposal. Do not obtain count by counting accepted OCR strings or by reusing count-fitted assignments as independent evidence. Preserve missing-label holes and the existing independent-support requirements. Rotated sample labels should improve ordinary column-count support where present, not excuse false counts on unlabelled/merged axes.

### Real-figure trials and evidence

Select a small diverse lawful publication-image set with continuous bars and regular grids, including 0°/±90° sample labels and multiple palettes. Record publication URL/identifier, panel, source licence/reuse constraints, image resolution, preprocessing and annotation/review history. Do not redistribute assets without permission/licence; retain evaluation-only assets and a retrieval manifest where necessary. Exact reuse rules must be checked for each source.

Build reference annotations for ROIs, counts, label text/orientation, tick positions/values and supported scale model. Prefer two independent annotations or a second-pass audit. Numeric biological truth exists only where source data is actually available; otherwise manual digitisation measures workflow agreement, not true numeric accuracy. Mark unverifiable dimensions/values explicitly.

Keep original full panels; do not silently pre-crop away distractors and claim whole-figure automation. A cropped heatmap trial is a separate recorded mode. Avoid selecting only successful figures. Include failures such as multiple panels, shared bars, diagonal sample names and nonlinear legends as explicitly unsupported/review-required examples.

Freeze a new synthetic validation split before tuning finishes; derivatives stay with base figures, palette/layout/font diversity is recorded and layout QA checks rotated text before recognition. Previously inspected M12/corrections cases remain development regressions. Use real figures for formative trials first; reserve unseen sources if reporting generalisation.

Measure numerical extraction with supplied dimensions/scale, inferred dimensions, OCR scale and rotated-label parsing independently, then the composed requested workflow. Report exact label accuracy, false text acceptance, slot/order correctness, count exactness, palette-specific numerical errors, per-capability coverage and challenge errors. Label review must not silently block a deliberately numeric-only workflow; requested labels cannot silently become generated positional IDs.

### Usable output and release scope

Retain the current experimental export guard for supplied-count/supplied-or-reviewed-scale numeric-only requests. Do not automatically expand export permission to inferred counts, OCR scales or identifiers because a clean rotated example works. Enable each additional capability only through explicit runtime policy changes supported by its own accepted-case and negative evidence.

Provide practical reviewed examples: manual ROI + ±90° OCR + label review; auto ROI + supplied counts/scale on viridis; teal–black–red with legend verification. For supported ordinary cases seek useful per-image automation, preserving the separation of coverage targets from safety gates. When a required capability fails, save actionable review artefacts and the reason without exporting a misleading complete result.

## Implementation instruction

Inspect repository instructions and the latest correction reports. Implement **M13 only** first: explicit ±90° column-label sources, transform/grouping/slot handling, focused tests and one reviewed example. Report actual exact-string and assignment results. Do not start gene database integration, arbitrary-angle rotation or a GUI. Advance to M14/M15 only when instructed. Palette/QC and orientation evidence should extend the existing modular pipeline, not create a parallel decoder.
