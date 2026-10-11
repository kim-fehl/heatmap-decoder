# heatmap-decoder — automatic ROI detection (v0.3)

## Objective and prerequisites

Continue `spec.md` and `spec_2.md` with M7–M9: propose heatmap, continuous colourbar and upright label-region coordinates from raster figures, then feed reviewed coordinates into the existing extraction pipeline. This replaces the earlier prohibition on automatic ROI discovery only. Retain `heatmap-decoder` for public naming and `heatmap_decoder` for Python imports.

Inspect the repository and verify M1–M6 before implementation; this specification does not establish their completion. Keep numerical reconstruction and OCR behaviour intact. Automatic localisation and count inference have separate evidence and acceptance gates. Reading colourbar values and determining biological meaning remain outside scope.

## Scope and deliberate constraints

- Axis-aligned PNG/JPEG figures containing regular rectangular heatmaps and continuous linear colourbars.
- Single-panel detection initially; multiple plausible panels produce selectable candidates, not a guessed target.
- Rows remain genes, columns remain samples. Label text must remain horizontal, upright and single-line under the v0.2 restrictions.
- M7 requires user-supplied row/column counts. M8 adds optional per-axis count proposals; M9 may accept inferred counts under the gate below. Numerical colourbar endpoints/orientation semantics remain user-supplied. Detection may propose physical bar orientation; it must not infer scale direction or endpoint values from colour alone.
- Manual coordinates remain a first-class override for any region. Hybrid operation is supported: for example, fixed heatmap bounds plus detected colourbar and labels.
- No rotation, perspective correction, PDF import, learned detector, cloud service, GUI, dendrogram reconstruction, colourbar tick OCR or nonlinear/categorical scale support.
- Missing, uniform, extremely small or visually indistinguishable regions may require manual coordinates. Rectangular annotation strips and unrelated image panels are explicit confounders.

## Design principles

Produce ranked **candidates with evidence**, rather than a single unqualified box. Keep heatmap, colourbar and label-region scores separate from OCR confidence and per-cell colour errors. Heuristic scores are not probabilities. Automatic extraction requires a conservative acceptance gate; review is the default in this release.

Do not select a region because it produces plausible gene names or a biologically familiar numerical matrix. Colour consistency and geometry can constrain proposals but cannot verify that a panel is the user's intended panel. Detection should abstain when the evidence is insufficient.

## Architecture

Add `detection.py` for deterministic candidate generation/scoring and coordinate transforms. Reuse existing image loading, colour conversion, geometry, OCR adapter, result models and output infrastructure. Split helpers into small functions rather than building a general detector/plugin framework. Detection functions accept in-memory arrays and typed options; no CLI/file dependencies.

Use existing NumPy/Pillow/scikit-image functionality initially. Add no new image-processing dependency unless a measured requirement cannot reasonably be met with existing tools. Keep OCR optional: heatmap and colourbar proposals must run without it. Label-region detection explicitly requires the existing OCR extra and local engine; unavailable OCR leaves requested label detection unresolved, with an actionable error, rather than fabricating a region.

## Detection sequence

### Heatmap candidates

1. Build bounded-resolution analysis images, retaining exact scale transforms to original coordinates. Cap working size and candidate count through documented settings; never inspect only the image centre.
2. Generate axis-aligned rectangle proposals from colour/edge structure and connected regions. Use chroma and variation together: neutral or grayscale palettes must not be rejected solely for low saturation. Handle disconnected cells separated by thin gridlines without including neighbouring annotations through unrestricted component merging.
3. In M7, given supplied counts, score rectangular occupancy, approximate piecewise-constant cell interiors, consistency with an equal grid and boundary evidence. Allow weak/missing gridlines; visible periodic lines are corroborating evidence, not a requirement. Retain plausible alternatives when borders are uncertain.
4. Refine shortlisted boundaries at original resolution. Boxes must cover the numeric cell grid, excluding border strokes, dendrograms, labels and annotation tracks where separable. No fixed assumption that the largest colourful rectangle is the heatmap.

### Colourbar candidates and association

1. Propose elongated rectangles with low variation across their short axis and continuous colour variation along their long axis; support horizontal and vertical bars.
2. Score candidates jointly with each heatmap using palette coverage, geometry and proximity, while retaining the individual evidence. Similar palettes or spatial proximity alone are insufficient.
3. Exclude tick text and border strokes during refinement. Discrete categorical strips should fail continuity checks or remain ambiguous; do not stretch a partial or clipped bar into a presumed full scale.
4. Return possible heatmap–bar associations. Multiple plausible associations require selection; shared bars or multipanel layouts are review-only. The configured numerical endpoints remain user-owned input.

### Label regions

1. For a selected heatmap, search bounded strips on its left/right and above/below using the existing OCR adapter in a documented sparse-text localisation mode. Transform token boxes into original-image coordinates.
2. Group upright text geometrically and compare alignment to supplied or reviewed row/column centres. For count proposals, retain raw text positions before assignment so count inference does not depend on a preselected count. Reuse v0.2 assignment rules, preserving all slots and unresolved states. Tokens from captions, panel letters, ticks and dendrogram areas must not be accepted simply because they are nearby.
3. Propose a padded union box around the aligned candidates on each side, clipped to the image. Keep padding in image pixels or a documented fraction of cell pitch. Missing labels may yield a partial proposal flagged incomplete; do not infer invisible text extents.
4. Competing sides, conflicting tokens, unsupported orientation and inadequate alignment yield ambiguous/unresolved results. Geometry may locate likely text without successfully recognising every identifier; distinguish localisation QC from text accuracy. Never relax OCR gates just to accept a region.

### Optional count proposals (M8), automatic acceptance (M9)

- Infer rows and columns independently; a supplied positive count remains authoritative and is never silently replaced. Expose disagreement as QC evidence. Count inference requires an explicit per-axis request; missing legacy fields do not implicitly enable it.
- Generate a bounded set of positive integer candidates from repeated cell boundaries and aggregated colour-transition positions, using original-resolution evidence. Include pitch, phase, outer-grid alignment and competing fundamental/harmonic periods. Adjacent cells with equal colours hide transitions: an observed spacing may span multiple cells. Do not assume every detected transition is one boundary.
- Use OCR label positions as supporting evidence when available, keeping token grouping and text accuracy separate. Fit positions to proposed grid slots while allowing missing labels; never set the count to the number of recognised labels. Do not use assigned labels as independent evidence when that assignment already assumed the candidate count.
- Count evidence and ROI evidence must be recorded separately. Test count candidates against shortlisted boxes; preserve competing box/count hypotheses. A joint optimisation score cannot by itself validate both. No unbounded iterative box/count refinement or silent switch to the hypothesis with the most plausible output.
- Independent support means different observations, such as visible gridline positions plus spatially distributed colour transitions, or image-based cell periodicity plus unassigned OCR label positions. Two transforms or statistics of the same edge profile are not independent signals. For M9 automatic acceptance, at least two such signals must agree on exact count, pitch and phase; one signal yields review-only proposals. Completeness and harmonic ambiguity must also pass explicit checks.
- Low contrast, uniform axes, hidden boundaries, partial crops, missing labels and indistinguishable integer-multiple counts require abstention unless the count is supplied/reviewed. Some raster images simply do not identify dimensions uniquely.
- Extend detection JSON with per-axis candidate counts, signal-specific evidence, pitch/phase, conflicts, selection and origin (`manual`, `reviewed_detection`, `automatic_detection`). Overlay proposed grid boundaries and display dimensions. Use actual ground-truth pitch when computing benchmark edge tolerances, not inferred pitch.
- M8 proposals are review-only: write alternatives to detection metadata; fill the ordinary extraction configuration only after the user supplies/reviews a count. M9 explicit automation can fill a count only after its separate gate passes. Existing extraction continues to require concrete positive integer counts, with no special inference logic inside numerical reconstruction.

## Contracts, provenance and review

All boxes remain `[left, top, right, bottom]` in original-image pixels, right/bottom exclusive. Analysis-space transforms must round outward and be validated after full-resolution refinement. Candidate IDs must be deterministic for identical input/options, scoped to the saved detection result rather than assumed stable across versions.

Each candidate records ID, kind, box, physical orientation where applicable, score components, rejection/ambiguity reasons and associations. Each selected region records `manual`, `reviewed_detection` or `automatic_detection` origin. Manual bounds are never silently adjusted by detection; emit a warning if surrounding evidence disagrees. Do not reuse v0.2 label statuses as ROI statuses.

Write:

- `roi_candidates.json`: schema version, image SHA-256 and dimensions, detector version/options, candidates and associations, selected regions and unresolved requirements.
- `roi_diagnostic.png`: numbered boxes, region kinds, selected candidates and ambiguity markers over the original image.
- `config_proposed.json`: editable extraction configuration, using the existing schema. Include only resolved boxes; clearly mark it as incomplete in accompanying detection metadata when required boxes are missing. Include supplied or explicitly reviewed counts; M9 may include automatically accepted counts with recorded evidence. Leave unresolved counts absent. Never emit guessed endpoints or fabricated values merely to pass extraction validation.

Review consists of editing the proposed configuration or selecting candidate IDs against the saved JSON. A reviewed configuration is a valid ordinary extraction configuration; extraction must not require the detector to rerun. When importing a saved detection result, check image hash/dimensions, schema, IDs and box validity. Reject mismatched images. Preserve detection evidence and review hash in extraction provenance.

Design-target commands:

```bash
heatmap-decoder detect figure.png --config hints.json --out roi_review/
heatmap-decoder extract figure.png --config roi_review/config_proposed.json --out results/
```

`detect` accepts a separate validated hint model: counts (required in M7; independently supplied or requested for inference per axis from M8), any fixed boxes, requested region kinds and detector settings. It does not require colourbar endpoints for localisation. For label-only requests a bar is unnecessary. Unknown fields and conflicting manual/detection instructions fail clearly. A requested kind may be unresolved; return a documented distinct status/exit code while still saving review artefacts. Document exact hint examples from the implemented schema rather than inventing a parallel extraction format.

Optional automatic hand-off is introduced only in M9 through an explicit flag, e.g. `extract --auto-roi`, with required endpoints, supplied counts or explicit per-axis count inference requests, and requested label axes. Per-region/manual options take precedence only when explicitly configured; reject contradictory inputs. An incomplete or ambiguous detection writes diagnostics and stops before new matrix outputs. Retain existing overwrite rules. Never reinterpret an existing explicit coordinate configuration as a request to detect regions.

## Benchmarks and acceptance

Create fixtures with known numeric-grid bounds, usable bar bounds and label-text bounds. For real images annotate a defensible tolerance interval for uncertain border edges. Record source/licence and include only redistributable examples. Split development and held-out fixtures by base figure/layout **before tuning**; transformations of one figure stay in the same split.

Cover sequential/diverging/grayscale palettes; tall/wide grids; variable backgrounds and margins; gridlines/no gridlines; bars on different sides; labels on supported sides; JPEG/blur/downscaling; dendrograms, categorical annotation strips, captions, multiple panels, absent bars and ordinary non-heatmap figures. Include uniform/low-contrast matrices as abstention cases. Start with synthetic fixtures; add a small independently annotated real-figure set before claiming practical coverage.

Report per-kind IoU, edge error in pixels and fractions of cell pitch, top-k candidate recall, accepted coverage, correct-acceptance rate, abstention rate and false acceptance on negative/confounded figures. Report per-axis exact count accuracy, candidate-count recall, count abstention/coverage, and off-by-one and integer-multiple errors. Separate localisation, count inference, OCR exact-match accuracy and downstream numerical error. A high heatmap IoU can hide a one-cell offset; require edge tolerances and a downstream comparison against manually cropped extraction too.

M7/M8 fixture requirements are deterministic regression gates, not universal accuracy claims. M9 automation gate: on the frozen held-out set, every automatically accepted complete layout must have heatmap edge error at most 0.1 of the corresponding cell pitch, usable colourbar edges within annotated tolerances, and correct label-to-slot association for any accepted label region. Every automatically accepted inferred count must exactly match ground truth on its axis; approximate counts do not pass. No negative figure may yield an automatically accepted complete layout. Report sample counts and coverage; zero accepted cases is not evidence of useful automation. Require at least ten accepted held-out complete layouts spanning at least three distinct layout families before enabling automatic hand-off. These finite checks do not establish a population error rate; if unmet, ship review-only detection and report the limitation.

## Next three milestones

### M7 — Heatmap proposals and coordinate review

Implement typed detection hints/results, deterministic heatmap candidates, multiscale transforms, original-resolution refinement and JSON/overlay/proposed-config exports. Manual boxes bypass localisation. Candidate ranking is review-only.

Acceptance: fixed supported synthetic fixtures include the ground-truth grid among the top three proposals within 0.1 cell-pitch edge error; tests cover grayscale, gridlines, borders and resize transforms. Negative/uniform and multipanel examples remain explicitly unresolved or ambiguous. Numeric extraction from a reviewed proposed configuration matches extraction with manually specified bounds within the existing numeric acceptance criterion. Existing tests/checks pass.

### M8 — Colourbars, label regions and count proposals

Add continuous colourbar candidates, heatmap–bar associations, optional OCR localisation of supported label regions and hybrid manual/detected workflows. Keep colourbar detection independently usable without OCR. Add optional per-axis count proposals using the contract below. Extend diagnostics and fixture coverage to annotation tracks, misleading text and unidentifiable counts.

Acceptance: supported fixed fixtures offer correct region candidates and associations within annotated tolerances; ambiguous bars/panels are not silently selected. Requested absent bar or unavailable OCR produces unresolved artefacts and an actionable outcome. Label assignment preserves displayed order and all slots. A reviewed end-to-end example exports unchanged scientific quantities and correct labels through the existing pipeline. Count fixtures cover complete grids, omitted boundaries, merged neighbouring colours and missing labels; ambiguous integer multiples stay unresolved. Tests check image-hash mismatch, contradictory counts and conflicting inputs.

### M9 — Conservative hand-off and practical validation

Freeze the held-out benchmark and publish localisation/coverage/abstention results. Add explicit automatic extraction hand-off only if the stated gate passes; otherwise release review-only proposals with the same review workflow. Document supported layouts, failure examples, coordinate overrides and when inferred counts can be accepted, when manual counts are needed, and why endpoints remain manual. Version thresholds/configurations in provenance.

Acceptance: benchmark report includes all attempted figures and negative cases, not only successes; accepted detected extraction is compared with the same figure using manual bounds and true counts. Include at least ten accepted held-out count-inference examples per enabled axis spanning three layout families; report that subset separately from ROI-only automation. Disable automatic count acceptance for an axis that does not pass its own gate, even if ROI automation passes. Ambiguous/incomplete layouts stop before matrix export and retain diagnostics. Manual-only and v0.2 workflows regress cleanly. Run pytest, Ruff, mypy and documented CLI examples. Do not claim general-purpose figure understanding or exact recovery of original biological measurements.

## Codex implementation instruction

Read all three specifications and inspect the repository. Implement **M7 only** first, using existing dependencies and focused tests. Report unsupported layouts and actual proposal quality. Do not begin M8 count inference during M7, rotated text, tick-value OCR or a GUI. Do not disguise unresolved geometry as a successfully reconstructed matrix.
