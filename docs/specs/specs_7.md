# Phase 7 — Blocked grids, arbitrary-angle sample labels and common-case automation

**Status:** proposed implementation specification.  
**Milestones:** M17–M19.  
**Suggested repository path:** `docs/specs/specs_7.md`.  
**Starting point:** branch `work`, inspected at `452c8c0`, following M16.  
**Naming:** project/CLI `heatmap-decoder`; Python package `heatmap_decoder`.

## 1. Objective

Support horizontal and vertical group gaps and arbitrarily rotated sample names, while making confident automatic numeric decisions on common supported figures a first-class deliverable.

The phase must improve complete workflows, rather than deliver new diagnostic proposals that always require review. Start each new capability with explicit configuration and deterministic geometry; then integrate detection and acceptance. Preserve a useful manual path and existing numerical ambiguity protection.

## 2. Reviewed starting point

- M16 fresh synthetic validation completes 43/48 numeric attempts accurately, with no observed wrong complete exports and no exports on six negative cases. This is conditional evidence: the evaluator supplies heatmap/bar ROIs, dimensions and reviewed calibration. It is not an automatic ROI/count/scale success rate.
- The inspected squid publication crop yields 251/264 finite cells with supplied geometry and reviewed calibration. Automatic ROI and the composed workflow abstain. Its five physical legend ticks are recognised exactly, but scale coverage remains unresolved beyond the outer labelled ticks.
- Publication label OCR is exact for 26/33 genes and 8/8 samples. Six wrong genes pass the heuristic confidence threshold. Confidence alone cannot authorise identifiers.
- `axis_slots()` divides each entire heatmap axis uniformly. Extraction and label assignment share those slots. Gaps require extending this shared model, rather than independently patching sampling and OCR.
- Sample-label transforms currently support only 0/±90°, using exact transposes and original-x assignment. Diagonal text crosses column bands and needs a different crop/association path.

Keep historical reports unchanged. The existing inspected publication is development evidence. Numeric calibration, source-data provenance and identifier accuracy remain separate questions.

## 3. Scope

### Supported target family

An axis-aligned rectangular matrix, with upright gene labels and sample names sharing one angle per selected label region. The matrix may contain full-width horizontal gaps, full-height vertical gaps, or both. Blocks share one colourbar and numeric scale. Cells are regular within each axis segment; explicitly supplied geometry may describe different segment pitches.

Automatic support starts with common-pitch cells across segments, variable gap widths, and clear association to one legend. Arbitrary-angle sample OCR means an explicitly supplied finite angle can be normalised; automatic angle search is bounded and may require review. Text may be above or below the matrix. Reading order follows displayed columns.

### Out of scope

Rotated gene labels, rotated matrices, perspective correction, multiline or individually mixed-angle sample names, automatic biological name correction, identifier databases, independent panel assembly, separate block scales, GUI work, guessed edge values and recovery of original expression units from displayed z-scores.

## 4. Shared architectural requirements

Introduce a resolved grid geometry consumed by sampling, count inference, OCR ownership, diagnostics and exports:

- Ordered row and column cell intervals in original-image coordinates.
- Explicit gap intervals and axis segment/group boundaries.
- Stable global cell indices and segment-local indices.
- Source, uncertainty, proposal/review status and geometry version.

Existing regular-grid configurations resolve to the same intervals and retain their existing numerical outputs. Do not create separate geometry representations for numeric and text paths.

Keep proposed alternatives separate from resolved geometry. No undecided gap or count may silently enter sampling. User-supplied geometry remains authoritative subject to validity checks; it is an assumption, not evidence of successful automatic detection.

All exported geometry, labels and values must refer to the same resolved grid hash. Group boundaries are metadata; gaps are not matrix rows or columns.

## 5. M17 — Block-aware geometry and gap/count detection

### M17.1 Explicit geometry

Add a documented configuration path for axis segments with original-coordinate bounds and cell counts. For example, a row segment is `[start, end)` plus its row count; gaps are the uncovered intervals between ordered segments inside the heatmap ROI. Equivalent explicit gap/count syntax is acceptable if it resolves unambiguously to the same geometry.

Requirements:

- Validate finite, ordered, non-overlapping, in-bounds segments and positive integer counts. Counts must sum to any supplied total; reject contradictions.
- Preserve at least one sampleable interior pixel per cell after inset. Report insufficient resolution explicitly.
- Sample each cell inside its interval, excluding separators and strokes. Do not delete gap pixels and rescale the remaining image.
- Preserve global row/column order, including simultaneous horizontal and vertical gaps and unequal group sizes.
- Produce one matrix for one shared-scale blocked grid; retain group boundaries and source coordinates alongside it.
- Ensure OCR slot ownership and diagnostic overlays use exactly the same intervals.

### M17.2 Automatic proposals

Extend ROI detection to consider a union of aligned blocks. The target ROI must encompass the complete requested matrix, not just the largest connected block. Distinguish a blocked matrix from neighbouring independent panels through alignment, compatible pitches, transverse extent and legend association.

Generate separator hypotheses using transverse profiles, boundary evidence and continuity of the surrounding lattice. Background colour alone is insufficient. In particular, white can be a valid palette colour, and missing-data bands can look like separators.

Retain competing interpretations when a uniform-valued row/column, annotation strip or missing-data band is indistinguishable from a gap. Group labels may support a hypothesis but must not independently determine dimensions.

Infer counts per segment and aggregate them globally. Keep observed transitions/strokes, pitch estimates, residuals, endpoint evidence and count alternatives. Avoid circular validation: an assumed count used to fit a grid is not an independent confirming signal. Multiple colour transitions derived from one profile are also not independent votes.

Evaluate counts under the declared regular-cell model. Reject harmonics that fail observed interior constancy or require boundaries cutting through cells. Do not require universal uniqueness outside that model, but do retain genuinely indistinguishable alternatives.

### M17.3 Acceptance and tests

Explicit geometry must reconstruct horizontal-only, vertical-only and combined gaps correctly. Clear automatic common-pitch cases must accept, while ambiguous missing-data bands, white-valued cells, annotation strips and independent panels must not be falsely merged or removed.

Required regressions include no-gap compatibility, thin strokes versus wide separators, variable gap widths, unequal groups, constant neighbouring cells, blur/JPEG, cropped blocks, wrong segment counts and label/order preservation. Assert resolved intervals, dimensions, decisions and numeric accuracy, not merely file creation or review-required status.

## 6. M18 — Arbitrary-angle sample-label OCR

### M18.1 Explicit angle and transforms

Allow finite numeric sample-text angles in a documented canonical range, with the existing visual counter-clockwise convention. Reject booleans and non-finite values; document angle normalisation. Keep exact existing transpose paths at 0/±90° to avoid introducing interpolation changes there.

For other angles, use an explicit affine transform with expanded canvas and recorded padding/interpolation. Retain its inverse. Only text crops rotate; matrix geometry and order do not.

Keep text polygons or equivalent oriented geometry through the transform. Axis-aligned bounding boxes may be provided for display, but are insufficient as the sole association evidence. Record original and normalised coordinates and image/crop/transform provenance.

### M18.2 Segmentation and association

Detect/group text lines in the sample-label region before applying narrow per-column crops. Normalise each oriented line crop for recognition; do not assume the original column band contains the complete diagonal name.

Associate each line to a column using its attachment geometry near the matrix: the matrix-side text endpoint or baseline projected to the relevant boundary, with configured tolerances. Account for above/below placement, both angle signs and both reading directions. Do not assign diagonal names by bounding-box centroid alone.

Keep attachment interpretation explicit; inconsistent or overlapping attachments require review. Preserve missing slots, duplicate strings and competing assignments. Group headers must not become sample names. Spatial association must not depend on recognised spelling or a vocabulary.

Retain spaces, underscores, digits and raw OCR alternatives. Never silently concatenate conflicting readings or substitute generated names. Reuse bounded call/pixel/time budgets across detection, angle trials, preprocessing and recognition, including skipped/exhausted trials.

### M18.3 Bounded automatic angle proposals

Estimate line orientations geometrically, then use a small deterministic coarse-to-fine candidate set. Do not exhaustively OCR every degree. Orientation evidence and recognition confidence remain separate; the highest OCR score alone must not select an angle or settle an upside-down alternative.

An explicit angle overrides angle proposals. A unique supported angle/assignment may be accepted for geometry while recognised names remain unreviewed. Ambiguous orientation need not block numeric-only extraction.

### M18.4 Tests

Cover at least 0, ±15, ±30, ±45, ±60, ±75 and ±90°, plus non-integer angle transform tests. Include above/below placement, blocked columns, long names crossing neighbouring bands, both alignment directions, missing names, group headers, tight spacing, JPEG and blur.

Test affine point/polygon round-trips separately from engine-dependent OCR. Assert original-coordinate alignment, column ownership and exact order. Keep numeric-only output invariant under label settings. Raw/accepted exact strings and false heuristic acceptances must remain visible even when the geometry is correct.

## 7. M19 — Confident common-case automatic decisions

### M19.1 Capability-specific decisions

Replace unconditional review blockers for newly validated capabilities with explicit evidence-based decisions. Audit ROI uniqueness, block geometry, per-axis counts, legend association, tick calibration, numeric quality, orientation/assignment and identifier review separately.

Distinguish hard contradictions from incomplete or weak evidence. A hard contradiction rejects the relevant hypothesis; weak evidence requires a concrete explanation or bounded follow-up. Do not increase acceptance by weakening colour ambiguity, numerical-error limits or extrapolation protection.

Retain an auditable decision record containing evidence references, assumptions, alternatives considered, support scope and policy version. Scores are not calibrated probabilities unless independently evaluated as such; do not label heuristic scores as percentage certainty.

### M19.2 Automatic dimensions and linear tick scales

Allow automatic counts for accepted regular/blocked geometry when supported by measured boundaries and structure and when the relevant validation gate passes. Existing manual counts remain an override, not a prerequisite forever.

Allow a linear scale from recognised legend ticks when:

- At least three distinct physical ticks have reliable position/value associations.
- Signs, ordering and monotonic direction are consistent; a preregistered position/value fit passes.
- No unresolved tick reading or competing legend changes the model.
- The required matched positions lie within the validated calibration span. No extrapolation is implicit.

A bar may extend beyond its outer ticks without invalidating cells that match within the calibrated span. Cells outside that span remain unresolved; obtain reviewed edge values only if necessary for them. Failed linear fitting must not automatically choose piecewise calibration, log scales or a subset of convenient ticks. Reviewed nonlinear calibration remains supported separately.

Do not count interpolated colours as numeric anchors. Keep colour-trajectory matching independent of numeric-scale fitting.

### M19.3 Numeric acceptance versus names

An uncertain name must not invalidate correct numeric geometry and values. Expose a numeric matrix with clearly positional row/column IDs and a separate label-evidence table when identifier review is incomplete. Do not present those IDs as recognised names, and do not silently fulfil a labelled-export request with positional output.

Provide explicit completion states for numeric output, label geometry and reviewed identifiers. A requested composed run should retain successful numeric results while reporting identifier work still needed. Unresolved numerical cells remain a clearly partial output and never count as a complete export.

Biological identifiers do not gain automatic acceptance merely from OCR confidence. No database integration is included in this phase.

### M19.4 Export policy

Newly validated counts and linear OCR scales may enter the explicit experimental export path. Remove blanket scope restrictions only for capabilities whose registered validation gates pass. Default diagnostic behaviour and explicit experimental export remain until a separate release decision; this phase does not require another confirmation for each eligible image.

Separate corpus-level release qualification from the per-image decision. A global coverage target cannot prove an individual crop correct, and should not permanently force every otherwise qualified image to abstain. Document support scope and maturity without inserting a blanket review-required result into every supported case.

## 8. Validation design and preregistered targets

### Partitions and workflows

Use development/regression cases to fix failures, then freeze source, models, dependencies, inputs, policy and thresholds before evaluating fresh validation. Keep derivatives with their base; report bases and correlated attempts separately. Vary layout, text/font, matrix size, gap placement and renderer choices as well as numeric seeds.

Minimum synthetic validation: 40 new base figures, each clean/JPEG/blur, balanced across no gaps, horizontal gaps, vertical gaps and both axes. Include palette diversity, bar orientations/directions and a balanced explicit-angle subset. Add at least 12 negative bases spanning separator impostors, independent panels, wrong counts, truncated text, competing legends, inconsistent ticks and unsupported calibration edges.

Evaluate progressively reduced hints on every applicable case:

1. Supplied geometry/counts/reviewed scale: compatibility and reconstruction ceiling.
2. Automatic ROI/gaps/counts with supplied reviewed scale: geometry automation.
3. Automatic ROI/gaps/counts and linear OCR scale: complete numeric automation.
4. Explicit-angle labels, then bounded angle/assignment proposals: label capabilities.
5. Composed numeric and label workflow: retained numeric success plus unresolved identifier status.

A supplied truth annotation must never enter production detection or OCR. Reference tables are evaluation evidence only, with documented transformation/provenance limits.

### Targets

| Capability / partition | Required outcome |
| --- | --- |
| Explicit geometry, clean supported fixtures | Correct dimensions/order on every case; full numeric coverage where scale coverage permits |
| Auto geometry/counts + supplied scale, clean no-gap bases | ≥80% complete and accurate |
| Auto geometry/counts + supplied scale, clean blocked bases | ≥70% complete and accurate in each gap family |
| Same geometry workflow, all ordinary derivatives | ≥70% complete and accurate overall; report each family/degradation |
| Fully automatic numeric workflow, clean bases with readable ticks and fully supported scale span | ≥70% complete and accurate; eligibility declared before recognition |
| Explicit-angle label geometry, clean resolvable cases | 100% correct column association/order; no silent slot shifts |
| Automatic angle/assignment, clean resolvable cases | ≥80% correctly resolved; zero observed wrong accepted associations |
| Numeric accuracy for every complete accepted case | MAE <2%, maximum ≤5% of displayed range; all cells finite; ≤1% flagged |
| Accepted complete numeric outputs / negative cases | Zero observed wrong complete exports; no misleading negative exports |

These are completion gates, not claims of zero population risk. Record confidence intervals or base-level uncertainty where appropriate. Label spelling remains a reported metric, not an automatic identifier-export authorisation.

Do not retrospectively exclude failures as unsupported. Register eligibility criteria and family denominators before evaluation. If a target fails, retain the report and restrict that capability's scope; changes require a new freeze and validation partition.

### Publication trials

Use at least six distinct publication figures, with at least three held aside from tuning. Represent real separators and diagonal sample names; record licence/source, crop identity, hashes, independently audited geometry/label/tick annotations and unresolved numerical assumptions. Additional external images must be obtained within available access and reuse conditions; report a blocked trial honestly rather than inventing fixtures.

Report whole-figure and supplied-panel results separately. The already inspected squid crop is a regression/development case. Do not require scientifically unknowable edge values to be guessed simply to achieve completion. Synthetic gates may qualify a bounded capability, but publication generalisation claims require actual held-out publication evidence.

## 9. Diagnostics, reports and regression policy

Show cell intervals, separator spans, group boundaries, full-grid ROI alternatives, text polygons/baselines, attachment positions, angle hypotheses and tick calibration span in the existing diagnostic style. Link rejection reasons to these observations.

Every attempted workflow must retain coverage, exact dimensions/order, numeric error, flags, label raw/accepted accuracy, wrong acceptances, unresolved proportions, runtime and supplied-hint provenance. Break abstentions down by the earliest unresolved requirement and subsequent detectable blockers. Report how many failures are resolved by one manual field versus substantial reannotation.

Keep earlier low-level and ambiguity regressions. Strengthen positive tests that previously asserted only review-required/file existence when the supported capability is now intended to accept. Pair each ordinary acceptance case with a nearby misleading case. Historical frozen reports remain unchanged and inspected validation becomes regression evidence.

Run focused tests, the full suite, real OCR checks, Ruff, strict mypy, documented CLI examples and installed-wheel smoke checks. Record actual results; do not inherit old pass counts as verification of new code.

## 10. Implementation sequence and completion

Implement M17 first: shared geometry, explicit blocked extraction, then separator/count proposals and positive/negative acceptance. Implement M18 next: explicit affine text geometry, attachment assignment, then bounded angle proposals. Implement M19 after those interfaces stabilise: targeted acceptance changes, automatic linear tick scales, partial label-independent outputs and frozen full-workflow validation.

Before detector changes, save current baselines for the whole-image and supplied-hint workflows. Select repairs by failure frequency and severity, including existing ROI failures; do not spend the phase only adding features while leaving ordinary automatic detection untouched.

Deliver implementation, migration/configuration documentation, runnable examples, milestone reports, reproducible frozen benchmarks and a capability table identifying automatic, reviewed and unsupported paths. The phase is complete only when registered supported-scope gates pass or failed capabilities are explicitly documented as incomplete. No stable-release or full-publication-automation claim follows from synthetic completion alone.
