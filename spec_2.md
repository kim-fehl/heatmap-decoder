# heatmap-decoder — label extraction specification (v0.2)

## Goal and relationship to spec.md

Extend the existing numerical reconstruction tool with semi-automatic extraction of **gene symbols on rows** and **sample names on columns**, using manually supplied pixel regions. Preserve the displayed matrix order. This document continues M1–M3 with M4–M6; it does not certify that earlier milestones have been implemented or tested. Inspect the repository before changing it and keep existing reconstruction behaviour and tests intact.

This specification supersedes the old project name and the prohibition on OCR for this limited scope. Other v0.1 limitations still apply. Build a CLI and Python library, without a GUI, service, database or automatic layout detection.

## Naming decision

| Surface | Required name |
| --- | --- |
| Project, repository, documentation and distribution metadata | `heatmap-decoder` |
| Primary executable and CLI examples | `heatmap-decoder` |
| Importable Python package and source directory | `heatmap_decoder`, `src/heatmap_decoder/` |
| Ordinary project directory | `heatmap-decoder/` |

Hyphens are acceptable in ordinary paths; underscores are required for the importable Python identifier. Update packaging, README, examples, diagnostics and help text consistently. Inspect existing public API usage before renaming. If compatibility aliases are necessary, keep them thin, documented and deprecated; do not duplicate implementations. Do not require users to rename existing input or output files.

## Supported scope

- PNG/JPEG inputs, existing regular heatmap grid and explicit row/column counts.
- One manually specified label region for rows and one for columns; either axis may be omitted.
- **Horizontal, upright, single-line text only**, including column labels whose text reads horizontally. No rotation, deskewing or orientation detection. Vertical/angled column labels are unsupported.
- Gene symbols occupy rows, sample names occupy columns; no transposition or automatic semantic axis inference.
- Row labels may lie left or right of the heatmap; column labels may lie above or below it. The alignment coordinate, not the side, determines assignment.
- Labels describe the displayed order, including a clustered order already present in the image. No dendrogram reconstruction or restoration of original experimental ordering.

Explicitly exclude automatic ROI discovery, scientific identifier databases, gene alias resolution, fuzzy correction, multilayer annotations and multiline labels. Nearby overlapping text may remain unresolved and must be reviewable.

## Architecture and dependencies

Retain the existing modules; add only focused responsibilities:

- `ocr.py`: a small typed adapter for local Tesseract through `pytesseract`; returns text, token boxes and engine confidence. No cloud OCR or heavyweight neural dependencies. Make OCR an optional packaging extra, e.g. `heatmap-decoder[ocr]`; document the separately installed engine.
- `labels.py`: label-region geometry, deterministic preprocessing, grouping and assignment to fixed axis slots. Keep geometry and assignment testable without the OCR engine.
- Extend existing `models.py`, `io.py`, `extract.py` and `cli.py` for configuration, results, export and orchestration; avoid a new parallel pipeline.

No general plugin framework is needed. Numeric-only operation and explicit text-file labels must work without OCR dependencies. Check the OCR dependency and engine only when OCR is requested; report an actionable installation error. Preprocessing may use existing Pillow/NumPy operations: grayscale and configurable upscale factor. Do not add OpenCV for this scope. Keep one documented default OCR mode and a small explicit configuration surface; do not search dozens of variants and silently select plausible text.

## Configuration contract

Extend the existing configuration with an optional `labels` object. Reuse existing heatmap geometry rather than asking for a second grid definition. Example addition to a complete v0.1 configuration:

```json
{
  "labels": {
    "rows": {"source": "ocr", "bbox": [510, 80, 650, 480]},
    "columns": {"source": "ocr", "bbox": [100, 35, 500, 75]},
    "ocr": {
      "language": "eng",
      "upscale_factor": 3,
      "min_confidence": 80,
      "max_axis_offset_fraction": 0.4
    }
  }
}
```

These example regions accompany the original `[100, 80, 500, 480]` heatmap. They demonstrate coordinate syntax, not a guarantee that densely packed horizontal column names fit.

- Pixel boxes use original-image coordinates `[left, top, right, bottom]`, with right/bottom exclusive. Validate non-empty bounds inside the image before cropping.
- An axis configuration has exactly one source: `ocr` with a bounding box, or `file` with a path. Reject competing sources; do not silently choose precedence. Preserve existing text-file options with a compatibility mapping if necessary.
- Omitted axes retain existing generated positional identifiers. OCR requests cannot silently become numeric-only requests.
- Reject invalid confidence thresholds, upscale factors below 1 and axis offset fractions outside `(0, 0.5]`.
- Document confidence as an OCR heuristic, not a probability. Thresholds are configurable engineering defaults, not validated biological guarantees.
- Labels are UTF-8 strings. Trim outer whitespace only; preserve case, punctuation, hyphens, underscores, digits and internal spaces. Do not replace `O/0`, `I/1`, uppercase gene symbols or strip suffixes.

## Extraction and assignment

1. Crop each requested region, preprocess deterministically and obtain tokens with boxes and confidence. Preserve raw results and transform boxes back into original-image coordinates.
2. Group tokens into single-line candidates using deterministic geometric rules; use engine line IDs only as hints. Preserve internal spaces when joining tokens. Record token membership and candidate confidence (minimum valid token confidence); ignore non-text structural records such as confidence `-1`.
3. Compute expected row centres and column centres from the **same grid** used for numerical reconstruction. Row matching uses candidate vertical centres; column matching uses horizontal centres.
4. Assign a candidate only if its centre is within `max_axis_offset_fraction × cell pitch` of exactly one slot. Require its extent along the assignment axis to remain inside that slot's full band; otherwise mark it ambiguous. Do not split text heuristically to force a match. This deliberately restricts overlapping horizontal column labels.
5. Multiple candidates for one slot, a candidate spanning slots, or unmatched candidates are QC issues. Never resolve collisions solely by confidence, compress away missing slots or shift subsequent labels. Keep exactly `rows` and `columns` slots.
6. For each slot return its index, raw candidate text, final text if resolved, confidence, original-coordinate box, source and status. Status values: `accepted`, `low_confidence`, `missing`, `ambiguous`, `manual`, `generated`. Duplicate accepted strings are permitted but flagged separately; never deduplicate, invent suffixes or merge matrix rows/columns.

Unresolved or low-confidence OCR text remains available for review but must not silently become an accepted matrix label. OCR cannot establish biological identity: even high-confidence text can be wrong.

## Review, output and CLI contract

Retain `matrix.csv`, `quality.csv`, `provenance.json` and existing numeric diagnostics. Add `labels.tsv` and `labels_diagnostic.png` when label extraction is requested.

- `labels.tsv`: one row per axis slot, in displayed order; fields `axis`, `index`, `raw_text`, `final_text`, `confidence`, `bbox`, `source`, `status`, `duplicate`, `issues`. Indices are zero-based. Use proper TSV quoting, and empty values for unavailable text/confidence/boxes.
- Overlay: show configured ROIs, expected slot centres, recognised boxes, slot indices and statuses. Save an image that makes missed and ambiguous labels visible without opening a UI.
- Provenance: existing image hash and configuration plus OCR engine/wrapper versions, preprocessing parameters, raw tokens, assignment results and review-file hash when applicable. OCR confidence is separate from numerical colour-matching diagnostics.
- `matrix.csv` and `quality.csv` use identical row/column labels. CSV writers must preserve strings and duplicate names; warn that some readers rename duplicate headers automatically.
- With no OCR requested, retain existing extraction behaviour. With OCR requested, write review artefacts first; publish labelled matrices only when every requested slot is accepted or manually resolved. On unresolved slots, return a documented distinct exit code and do not create new matrix/quality outputs. Existing files must not be overwritten without the existing explicit overwrite policy. Numeric reconstruction is available through a separate numeric-only invocation; no hidden fallback.

Design-target commands (adapt existing CLI consistently):

```bash
heatmap-decoder labels figure.png --config example.json --out review/
heatmap-decoder extract figure.png --config example.json --out results/
heatmap-decoder extract figure.png --config example.json --labels-review review/labels_review.tsv --out results/
```

`labels` runs only label extraction and writes review artefacts/provenance without requiring a colourbar configuration. It still requires heatmap bounds and row/column counts. Both commands call the same label implementation.

Provide a separate editable `labels_review.tsv` template with exactly `axis`, `index`, `text`; leave unresolved text empty. Users may correct accepted labels too. On import validate one non-empty string per requested slot, exact counts, no missing/duplicate/out-of-range indices and no unexpected axes. Preserve duplicate **text** values with warnings. Apply reviewed labels by index, mark their source/status manual and record their provenance; never mutate the raw OCR evidence. An existing text-file label source remains a simpler alternative to OCR.

## Next three milestones

### M4 — Naming and label geometry

Rename public surfaces and the Python package, preserving justified compatibility. Add validated optional label configuration, crop/coordinate transforms, fixed axis slots, candidate grouping/assignment and typed label results. Use fake OCR tokens to test geometry independently; do not require OCR installation for ordinary CI.

Acceptance: numeric regression tests pass; imports and CLI use new names; left/right and above/below ROIs map correctly; missing, duplicate and conflicting candidates preserve slot counts and order; out-of-bounds configurations fail clearly. Verify transformed boxes after upscaling. Deliver a short naming/migration note.

### M5 — Local OCR and review artefacts

Implement the optional Tesseract adapter, minimal preprocessing, confidence handling, label-only command, TSV exports and label diagnostic overlay. Include a generated image with known upright horizontal gene symbols and sample names, plus degraded variants. Record engine availability and versions.

Acceptance: an OCR-enabled integration job recovers every exact string and slot on at least one fixed, clean fixture with non-overlapping labels; examples include mixed case, digits, hyphens, underscores and a sample name containing a space. Geometry tests cover multi-token grouping and collisions with mocked data. A fixed deliberately unreadable fixture must produce unresolved slots; no threshold tuning solely to make a test pass. Report exact-match accuracy and unresolved proportion on blur/JPEG variants without promising universal performance. Numeric-only operation succeeds with OCR unavailable; OCR requests fail helpfully in that environment.

### M6 — Reviewed labelled reconstruction and v0.2 documentation

Integrate extraction, strict label gating, manual review import, matching matrix/quality labels and full provenance. Document installation of the optional engine on Linux/macOS/Windows, supported horizontal-only layouts, manual coordinate selection and the complete review workflow. Package a runnable example and add an OCR-enabled CI job alongside dependency-free core tests. Keep OCR tests against a documented engine/language-data environment; distinguish deterministic mocked tests from engine-sensitive integration tests.

Acceptance: one end-to-end run exports correctly ordered labels and unchanged numeric values; a failed OCR run emits review artefacts and a distinct exit code without publishing new matrices; a corrected review file produces complete labelled outputs. Tests cover duplicate strings, invalid review indices/counts, one-axis OCR, explicit file sources, absent OCR and backwards-compatible numeric configs. Run existing pytest, Ruff and mypy checks and the documented example. Explain that recovered values describe the figure's plotted scale and OCR labels require human verification.

## Instructions for Codex

Inspect `spec.md`, this document and the current repository first. Report which M1–M3 requirements actually exist; do not infer completion from the conversation. Implement **M4 only** first, with focused tests and a concise change report. Advance to M5/M6 when instructed. Resolve schema differences by extending existing models rather than rewriting reconstruction. Keep scope limited to manual regions, upright text, genes on rows and samples on columns. If layout assumptions fail, preserve evidence and report unresolved labels instead of returning plausible identifiers.
