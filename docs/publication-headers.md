# Real figures through the existing decoder

Use `group-labels`, `labels`, `legend`, and `extract`. There is no separate
`publication` command or dependency extra. The examples supply explicitly
reviewed image boxes and grid counts; these are not automatic ROI/count results.
Original images and independent references live under `examples/real_world/publication_headers`.

## Reproduce

Install the normal application and optional Tesseract wrapper with `uv sync --extra ocr`.
The group-header fixtures use the local Tesseract executable and English data;
the named-sample fixture uses the bundled RapidOCR model.

```sh
uv run --extra ocr heatmap-decoder group-labels examples/real_world/publication_headers/omtn-2021-07-019/figure-3a.jpg --request examples/real_world/publication_headers/omtn-2021-07-019/groups-request.json --out artifacts/omtn-groups
uv run --extra ocr heatmap-decoder labels examples/real_world/publication_headers/omtn-2021-07-019/figure-3a.jpg --config examples/real_world/publication_headers/omtn-2021-07-019/labels-config.json --out artifacts/omtn-labels
uv run --extra ocr heatmap-decoder extract examples/real_world/publication_headers/omtn-2021-07-019/figure-3a.jpg --config examples/real_world/publication_headers/omtn-2021-07-019/config-reviewed.json --out artifacts/omtn-reviewed
uv run --extra ocr heatmap-decoder legend examples/real_world/publication_headers/plos-pone-0244065/figure-7c.png --config examples/real_world/publication_headers/plos-pone-0244065/legend-config.json --out artifacts/mir29-legend
uv run --extra ocr python scripts/check_real_examples.py
HEATMAP_DECODER_OCR_TESTS=1 uv run --extra ocr pytest tests/test_publication_headers.py tests/test_publication_geometry.py tests/test_header_workflow.py -q
```

Label/legend inspection can exit 4 when review remains outstanding. The new
Fig. 3A gene OCR configuration deliberately requires identifier review. The
reviewed extraction configuration uses a separate visual transcription; it does
not silently substitute reference names into OCR results. `config.json` retains
positional gene rows and reviewed sample names for numeric-only inspection.

## Group tracks and individual names

`GroupLabelsRequest` extends the existing categorical-strip API:

- `source: "swatches"` (the backward-compatible default) classifies strip colors
  against reviewed legend swatches.
- `source: "lines"` accepts region `bbox` for each printed rule and a separate
  `text_bbox` for its label. Text width never defines membership.
- `source: "rectangles"` accepts header rectangles and reads their internal
  light/dark text without needing a separate legend.
- Track `name` is optional. Unnamed tracks receive positional keys `track_1`,
  `track_2`, etc.; these keys do not assert visible titles or biological meaning.
- Multiple tracks remain separate. Repeated labels such as `Normal` retain
  distinct region evidence. Short-rule gaps are bisected and snapped to supplied
  grid edges; ambiguous gaps and subpixel columns remain unresolved. No equal
  group size is assumed.
- Optional `grid` uses the existing label extractor to read individual sample
  names independently of groups. Its extent, count, and segments must match the
  group axis. File label paths resolve relative to the request JSON.

`groups.json` retains raw OCR, confidence, preprocessing, region boxes, inferred
spans, sample status, and every track assignment. `groups.tsv` includes both
sample and generated names. If individual sample names were requested,
`columns.txt` contains those names; otherwise it contains group/position display
names. Generated ordinals never establish GEO accessions. Unresolved requests
produce evidence and exit 4 without a complete names file.

The miR-29b fixture explicitly uses a permissive group OCR threshold of 45:
`Col1a1` is correctly read but has low engine confidence. Its results are OCR
candidates requiring scientific review, not verified identifiers. Greek beta is
read as Latin `B`; raw output is preserved.

## Tickless legends and small fonts

`legend.tick_position_source: "text_centres"` explicitly uses numeric-label box
centers rather than tick strokes. `legend.label_side` can restrict the search to
`left`/`right` for vertical bars or `above`/`below` for horizontal bars. Vertical
labels are localized into separate ink rows before bounded numeric recognition;
horizontal labels retain ordinary OCR localization with separate numeric bands.
No endpoint values or missing minus signs are invented. Text centers remain
weaker evidence and preserve the existing scale-review gate.

For slot label OCR, `trim_whitespace: true` crops observed dark ink on light
paper before rotation/upscaling. This helps short `NC1` labels alongside long
`Reactivation1` labels. Original-coordinate crop geometry and raw runs survive.
Configure a region that excludes adjacent panels: including a fragment of panel
B below Fig. 3A caused a misleading high-confidence `C1` reading in development.

`require_review: true` on an OCR label source prevents even high-confidence
strings from becoming final identifiers until the normal review workflow runs.
This is used for Fig. 3A's condensed gene font. OCR confidence is not exact-match
accuracy, and enlargement does not restore missing pixels.

## Validation and limits

[Evaluation results](../examples/real_world/publication_headers/evaluation.json)
are regenerated by the script above from all five original images. They report
explicit geometry, group assignments, image hashes, sample-name accuracy, gene
OCR accuracy, and color-distance diagnostics separately. Numeric estimates use
reviewed visual scale anchors. They have not been validated against deposited
biological values. Saturated/ambiguous colors can remain flagged or NaN under the
normal extraction policy.

For the dense iScience figure, the article-derived total of 1,237 exceeds the
available horizontal pixels. Header names can be read, but per-cluster sample
counts and numeric columns remain unresolved. The supplied total is not an
image-derived measurement. See [the catalogue](real-world-examples.md) for source
attribution, article inference, and GEO feasibility.
