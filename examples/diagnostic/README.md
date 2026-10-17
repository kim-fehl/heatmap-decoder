# Recognition overview

Run `uv run python examples/diagnostic/render.py` from the repository root after
`uv sync --locked`. The bundled English RapidOCR model reads labels and ticks;
DejaVu fonts are required for rendering.

The script extends the M18 synthetic development fixture with four gene labels
and three signed colourbar ticks, adding a 17-pixel row gap between gene groups. It supplies regions/counts and a +30° column
angle, then runs the production label and legend pipelines. It does not claim
automatic region detection or use reference text to repair recognition.

`docs/images/recognition-latest.png` shows resolved cells, both row/column block gaps,
actual line polygons/baselines, attachments and three tables of actual OCR results.
The tables retain raw errors, score and state; numeric ticks also show parsed value.
“Accepted” is the existing heuristic slot state, not identifier review. Non-ASCII
characters are escaped so missing font glyphs cannot conceal recognition errors.
`evidence.json` retains the full strings, geometry, engine/model metadata and
alternative tick readings. `numeric_config.json` supplies both segment axes;
the renderer verifies that adding the row gap preserves the original numeric
values and QC flags. The displayed source is cropped and enlarged only
after drawing overlays in original coordinates.

The reusable `heatmap_decoder.diagnostics.recognition_diagnostic_image(image,
labels, legend)` renderer consumes `LabelExtractionResult` and `LegendResult`.
Existing detailed CLI diagnostic artifacts remain available for deeper review.

English recognition now preserves the four sample names
(`Sample_1 long 51` through `Sample_4 long 51`). Actual mistakes such as `P53`
for `TP53` remain visible; the renderer uses no reference-name correction.
