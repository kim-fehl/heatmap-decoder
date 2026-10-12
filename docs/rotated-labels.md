# Explicit quarter-turn sample labels (M13)

Genes remain rows and samples remain columns. Only the label crop rotates;
matrix pixels, values and column order remain unchanged. Rows and legend ticks
must stay upright. Arbitrary angles, automatic orientation selection, perspective
correction and multiline text remain unsupported.

Add integer text_rotation_degrees to a column OCR source: 0 (default), 90 or -90.
The sign is the visual counter-clockwise rotation from upright in the original
figure. Strings, floats, bools and other angles are rejected. File labels have no angle.

| Value | Reading direction in original figure | Crop normalisation |
| --- | --- | --- |
| 0 | left → right | none |
| +90 | bottom → top | clockwise 90° |
| −90 | top → bottom | counter-clockwise 90° |

    +90°            0°            −90°
      C           A B C             A
      B             →               B
      A                             C
      ↑                             ↓

Glyphs turn with the word: they are sideways at ±90°. The diagram shows their
positions and reading direction, not upright stacked text.

Add this columns object within your existing labels configuration:

    {
      "columns": {
        "source": "ocr",
        "bbox": [40, 20, 552, 360],
        "text_rotation_degrees": 90,
        "recognition": "slots"
      }
    }

Bounds remain original-image [left, top, right, bottom], right/bottom exclusive.
Use a region tall enough for complete names with padding, above or below the
matrix. Explicit counts and regular column bands are required.

Both sparse (whole region) and slots (single upright line per original column
band) use exact Pillow transposes before existing grayscale/bicubic upscale.
Words group while upright. All four half-open box corners map back to original
pixels; assignment uses original x. Results follow original column indices,
retaining missing holes, duplicates and ambiguous spans.

Slot OCR keeps its 128-call, 16-million upscaled-pixel and 60-second axis budgets.
Whole-region OCR crops are capped at 16 million upscaled pixels. Rotation swaps
dimensions without changing area.

Provenance links raw upright OCR tokens to original tokens/candidates via
membership, crop, angle, transform, upscale and engine run. Confidence does
not verify identifiers. Underscores can be lost or separated.

## Reviewed example

The synthetic +90° figure has eight sample names above a 4 × 8 matrix and a
supplied grayscale bar/scale. Its underscore failures remain visible. The
review TSV is a synthetic reference annotation checked against the generator
and figure, not an automatic spelling correction.

    uv run --extra ocr heatmap-decoder labels examples/m13/figure.png --config examples/m13/config.json --out work/m13-labels
    uv run --extra ocr heatmap-decoder extract examples/m13/figure.png --config examples/m13/config.json --labels-review examples/m13/reviewed_labels.tsv --out work/m13-reviewed --fail-on-qc

For your figure, verify/fill every requested slot in the generated
labels_review.tsv against the original, then pass that file instead. Review
preserves raw evidence. Experimental automation export policy is unchanged.

See [M13 results](progress/M13.md) and the
[conditional OCR report](../benchmarks/m13/report.json) for exact spelling,
geometric assignment, unresolved proportions and errors.
