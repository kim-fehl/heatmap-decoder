# Reviewing a rotated sample-label proposal

This is a synthetic six-gene/eight-sample viridis figure with +90° column names.
`truth.json` records actual plotted values, label patches, font hash and LUT.
No publication image or biological measurement is represented.

    uv run --extra ocr heatmap-decoder detect examples/m15/figure.png --config examples/m15/hints.json --out work/m15/detect
    uv run --extra ocr heatmap-decoder extract examples/m15/figure.png --config examples/m15/config.json --roi-review work/m15/detect/roi_candidates.json --roi-selections examples/m15/roi-review.json --labels-review examples/m15/reviewed_labels.tsv --out work/m15/extract --fail-on-qc

Inspect the ROI overlay and trial text before applying the bundled reference
reviews. `roi-review.json` chooses +90°; `reviewed_labels.tsv` resolves every
column spelling by original index. The opposite sign also produces aligned
upside-down strings and remains a competing hypothesis. Confidence cannot verify
names. The worked extraction has 48 finite cells, no flags, and bit-identical
numeric values to extraction without labels.

Manual ROI + quarter-turn/name review is demonstrated here; the auto ROI +
supplied counts/scale viridis and teal–black–red legend-verification examples are
in [M14](../m14/README.md). Source data is not inferred from a palette name.

Generate another synthetic example using `benchmarks/m15/fixtures.py` with the
benchmark extra and the recorded DejaVu font files. Recognition/extraction of the
shipped image requires neither Matplotlib nor a palette catalogue.
