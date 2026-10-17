# M14 worked observed-palette examples

Both images are synthetic 24 × 12 grids with a continuous vertical bar, explicit
counts and fixed displayed scale -3 → 5. Truth and LUT/anchor metadata remain in
the adjacent truth JSON files. The decoder requests contain no palette name,
truth matrix or ground-truth ROI for automation.

Regenerate with the benchmark-only dependency:

    uv run --group benchmark python examples/m14/generate.py

Supplied counts/scale, detected viridis ROIs, scoped experimental numeric export:

    uv run --extra ocr heatmap-decoder automate examples/m14/viridis.png --request examples/m14/viridis-request.json --experimental-export --out work/m14-viridis

Manual ROI and verified teal–black–red legend direction/range:

    uv run heatmap-decoder extract examples/m14/teal_black_red.png --config examples/m14/teal_black_red-config.json --out work/m14-teal-black-red --fail-on-qc

The latter uses supplied manual geometry and scale, not tick OCR. Verify top = -3,
bottom = 5 and the black centre = 1 against the supplied figure/configuration.
RGB anchors are (0,180,180), (0,0,0), (210,0,0) at 0, 0.5, 1 with linear RGB
interpolation. This is an explicit fixture palette, not a named publication
palette. On your publication, verify its actual legend values and observed core.

Both worked commands passed with 288/288 finite cells and zero flags. Palette
compatibility does not certify gene/sample identifiers or original biological
measurements. See [results and limits](../../docs/progress/M14.md).
