# Heatmap2Matrix: MVP specification (v0.1)

## Goal

Build an open-source Python tool that reconstructs an **approximate numerical matrix** from a published raster heatmap using its colourbar. The first release is a **reproducible, semi-automatic CLI and Python library**, not a web application. Target use: recovering labelled or unlabelled omics heatmap data when original matrices are unavailable.

## Scope

### In v0.1
- Read PNG and JPEG heatmaps (Pillow).
- Accept user-specified pixel bounding boxes for the heatmap and a continuous colourbar, plus `rows`, `columns`, colourbar orientation and endpoint values.
- Support regular rectangular grids and continuous **linear** colour-to-value scales, including sequential and diverging palettes.
- Sample the interior of each cell robustly (avoid borders/gridlines) and map its colour to the closest position on the **observed colourbar**.
- Export a CSV matrix, machine-readable JSON provenance/configuration, and a diagnostic reconstruction/quality image.
- Report per-cell colour-matching error and flag unreliable, masked, or ambiguous values. Never silently treat poor matches as exact measurements.
- Expose a documented Python API and CLI. Run on Linux, macOS and Windows.

### Explicitly out of scope
- OCR and automatic gene/sample label extraction; user-defined labels may be supplied through optional text files.
- Automatic panel, grid, or colourbar detection; dendrogram reconstruction; annotation strips.
- Irregular cells, categorical legends, nonlinear/logarithmic legends, multiple scales, embedded colourbars and PDF import.
- Web UI, machine learning, backend services and persistent databases.
- Recovering pre-transformation biological measurements from normalised, scaled, or z-scored figures.

## Architecture

Use `src/heatmap2matrix/` with modules separated by responsibility:

- `io.py`: read raster images; write CSV/JSON/diagnostics.
- `models.py`: typed extraction configuration, bounding boxes, validation and result objects.
- `geometry.py`: validate bounding boxes and calculate rectangular cell coordinates.
- `sampling.py`: robust centre-interior pixel sampling of cells and colourbar.
- `calibration.py`: associate colourbar positions with endpoint values; perceptual colour matching (CIELAB / ΔE, via scikit-image).
- `extract.py`: pure orchestration of in-memory arrays to results; no file or CLI dependencies.
- `cli.py`: Typer CLI for configuration, errors and output paths.

Keep core functions deterministic and NumPy-based. Separate I/O from calculations; no Streamlit or framework coupling. Prefer simple, well-documented data models over a plugin architecture.

## Dependencies and development tools

- Python >=3.11; `uv` and `pyproject.toml` packaging.
- NumPy, Pillow, scikit-image, pandas, Typer; Pydantic for validated external configuration.
- OpenCV **only if needed** for a demonstrated grid/image-processing requirement; avoid unnecessary dependencies.
- pytest, Hypothesis, Ruff, mypy, GitHub Actions.
- MIT licence. Add a short note that the input figure's copyright remains with its owner.

## Input contract

CLI example (design target; naming may be adjusted consistently):

```bash
heatmap2matrix extract figure.png --config example.json --out results/
```

Example `example.json`:

```json
{
  "heatmap_bbox": [100, 80, 500, 480],
  "colourbar_bbox": [550, 80, 575, 480],
  "rows": 20,
  "columns": 20,
  "colourbar_orientation": "vertical",
  "colourbar_start_value": 2.0,
  "colourbar_end_value": -2.0,
  "sampling_inset_fraction": 0.25
}
```

Coordinates are image pixels, `[left, top, right, bottom]`, with right/bottom exclusive. For vertical colourbars, **start = top** and **end = bottom**; for horizontal colourbars, start = left and end = right. Bounds must lie inside the image, and row/column count must be positive. Document the expected assumptions: no perspective distortion, regular grid, no overlaid text within sampled cell interiors, colourbar spanning the intended numerical limits.

## Extraction and QC behaviour

1. Load image and validate configuration.
2. Divide heatmap bounding box into `rows × columns` equal rectangles; select inset central pixels for a robust representative colour (e.g. median RGB).
3. Sample the colourbar at many positions, using its central strip; ignore outer edges. Interpolate value linearly by position **along the bar**, not by RGB intensity.
4. Find each cell's closest colourbar sample in perceptual colour space; return its corresponding value.
5. Preserve both inferred values and a colour-distance/error score per cell; flag cells over a documented configurable threshold. Mark no-data cells as missing, not zero.
6. Emit `matrix.csv`, `quality.csv`, `provenance.json`, and `diagnostic.png` (the latter visually compares source sampling and reconstructed colours).

Important: a diverging colourbar can contain similar colours at different values, and clipped/saturated values may be fundamentally non-identifiable. Flag ambiguous nearest matches and do not pretend to resolve them. Distances in colour space are **not** numerical uncertainty intervals; describe them as matching diagnostics. Include the original input filename and SHA-256, config, software version and QC thresholds in provenance.

## Verification and acceptance criteria

- Generate **synthetic reference heatmaps** programmatically, with known matrices and colourbars (sequential and diverging), several cell sizes and resolutions.
- For at least one clean, lossless 20 × 20 reference with an unambiguous continuous colour scale, reconstruction MAE must be **<2% of the full numeric colourbar range**; also report maximum error and proportions flagged. Test both vertical and horizontal colourbars.
- Unit tests cover geometry boundaries, top/bottom and left/right bar direction, value interpolation, missing data, invalid coordinates, colour matching and reproducibility.
- Test degraded images (JPEG compression, blur, gridlines); expect increased diagnostics/flags rather than a universal accuracy guarantee.
- `uv run pytest`, `uv run ruff check .`, and `uv run mypy src` pass in CI; verify the CLI on a bundled example.
- Provide README with installation, one command-line example, an example figure/config and explicit scientific limitations. Exported CSV opens correctly in pandas.

## Development milestones

**M1: Core reconstruction.** Package scaffold, configuration, raster loading, geometry, sampling, calibration and CSV export; synthetic happy-path test.

**M2: Trustworthiness.** Per-cell QC, ambiguous-colour detection, provenance, diagnostic figure, edge-case and degraded-image tests.

**M3: Public release.** CLI polish, README, examples, cross-platform CI, type/lint checks, versioned v0.1 release.

**Later (not in this spec):** visual ROI editor/UI, OCR and label review, PDF/vector handling, dendrogram ordering, annotation tracks.

## Implementation instructions for Codex

Work milestone by milestone. First inspect the repository and propose only the minimal changes needed for M1. Keep modules independently testable; include tests with each feature. Do not add GUI, ML, server, database or PDF support without changing this specification. Do not infer original RNA-seq counts or claim recovered values are exact. If an extraction assumption fails, surface a clear warning or validation error instead of silently returning plausible numbers.
