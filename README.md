# heatmap-decoder v0.2.0

Reconstruct an **approximate** numerical matrix from a PNG/JPEG heatmap and its
observed continuous, linear colourbar. Python 3.11+.

A deterministic Python library and command-line tool for semi-automatic reconstruction.
Supply the grid, heatmap/colourbar bounding boxes and endpoint values; inspect the
resulting QC before using the estimates scientifically. Supports sequential and
diverging palettes with continuous linear scales on Linux, macOS and Windows.

The optional local OCR adapter extracts upright horizontal gene/sample labels
from manually supplied regions, preserving displayed order and raw evidence.
Review files can correct any recognised string by its displayed axis/index.
Labelled matrix publication requires every requested slot to be accepted or
manually resolved. See the [complete v0.2 workflow](M6.md) and
[migration note](MIGRATION.md). Numeric-only and explicit file-label use require
no OCR dependency or engine.

```bash
uv sync --locked --extra ocr  # separately install Tesseract and English data
uv run --extra ocr heatmap-decoder labels examples/labels/clean.png \
  --config examples/labels/config.json --out review/
```

This writes label TSVs, an overlay, and provenance. Exit 4 leaves unresolved
slots for review; confidence is an OCR heuristic and accepted text can be wrong.

## Install and use

Install this checkout into your own Python environment:

```bash
python -m pip install -e .
heatmap-decoder --version
```

For the bundled example and development tools, clone the development branch:

```bash
git clone --branch work https://github.com/kim-fehl/heatmap-decoder.git
cd heatmap-decoder
uv sync --locked
uv run heatmap-decoder extract examples/figure.png --config examples/config.json --out results/
```

The [bundled figure](examples/figure.png) is a clean synthetic 20 × 20 diverging
heatmap with known [truth](examples/truth.csv) and [configuration](examples/config.json).
Its generator uses a fixed seed and an independent analytic palette. Regenerate it
with `uv run python examples/generate.py`.

![Synthetic example heatmap and colourbar](examples/figure.png)

For your own figure:

```bash
heatmap-decoder extract figure.png --config config.json --out results/
```

Use `heatmap-decoder extract --help` for all options. `python -m heatmap_decoder`
provides the same CLI. Outputs are not replaced unless `--overwrite` is passed.
`--mask mask.npy` accepts a boolean H × W NumPy array saved with `numpy.save`;
`True` means missing. Object arrays/pickled input are not accepted. `--fail-on-qc`
writes outputs and exits 3 when any cell is flagged, for automated pipelines.
Exit 0 means outputs were written; exit 2 means invalid input or a filesystem error.
QC warnings appear on stderr even when `--fail-on-qc` is omitted.

Python API:

```python
from pathlib import Path
from heatmap_decoder import ExtractionConfig, extract_file

config = ExtractionConfig.model_validate_json(Path("config.json").read_text())
result = extract_file("figure.png", config, "results/")
```

Configuration follows this contract (QC fields are optional):

```json
{
  "heatmap_bbox": [100, 80, 500, 480],
  "colourbar_bbox": [550, 80, 575, 480],
  "rows": 20,
  "columns": 20,
  "colourbar_orientation": "vertical",
  "colourbar_start_value": 2.0,
  "colourbar_end_value": -2.0,
  "sampling_inset_fraction": 0.25,
  "max_colour_distance": 10.0,
  "ambiguity_colour_tolerance": 1.0,
  "ambiguity_position_fraction": 0.05,
  "minimum_valid_fraction": 0.5,
  "missing_colours": [],
  "missing_colour_tolerance": 3.0
}
```

Bounds are `[left, top, right, bottom]`, right/bottom exclusive, within the image.
Rows run top to bottom; columns left to right. Bar start means top for vertical
bars and left for horizontal bars, regardless of which endpoint is larger.
The bar ROI must exclude ticks, text and axial borders. Its first and last axial
pixel samples represent the stated endpoint values. All axial pixels are sampled;
the central half across the bar width is reduced by median. Cell sampling uses
pixel centres inside the inset interior, also reduced by median RGB. Empty
interiors fail clearly; lower the inset or use a finer raster.

## QC and missing data

Distances are CIELAB ΔE76 under D65: **colour matching diagnostics, not numerical
uncertainty intervals**. Thresholds are configurable heuristics, not calibrated
scientific confidence levels.

| Configuration | Default | Meaning |
| --- | --- | --- |
| `max_colour_distance` | 10 | Flag `POOR_MATCH` when nearest ΔE76 exceeds this |
| `ambiguity_colour_tolerance` | 1 | Candidate samples lie within best ΔE76 + this tolerance |
| `ambiguity_position_fraction` | 0.05 | Flag `AMBIGUOUS` when candidate positions span more than this fraction of the bar |
| `minimum_valid_fraction` | 0.5 | Minimum valid fraction of sampled interior pixels |
| `missing_colours` | empty | Explicit no-data RGB sentinels, e.g. `[[255, 0, 255]]` |
| `missing_colour_tolerance` | 3 | Maximum ΔE76 from a sentinel for a pixel to count as missing |

Repeated colours, broad flat/saturated plateaus and weakly identifiable regions
can trigger ambiguity. Adjacent similar colours on a smooth, identifiable scale
need not trigger it. Missing and ambiguous cells have `NaN` in `values` and the
matrix CSV. Unambiguous poor matches retain their estimates with `POOR_MATCH`.
Flags can combine. `inferred_values` preserves the nearest candidate for diagnosis;
it must not be treated as an ambiguity-resolved measurement.

Missing pixels come from a boolean mask (`True` = missing), nonopaque alpha
(including partially transparent pixels), or configured sentinel colours. No
colour, including black or white, is automatically missing. Remaining valid
pixels determine the median; cells below the coverage threshold are `MISSING`.
The colourbar's central sampling strip must be fully valid or extraction fails.

## Labels, coordinates and review

Install the optional wrapper with `uv sync --locked --extra ocr` or
`python -m pip install -e '.[ocr]'`. Install the engine separately:

- Debian/Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`.
- macOS: `brew install tesseract` (English data is included).
- Windows: install a Tesseract Windows build with English language data and add
  the directory containing `tesseract.exe` to PATH; the maintained installers
  linked from [Tesseract's installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
  include Windows instructions. Restart your terminal after changing PATH.

Verify `tesseract --version` and `tesseract --list-langs`. The Python extra does
not install the engine or traineddata. Missing wrapper, engine or language data
produces installation guidance when OCR is requested. A review with an OCR source
reruns OCR to preserve evidence; it does not bypass a missing engine. File sources
are the alternative when OCR is unavailable.

Select regions manually in an image viewer/editor with a pixel-coordinate readout.
Use the original raster at its actual pixel size. Record the heatmap's outer grid
bounds, row/column counts, and a separate rectangle tightly containing each axis's
complete labels with padding. The coordinate origin is the upper-left corner;
convert inclusive endpoint readouts to exclusive right/bottom bounds. The image
must have a regular grid. Regions may be left/right for rows and above/below for
columns. Horizontal, upright, single-line text is required; vertical or angled
sample labels are unsupported. No rotation, automatic layout detection,
dendrogram reconstruction or semantic axis inference is performed.

Add this `labels` object to a complete numeric configuration (the regions below
are coordinate examples; adapt them to your own image):

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

Each axis has one source: OCR with a `bbox`, or file with a `path`. File sources
contain exactly one non-empty UTF-8 line per slot; paths resolve relative to the
config file. Only outer whitespace is trimmed. Omitted axes get `row_0`, `row_1`,
… or `column_0`, `column_1`, … when the other axis is labelled. With neither axis
requested, numeric-only output retains its existing format.

OCR uses grayscale, bicubic upscaling, and one `--oem 1 --psm 11` sparse-text pass.
Geometry assigns candidates to fixed grid bands without shifting missing slots
or choosing collision winners by confidence. Confidence is a heuristic, not a
probability or a guarantee of biological identity. Case, digits, punctuation and
internal spaces are preserved. Low-confidence, missing, spanning or conflicting
candidates remain unresolved. Inspect **all** OCR labels, including accepted ones.

```bash
heatmap-decoder labels figure.png --config config.json --out review/
# Edit review/labels_review.tsv after inspecting labels.tsv and the overlay.
heatmap-decoder extract figure.png --config config.json \
  --labels-review review/labels_review.tsv --out results/
```

`labels` needs grid/label configuration only; no colourbar is required. It writes
`labels.tsv`, `labels_review.tsv`, `labels_diagnostic.png`, and `provenance.json`.
The review template contains exactly `axis`, `index`, `text`, with zero-based
indices and unresolved text empty. Fill every slot of every requested axis; you
may correct accepted labels too. Keep omitted axes out of the review. Import
rejects missing/duplicate/out-of-range indices, unexpected axes, wrong field
counts and empty strings. Duplicate **text** is allowed and flagged separately.
Reviewed slots become `source=manual`, `status=manual`; raw OCR evidence is retained.
Use a separate results directory so input review files cannot be overwritten.

`extract` writes review artifacts before publication. Unresolved labels return
**exit 4** and do not create new matrix/quality outputs. Existing numeric files
are retained even with `--overwrite`; blocked provenance identifies those older
files. Inspect the current exit code and publication status before using outputs.
A complete review permits publication with matching labels and unchanged numeric
values. Numerical QC still uses exit 3 with `--fail-on-qc`; invalid input, missing
OCR and I/O errors use exit 2. For a separate numeric-only run, omit label sources.

Run the bundled complete example (local OCR required):

```bash
uv run --extra ocr python examples/labels/workflow.py --out work/v0.2-example
```

It verifies a clean labelled export, a blocked unreadable figure, and a reviewed
export using the fixture's supplied known labels. For real figures, a person must
verify/fill the review file; the tool does not infer corrections. For a file-only
example without OCR, use `examples/labels/file_config.json` with
`examples/labels/reconstruction.png`. See [M6.md](M6.md) for API and validation details.

For in-memory use:

```python
from heatmap_decoder import CellFlag, extract, load_raster

image = load_raster("figure.png")  # uint8 RGB, or RGBA if transparency is present
result = extract(image, config, mask=None)  # optional boolean H × W mask
poor = (result.flags & CellFlag.POOR_MATCH) != 0
```

Numeric-only `extract` performs no I/O and is deterministic. Labelled in-memory
extraction may invoke the local OCR engine; text-file contents and optional
review mappings can be supplied by the caller. Results contain matrices
`values`, `inferred_values`, `colour_distance`, `flags`, and `valid_fraction`;
`sampled_rgb` and `reconstructed_rgb` have shape `(rows, columns, 3)`.
`result.labels` is absent for numeric-only extraction and contains evidence and
final fixed slots for labelled extraction. `UnresolvedLabelsError` carries the
unresolved evidence for in-memory callers.

## Outputs

`extract_file` reads source bytes once, computes hashes and writes:

- `matrix.csv`: headerless/indexless numerical matrix; `nan` marks missing/ambiguous cells.
- `quality.csv`: one row per cell, with zero-based row/column, ΔE76, flags, valid fraction and nearest candidate value.
- `provenance.json`: source filename/SHA-256, image shape, software version, complete configuration, QC thresholds, methods, flag proportions and optional mask SHA-256.
- `diagnostic.png`: source representative colours, nearest colourbar candidates and QC. Grey crossed cells are missing; purple is ambiguous; orange is poor; a black outline also marks poor matches when flags combine.

For a supplied mask, SHA-256 covers its boolean bytes in C order; retain the mask
alongside the input if you need to reproduce the extraction. Exports are
byte-reproducible for identical inputs with the same installed dependencies.
Read the matrix with `pandas.read_csv("results/matrix.csv", header=None)` when
using pandas. The file adapter prevents outputs from overwriting the input figure.

For labelled extraction, `matrix.csv` has a `gene` index column and sample
headers; `quality.csv` keeps its indices and numeric diagnostics and adds
`row_label`, `column_label` with the identical strings. Read labelled matrices
with `pandas.read_csv(path, index_col=0)`. CSV writers preserve duplicate names
without suffixes; some readers, including pandas, rename duplicate column headers.
Use Python's `csv.reader` when exact duplicate headers matter.

Labelled extraction also exports the label TSVs and overlay. Provenance schema 2
combines the numeric methods/QC/input hash with raw OCR tokens, engine/wrapper
versions, preprocessing, original assignments, final reviewed slots, source-file
hashes, and review-file SHA-256. OCR confidence is separate from numerical QC.
Outputs and input files are protected by explicit overwrite checks, including the
file adapter's `overwrite=True` option. Neither flag permits overwriting an input.

## Validation

```bash
uv run pytest -q -s
uv run ruff check .
uv run mypy src
uv sync --locked --extra ocr
HEATMAP_DECODER_OCR_TESTS=1 uv run --extra ocr pytest -q -s -m ocr_integration
```

CI runs these checks and the example CLI on Linux, macOS and Windows with
Python 3.11–3.14. It also builds the distributions and runs the installed wheel
in a fresh environment outside the source checkout. Hypothesis exercises geometry
and calibration invariants. CLI tests verify exit codes, masks, overwrite protection,
and pandas CSV reading.
Core tests use mocked OCR and require no wrapper/engine. A separate Ubuntu 24.04,
Python 3.12 OCR job installs Tesseract/English data, logs engine/data versions and
the English traineddata hash, runs fixed OCR integration fixtures and the complete
v0.2 example. See [M5 measurements](M5.md) and [M6 validation](M6.md) for the
documented engine environments and distinction between mocked and sensitive tests.

Synthetic tests generate seeded 20 × 20 references independently of extraction,
using sequential and diverging palettes, both orientations, ascending/descending
endpoints and 4/9/16-pixel cells. They exercise PNG loading through CSV round-trip,
assert MAE <2% of the numerical range, and report MAE, maximum error and flagged
proportions. Clean references retain M1 accuracy with no flags.

QC tests cover masks/alpha/sentinels, partial coverage, all-missing matrices,
repeated colours, saturated endpoints, incomplete colourbars, provenance hashes,
combined flags, deterministic exports, gridline avoidance, JPEG compression and
blur. Degradation tests use a curved colour scale and explicitly stricter QC
thresholds; they verify increased colour errors/flags, not universal accuracy.

## Assumptions and limitations

Use a regular rectangular grid without perspective distortion or text in sampled
interiors. The colourbar must span the intended limits with a linear value scale.
JPEG and blur may change values without increasing colour distance if their
altered colours still lie on the observed colourbar. A small ΔE or no flag does
not guarantee an accurate numerical estimate. Endpoint matches cannot reveal
values clipped beyond the displayed limits; plateau detection cannot recover them.

These values approximate the figure's displayed scale, not exact measurements or
original pre-transformation biological values. OCR cannot establish biological
identity; labels need human verification. No automated ROI detection, rotated
text support, PDF support, web UI or inference of RNA-seq counts is included.

MIT licensed. Input figures remain copyrighted by their respective owners.


## Release packaging

```bash
uv build
uv run python tests/smoke_distribution.py
```

The wheel includes the typed Python package and console entry point. The source
distribution includes tests and the synthetic example. Version metadata has a
single source in `src/heatmap_decoder/version.py`. See [CHANGELOG.md](CHANGELOG.md).
The project has no PyPI publication configured; install this checkout locally.
A pushed `v*` tag publishes the wheel and source archive only after the entire cross-platform CI matrix passes; the tag
must match the package version. The release job uses GitHub Actions' built-in
repository token, with write access limited to that job.
