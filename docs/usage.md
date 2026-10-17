# Detailed usage and manual overrides

For a short recognition overview, see the [README](../README.md).
For numeric tick recognition and scale review, see [legend usage](legend.md).

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

The [bundled figure](../examples/figure.png) is a clean synthetic 20 × 20 diverging
heatmap with known [truth](../examples/truth.csv) and [configuration](../examples/config.json).
Its generator uses a fixed seed and an independent analytic palette. Regenerate it
with `uv run python examples/generate.py`.

![Synthetic example heatmap and colourbar](../examples/figure.png)

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

Gene/sample OCR defaults to **RapidOCR**, installed with the main package,
including the local **English PP-OCRv4 recognizer** (`language="eng"`). It uses
native-size grayscale input by default (`labels.ocr.upscale_factor=1`), with no added white border. No image data
is sent to an external OCR service. Explicit slot recognition uses each complete
field; sparse recognition uses RapidOCR's detector. Quarter-turn correction and
fixed slot ownership still apply. The [language and character policy guide](ocr-language.md)
explains model selection, Greek identifiers and auditable cleaned suggestions.

To select the legacy label engine, set `labels.ocr.engine` to `"tesseract"`.
Tesseract defaults to 3× enlargement unless explicitly overridden. Colourbar OCR
also defaults to English RapidOCR, with native-size fields and two pixels of white
padding. Its raw strings pass the strict numeric parser; it has no decoder whitelist.
To use the legacy numeric engine, set `legend.ocr.engine="tesseract"` (or
`scale.legend.ocr.engine` in an extraction config). For Tesseract paths, install the optional wrapper with `uv sync --locked --extra ocr` or
`python -m pip install -e '.[ocr]'`, and install the engine separately:

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
may have a regular or explicitly [blocked grid](blocked-grids.md). Regions may be
left/right for rows and above/below for columns. Rows must be upright single-line text. Columns also support explicit
±90° quarter turns; see [rotation signs and examples](rotated-labels.md).
Other finite column angles use [complete-line geometry](arbitrary-angle-labels.md)
and require identifier review. Dendrogram reconstruction and semantic axis inference
remain unsupported. Reviewable ROI proposals and M10 development decisions
are described below; they do not establish identifier correctness.

Add this `labels` object to a complete numeric configuration (the regions below
are coordinate examples; adapt them to your own image):

```json
{
  "labels": {
    "rows": {"source": "ocr", "bbox": [510, 80, 650, 480]},
    "columns": {"source": "ocr", "bbox": [100, 35, 500, 75]},
    "ocr": {
      "language": "eng",
      "engine": "rapidocr",
      "upscale_factor": 1,
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

RapidOCR runs on CPU in a reusable isolated worker. A timeout stops the worker;
model hashes, engine/runtime versions, geometry type and score semantics are
retained in provenance. Slot boxes describe source fields, not detected glyphs;
synthetic white padding is excluded from those source boxes. Sparse boxes describe
detected text. Recognizer scores are multiplied by 100 for the existing confidence
rules, and are explicitly marked uncalibrated and engine-specific. The bundled
model currently supports the `eng` setting; use Tesseract for other language packs.

The legacy engine uses grayscale, bicubic upscaling, and one `--oem 1 --psm 11`
sparse-text pass (PSM 7 for slots).
Geometry assigns candidates to fixed grid bands without shifting missing slots
or choosing collision winners by confidence. Confidence is a heuristic, not a
probability or a guarantee of biological identity. Case, digits, punctuation and
internal spaces are preserved. Low-confidence, missing, spanning or conflicting
candidates remain unresolved. Inspect **all** OCR labels, including accepted ones.
The M16 publication trial recovers `EsGalectin1` and 26/33 exact genes, but six
incorrect gene names still pass the confidence threshold. RapidOCR can also lose
internal spaces in rotated sample labels; these observations are preserved rather
than corrected from a dictionary. See the [default-engine trial](../benchmarks/m16/rapidocr-default/README.md).

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
`examples/labels/reconstruction.png`. See [docs/progress/M6.md](progress/M6.md) for API and validation details.

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
v0.2 example. See [M5 measurements](progress/M5.md) and [M6 validation](progress/M6.md) for the
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
identity; labels need human verification. Explicit column-label quarter turns are
supported; arbitrary angles, PDF support, web UI and RNA-seq count recovery are not.

MIT licensed. Input figures remain copyrighted by their respective owners.


## Release packaging

```bash
uv build
uv run python tests/smoke_distribution.py
```

The wheel includes the typed Python package and console entry point. The source
distribution includes tests and the synthetic example. Version metadata has a
single source in `src/heatmap_decoder/version.py`. See [CHANGELOG.md](../CHANGELOG.md).
The project has no PyPI publication configured; install this checkout locally.
A pushed `v*` tag publishes the wheel and source archive only after the entire cross-platform CI matrix passes; the tag
must match the package version. The release job uses GitHub Actions' built-in
repository token, with write access limited to that job.

## Review-only heatmap detection (M7)

M7 adds deterministic heatmap proposals from manually supplied row/column counts.
Detection scores are heuristics, separate from OCR confidence and numerical QC.
Colourbars, label localisation and count inference remain manual in this milestone.

```bash
uv run heatmap-decoder detect examples/figure.png --config examples/roi/hints.json --out work/m7-roi
```

Expected exit **5** saves `roi_candidates.json`, `roi_diagnostic.png` and
`config_proposed.json`; inspect them before choosing bounds. Candidates are never
selected by rank. After reviewing this bundled reference:

```bash
uv run heatmap-decoder extract examples/figure.png --config work/m7-roi/config_proposed.json --roi-review work/m7-roi/roi_candidates.json --heatmap-candidate heatmap-001 --out work/m7-results
```

Minimal hints are `{"rows": 20, "columns": 20}`. To preserve a fixed heatmap,
add `"heatmap_bbox": [24, 32, 264, 272]` and `"requested": []`. Optional fixed
colourbar fields use the existing extraction names; no endpoints are inferred.
`detector` accepts the settings documented in [docs/progress/M7.md](progress/M7.md). Unsupported requested
kinds and unknown fields fail validation. Missing required extraction fields remain
absent in proposed config and are listed in detection metadata.

You can instead edit the proposed config to add reviewed coordinates, then use
ordinary `extract --config` without rerunning detection. Include `--roi-review`
to validate the saved image identity and retain evidence in provenance. Conflicting
manual bounds/counts and unknown candidate IDs fail. Existing outputs require
`--overwrite`. Manual detection exits 0 when requested geometry is resolved, even
if extraction still needs a colourbar or endpoints; invalid input exits 2.

The six fixed clean regression layouts have exact top-three bounds. JPEG and some
blurred borders fail or abstain, and multiple panels remain ambiguous. See the
[complete M7 report](progress/M7.md) and [reproducible fixtures](../examples/roi/README.md) for
actual proposal quality and unsupported layouts. Automatic extraction is unavailable.

## Colourbar, label-region and count proposals (M8)

M8 extends detection with horizontal/vertical continuous-bar proposals, separate
heatmap–bar associations and optional local OCR searches for upright label regions.
Fixed coordinates/sources support hybrid workflows. Counts may be omitted only
with explicit `infer_rows` / `infer_columns` requests; supplied counts remain
authoritative. Count alternatives record original-resolution observations,
pitch/phase, missing support, conflicts and integer-multiple ambiguity. They are
never automatically accepted or inserted into extraction config.

```bash
uv run --extra ocr heatmap-decoder detect examples/labels/reconstruction.png --config examples/roi_m8/hints.json --out work/m8-roi
# Expected exit 5. Inspect the overlay and saved evidence before selecting IDs.
uv run --extra ocr heatmap-decoder extract examples/labels/reconstruction.png --config work/m8-roi/config_proposed.json --roi-review work/m8-roi/roi_candidates.json --roi-selections examples/roi_m8/selections.json --out work/m8-results
```

`--roi-selections` accepts explicit `regions` and `counts` ID mappings. Existing
`--heatmap-candidate`, edited ordinary configs and manual label files still work.
Selecting a label region preserves the ordinary OCR/review gates; it does not
accept the text. Missing requested OCR/regions save unresolved artefacts and stop
saved-review extraction. Colourbar/count operation remains independent of OCR.
Numerical endpoints are never inferred. See [docs/progress/M8.md](progress/M8.md) and the
[runnable M8 fixtures](../examples/roi_m8/README.md) for exact hints and review formats.

The fixed clean bar layouts have exact offered bounds, while the original M8
blur/JPEG evaluation abstained. M10 improves these proposals, as documented below. Merged neighbouring colours produce competing counts
with a misleading first rank. These are review-only proposals with documented
failures. M10 adds development decisions; M12 release validation remains future work.

## Historical v0.3 validation and hand-off policy (M9)

Automatic ROI/count acceptance remains disabled: the required gates are not
established. All proposals require explicit coordinate/count review. Selecting a
label region does not verify its text. The new frozen 36-figure synthetic benchmark
records every case and separates localisation, conditional counts, OCR exact text
and downstream numerical error. It contains no published real-figure set and
makes no claim of practical accuracy on arbitrary scientific figures.

The reviewed comparisons match manual extraction in 18 cases. The OCR challenge
also records 24 wrong strings among 48 accepted, so verify high-confidence strings.
See [docs/progress/M9.md](progress/M9.md) and the [frozen corpus/report](../benchmarks/m9/README.md) for all
failures, edge tolerances, count ambiguity, provenance and manual override rules.

```bash
uv run python benchmarks/m9/workflow.py --out work/m9-example
uv run --extra ocr python benchmarks/m9/evaluate.py --ocr --out work/m9-benchmark.json
```

Manual-only extraction, optional OCR, file labels and the v0.2 review workflow
remain supported. M11 tick OCR and explicit scale review are documented in
[legend usage](legend.md). Automatic hand-off and biological meaning inference
remain unavailable; manual scales remain authoritative.


## M10 development decisions

`detect` writes schema 3 evidence with per-ROI, association, count, scale and numeric
QC decisions. `accepted` here means the versioned **development checks** passed;
`automatic_export_allowed` is still false. No saved confidence, policy JSON or
benchmark result enables a released capability. Old schema 1/2 evidence remains
readable through the same explicit review workflow.

For an ordinary **30-gene × 10-sample** example with genes on the right, samples
below and a vertical bar above:

```bash
uv run heatmap-decoder extract examples/m10/figure.png \
  --config examples/m10/config.json --out work/m10-manual
uv run heatmap-decoder detect examples/m10/figure.png \
  --config examples/m10/hints.json --out work/m10-review
# detect exits 5: inspect its overlay and decisions; review coordinates explicitly.
uv run heatmap-decoder extract examples/m10/figure.png --auto-roi --numeric-only \
  --config examples/m10/hints.json --out work/m10-experimental
# exits 5, writes roi_review/, and does not publish a matrix.
```

`--numeric-only` explicitly requests positional identifiers for the experimental
path. It conflicts with requested gene/sample identities. OCR suggestions remain
review-required even when geometric assignment and heuristic recognition pass.
Supplied counts and scale endpoints are authoritative. Count ranking never
establishes exact dimensions; inferred count decisions remain review-only.

The [M10 fixtures](../examples/m10/README.md) cover 20–30 genes and 6–10 samples,
left/right genes, above/below samples, and both bar orientations on every side.
Noise-aware bar refinement now offers reviewable ordinary blur/JPEG bars. Strict
outer-bounds and numeric QC checks still abstain on many degraded images. These
are synthetic development results, with repeated matrices across layouts, not
independent real figures or M12 release evidence. Tick recognition and scale review are now covered in [legend usage](legend.md).

## M12 selective requests

See [automation and slot recognition](automation.md) for composable requests,
per-capability reports, exit status 5 and detailed override schemas. The README
uses the recognition workflow; manual extraction and reviews remain available.

M14 adds [observed-palette and candidate-component QC documentation](palettes.md)
and [worked numeric examples](../examples/m14/README.md). Numeric thresholds and
identifier/experimental export scope remain unchanged.
