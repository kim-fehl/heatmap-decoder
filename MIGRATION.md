# Naming migration (M4)

The project and distribution are now `heatmap-decoder`, the primary command is
`heatmap-decoder`, and the implementation and Python imports use
`heatmap_decoder` in `src/heatmap_decoder/`. The ordinary checkout directory
remains `heatmap-decoder`. Install the development checkout with `uv sync --locked`
or `python -m pip install -e .`.

Replace `from heatmap2matrix import ...` with `from heatmap_decoder import ...`,
and `python -m heatmap2matrix` with `python -m heatmap_decoder`. Existing public
library functions and numeric CLI options retain their signatures and behavior.
Thin deprecated import/submodule and command aliases preserve v0.1 callers;
they delegate to the same implementation. Imports emit a `DeprecationWarning`,
and the old command emits a visible `FutureWarning`.

Input files, numeric configuration keys, and existing output filenames need no
renaming. Numeric CSV remains headerless and quality CSV remains long-form.
Provenance records the new software name and the optional `labels` configuration.
No text-file label API existed in the inspected baseline, so there is no old
label option to map. OCR and file label sources now have validated schemas;
The `labels` command accepts OCR and file sources and produces review artifacts.
In v0.2, `extract` integrates these sources and strict review import. With labels
requested, matrix CSV has a gene index/sample header and quality CSV gains matching
label fields. Omitted label configuration retains v0.1 numeric CSV formats.
Unresolved labels write review artifacts and block numeric publication with exit 4.
See `README.md` and `docs/progress/M6.md` for the full workflow. The file adapter now enforces
explicit `overwrite=True`, matching the CLI's established overwrite policy.

The `0.2.0.dev4` and `0.2.0.dev5` versions identify milestone development code,
not a completed v0.2 release. Historical v0.1 artifacts retain their original names.
M6 sets package metadata to `0.2.0`; a tagged release is handled separately by CI.

## M17 segmented grids

Existing regular-grid configurations and numerical results are preserved. Optional
`rows_segments` / `columns_segments` describe original-coordinate `{start,end,count}`
groups. Totals may be inferred from explicit segments; contradictions are errors.
Numeric and label provenance now include the same resolved grid hash. File extraction
also writes `grid_diagnostic.png`. Detector decisions use `blocked-acceptance.1`;
old frozen reports retain their original policies. Reviewed segment configuration
is authoritative. See [blocked grids](docs/blocked-grids.md) and the
[M17 report](docs/progress/M17.md) before relying on automatic proposals.

## M18 arbitrary angles

Finite numeric sample angles now normalize modulo 360 to `[-180,180)`; integral
floats are accepted. Boolean/string/non-finite angles and rotated gene labels remain
errors. Existing 0/±90° paths keep exact transposes. `recognition: "lines"` or any
other angle uses complete oriented lines and matrix-side attachment ownership. New
line-based identifiers have `unreviewed` status and require a complete review TSV
before labelled extraction, regardless of OCR confidence. New provenance includes
polygons, baselines, attachments, line crops and affine inverses. SciPy is now an
explicit dependency (previously transitive through scikit-image). See
[configuration](docs/arbitrary-angle-labels.md) and [M18 status](docs/progress/M18.md).

M19 `automate` reports separate numeric/label-geometry/identifier completion.
Default diagnostics remain unchanged. Eligible experimental composed requests
retain `numeric_positional/numeric_positional.csv` and return 5 while names need
review; partial numeric results use `numeric_partial.csv`. Positional IDs are
explicitly marked and never presented as OCR names. OCR scales keep their
physical anchors/span instead of being converted into manual endpoints. A saved
OCR diagnostic config still needs legend review for ordinary `extract`.
