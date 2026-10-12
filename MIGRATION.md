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
