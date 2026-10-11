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
labelled reconstruction will be integrated in M6. For M4, an extraction with a
requested label source fails explicitly instead of silently ignoring it.

The `0.2.0.dev4` version identifies this milestone's development code, not a
completed v0.2 release. Historical v0.1 artifacts retain their original names.
