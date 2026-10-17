# Test ownership and execution

Tests are organized by functionality within four execution tiers. Add regressions
for an existing behavior to its owning file. A milestone or issue is context for a
case, not a reason to create another permanent module. Temporary development files
must be consolidated and removed before the change is finished; see
[repository instructions](../AGENTS.md#maintain-tests-by-functionality).

| Tier | Scope | Collected cases |
|---|---|---:|
| `unit` | Focused component contracts and property tests; external engines mocked | 243 |
| `pipeline` | Deterministic synthetic images, file adapters, CLI and composed workflows | 601 |
| `ocr` | Real local engine/model execution, synthetic and original publication images | 47 |
| `qualification` | Benchmark machinery, frozen evidence and independent publication references | 36 |
| **Total** | Parameterized pytest cases, including opt-in OCR | **927** |

Counts reflect the current suite after removing the v0.1 import-alias test;
the independent lazy OCR import check now belongs to `unit/test_core.py`.
Counts are an inventory, not a quota. Hypothesis examples and
unittest subtests run within collected cases. OCR cases remain visible in collection
and are skipped unless explicitly enabled. Four numeric publication cases formerly
hidden by a module-wide OCR skip now run with qualification tests.

## Commands

```sh
uv sync --locked --group benchmark
uv run pytest -q                          # all deterministic tiers; OCR skipped
uv run pytest -q tests/unit               # focused component contracts
uv run pytest -q tests/pipeline           # synthetic/file/CLI workflows
uv run pytest -q tests/qualification      # evaluation integrity and references
uv run pytest -q -m 'unit or pipeline'     # tiers can also be selected by marker
uv run pytest --collect-only -q           # inventory without executing cases
```

For real OCR, install the Tesseract engine/English data where required by existing
cases, then run:

```sh
uv sync --locked --extra ocr --group benchmark
HEATMAP_DECODER_OCR_TESTS=1 uv run pytest -q tests/ocr
```

`tests/conftest.py` assigns the directory's tier marker, the existing
`ocr_integration` marker, and the central OCR opt-in skip. `-m ocr_integration`
remains supported. Run `uv run ruff check .` and `uv run mypy src` as in CI.
Distribution checks stay in `tests/smoke_distribution.py`; run `uv build` before
`uv run python tests/smoke_distribution.py`.

Qualification tests verify evaluation code, references and frozen artifacts. They
do not requalify the current production implementation or rewrite historical
hashes. Full qualification remains an explicitly registered run of the relevant
`benchmarks/` evaluator; see [the evidence index](../benchmarks/README.md).

## File ownership

Paths below are relative to `tests/`. Where a capability spans tiers, the engine
boundary or evaluation purpose decides which file owns the case.

### Unit contracts

| File | Behavior owned | Cases |
|---|---|---:|
| [test_core.py](../tests/unit/test_core.py) | Configuration, regular geometry, sampling, calibration, raster validation and lazy OCR loading | 28 |
| [test_label_characters.py](../tests/unit/test_label_characters.py) | Scientific characters, raw text preservation, suggestion policies | 8 |
| [test_label_geometry.py](../tests/unit/test_label_geometry.py) | Coordinate transforms, angle schemas, pixel/slot ownership | 70 |
| [test_labels.py](../tests/unit/test_labels.py) | Token grouping, assignment, missing/duplicate/colliding candidates | 15 |
| [test_legend.py](../tests/unit/test_legend.py) | Strict numeric grammar, spatial fitting and scale eligibility contracts | 68 |
| [test_piecewise_calibration.py](../tests/unit/test_piecewise_calibration.py) | Piecewise knots, interpolation, confidence and export scope | 14 |
| [test_properties.py](../tests/unit/test_properties.py) | Generated geometry and calibration invariants (Hypothesis) | 2 |
| [test_quality.py](../tests/unit/test_quality.py) | Colour distance, ambiguity, missing data, QC and exports | 12 |
| [test_rapidocr.py](../tests/unit/test_rapidocr.py) | RapidOCR defaults, dispatch, worker lifecycle, language/model metadata | 15 |
| [test_tesseract.py](../tests/unit/test_tesseract.py) | TSV parsing, dispatch, missing dependencies and engine errors | 11 |

### Synthetic and application pipelines

| File | Behavior owned | Cases |
|---|---|---:|
| [test_automation.py](../tests/pipeline/test_automation.py) | Composed numeric/geometry/identifier state and retained partial output | 9 |
| [test_cli.py](../tests/pipeline/test_cli.py) | Help, errors, exit codes, overwrite rules and bundled numeric example | 12 |
| [test_detection.py](../tests/pipeline/test_detection.py) | ROI proposals, manual bounds, saved reviews and protected inputs | 38 |
| [test_detection_policy.py](../tests/pipeline/test_detection_policy.py) | Runtime publication policy, diagnostics and review-only handoff | 5 |
| [test_detection_regressions.py](../tests/pipeline/test_detection_regressions.py) | Bar cores, count competition, renderer QA and export regressions | 25 |
| [test_gridlines.py](../tests/pipeline/test_gridlines.py) | Gridline centres, terminal extents, degradation and adverse geometry | 10 |
| [test_group_labels.py](../tests/pipeline/test_group_labels.py) | Categorical swatches, unequal groups, repeated labels, independent gene/sample entities | 23 |
| [test_label_geometry.py](../tests/pipeline/test_label_geometry.py) | Rotated/arbitrary-angle lines, attachment order, raw evidence and review gates | 112 |
| [test_labels_io.py](../tests/pipeline/test_labels_io.py) | Label artifacts, file sources, provenance and input protection | 9 |
| [test_layout_detection.py](../tests/pipeline/test_layout_detection.py) | Omics layout matrix, count/ROI association and staged localization | 62 |
| [test_legend.py](../tests/pipeline/test_legend.py) | Numeric grammar, anchors, scale fitting, review and export decisions | 39 |
| [test_ocr_fields.py](../tests/pipeline/test_ocr_fields.py) | Numeric/label crop padding, shared bands, retries and call budgets | 23 |
| [test_orientation_search.py](../tests/pipeline/test_orientation_search.py) | Bounded angle/side search, competing directions and reviewed angles | 14 |
| [test_palettes.py](../tests/pipeline/test_palettes.py) | Palette independence, dark data preservation and ambiguity negatives | 28 |
| [test_piecewise_calibration.py](../tests/pipeline/test_piecewise_calibration.py) | Piecewise knots, interpolation, confidence and export scope | 5 |
| [test_quality.py](../tests/pipeline/test_quality.py) | Colour distance, ambiguity, missing data, QC and exports | 16 |
| [test_range_scale.py](../tests/pipeline/test_range_scale.py) | Range-valued extraction, open bounds, plateau/cross-band ambiguity and CLI output | 12 |
| [test_reconstruction.py](../tests/pipeline/test_reconstruction.py) | Synthetic numeric reconstruction across palettes and cell sizes | 12 |
| [test_reviewed_extract.py](../tests/pipeline/test_reviewed_extract.py) | Label review, numeric invariance, matching QC labels and export gates | 21 |
| [test_roi_association.py](../tests/pipeline/test_roi_association.py) | Colourbar/label association, counts, saved evidence and review conflicts | 36 |
| [test_scale_reconstruction.py](../tests/pipeline/test_scale_reconstruction.py) | Range/layout matrix, anchors, inferred axes and capability composition | 51 |
| [test_segmented_grids.py](../tests/pipeline/test_segmented_grids.py) | Segmented-grid reconstruction, block gaps, ordering and contradictions | 39 |

### Real OCR

| File | Behavior owned | Cases |
|---|---|---:|
| [test_count_inference.py](../tests/ocr/test_count_inference.py) | Real-engine label geometry and measured-count completion | 2 |
| [test_labels.py](../tests/ocr/test_labels.py) | Token grouping, assignment, missing/duplicate/colliding candidates | 6 |
| [test_layout_detection.py](../tests/ocr/test_layout_detection.py) | Omics layout matrix, count/ROI association and staged localization | 2 |
| [test_legend.py](../tests/ocr/test_legend.py) | Numeric grammar, anchors, scale fitting, review and export decisions | 1 |
| [test_orientation_search.py](../tests/ocr/test_orientation_search.py) | Bounded angle/side search, competing directions and reviewed angles | 4 |
| [test_publication_headers.py](../tests/ocr/test_publication_headers.py) | Original-image group headers, signs, tiny identifiers and CLI artifacts | 10 |
| [test_publication_labels.py](../tests/ocr/test_publication_labels.py) | Original-image OCR compared with reviewed gene labels | 1 |
| [test_rapidocr.py](../tests/ocr/test_rapidocr.py) | RapidOCR defaults, dispatch, worker lifecycle, language/model metadata | 3 |
| [test_roi_association.py](../tests/ocr/test_roi_association.py) | Colourbar/label association, counts, saved evidence and review conflicts | 1 |
| [test_rotated_labels.py](../tests/ocr/test_rotated_labels.py) | Real engines on rotated slots and reviewed numeric invariance | 17 |

### Benchmark and publication qualification

| File | Behavior owned | Cases |
|---|---|---:|
| [test_detection_evidence.py](../tests/qualification/test_detection_evidence.py) | Frozen corpus hashes, denominators, error summaries and release gates | 6 |
| [test_geometry_metrics.py](../tests/qualification/test_geometry_metrics.py) | Boundary tolerances, exact counts and group membership scoring | 5 |
| [test_publication_sources.py](../tests/qualification/test_publication_sources.py) | Workbook/reference integrity, independent annotation axes and numeric image correspondence | 13 |
| [test_publication_trials.py](../tests/qualification/test_publication_trials.py) | Publication annotation audit and unavailable-scale abstention | 2 |
| [test_validation.py](../tests/qualification/test_validation.py) | Fresh partitions, immutable freezes, source/runtime checks and evaluator anchors | 10 |

## Fixtures and parameter matrices

`tests/support/synthetic.py` provides the canonical layout, legend, range and
palette constructors. They call existing generators and return fresh data per
call. Domain support modules share anchor builders, crop configurations, quality
images and fixed angle lists across tiers. `tests.support.ROOT` resolves repository
paths once; `load_script` gives evaluator tests independent namespaces so patched
policy/runtime functions cannot leak between tests. Test modules never import
other test modules for fixtures. Keep the example/benchmark renderers and their
frozen artifacts unchanged during test maintenance.

| Matrix | Conditions | Cases |
|---|---|---:|
| Arbitrary-angle attachment | 17 angles × 2 label sides × regular/segmented grid | 68 |
| Affine round trips | 19 angles × 2 upscale factors | 38 |
| Omics layouts | 2 row-label sides × 2 column-label sides × 4 bar sides × 2 bar orientations | 32 |
| Range reconstruction | 3 layout families × 3 matrix sizes × 5 ranges, with direction/side assigned by index | 45 |
| Linear fitting | 2 orientations × 4 anchor counts × 2 directions | 16 |
| Real rotated OCR | 2 engines × 2 rotations × 2 placements × 2 recognition modes | 16 |
| Segmented reconstruction | 4 segment families × clean/JPEG/blur | 12 |

Omics matrix size, palette and gridlines cycle within its 32 cases; they are not
additional independent factors. Range direction, bar side and gridlines likewise
vary by index rather than forming a full Cartesian product.

## Consolidation and overlap review

The test-layout migration retained all **913 test names/parameter IDs** and all **1,229 Python
`assert` statements unchanged. Existing unittest assertions and subtests are also
retained. No test was deleted as a duplicate during that migration. The subsequent v0.1
interface removal deleted its alias-compatibility case; current counts above
reflect that intentional removal. Related former milestone modules
now share behavior owners; real engines and benchmark machinery have separate tiers.

| Apparent overlap | Distinct behavior retained |
|---|---|
| Orthogonal and arbitrary-angle transforms | Exact pixel transposes/half-open boxes versus floating affine polygons and expanded crops |
| Missing/duplicate labels across assignment paths | Sparse tokens, owned slots and oriented lines exercise different association interfaces |
| Review gates in extraction and automation | Direct API/writer rejection versus composed partial-result retention and CLI exit behavior |
| Group-label regressions | Colour swatches versus unequal line spans, repeated categories, multiple tracks and OCR budgets |
| Frozen benchmark checks and reconstruction cases | Evidence integrity/denominators versus numerical accuracy and per-image abstention |

Before removing future duplicates, compare their fixtures, parameter values,
assertions, failure modes and entry points, then record which surviving case
protects each contract. Shared setup or similar names alone do not justify deletion.
