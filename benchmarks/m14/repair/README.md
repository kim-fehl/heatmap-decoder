# M14 geometry repair evidence

The original M14 validation was inspected to diagnose its low coverage. Both
original partitions are development regression data for this repair. Their
retained reports remain unchanged.

`freeze.json` records the complete decoder source hashes, unchanged numeric
threshold definitions, fixture/evaluator hashes, and all new seed-partition cases
before recognition. The new partition uses the same synthetic generator and
four correlated layouts, with numeric matrix seeds 714000–714003 and challenge
seed 814000. This tests another draw of the same generator, not independent
publication generalisation or new layout coverage.

Reproduction (evaluators refuse to overwrite retained reports):

    uv run --group benchmark python benchmarks/m14/repair/evaluate.py development
    uv run --group benchmark python benchmarks/m14/repair/evaluate.py validation
    uv run --group benchmark python benchmarks/m14/repair/evaluate.py new-seeds

`validation.json` here is explicitly an inspected development regression;
`new-seeds.json` is the frozen additional check. Every ordinary failure stays in
the denominator, and each report retains truth, candidates, QC and export status.
The seven original negative families and both structural challenges accompany
each partition. Ten focused tests in `tests/test_m14_repair.py` cover final
strokes, external frames, blur/JPEG measurements, wrong counts, irregular
spacing, wide strokes, strong blur and smooth within-cell structure.

`corrections-regression.json` repeats the earlier supplied-count/scale historical
regression using `benchmarks/m14/regression.py`'s evaluator, writing to this new
path. These older cases are also inspected development data.

The original M14 policy-freeze files describe the historical runtime and therefore
will not match the updated decoder. Their original report checksums remain valid.
See [repair report](../../../docs/progress/M14-repair.md) for results and limitations.

`runtime.patch` records the exact decoder change from pre-repair M14. Reverse it
in an isolated checkout to reconstruct that historical runtime for investigation
reproduction. It is not necessary to apply the patch to the current source.
