# M17 example

The fixture is development evidence: 10 rows × 8 columns, unequal groups and gaps
on both axes, with one reviewed linear legend. `truth.json` is evaluation-only.

From the repository root:

```sh
uv run heatmap-decoder extract examples/m17/figure.png --config examples/m17/config.json --out work/m17/explicit
uv run heatmap-decoder detect examples/m17/figure.png --config examples/m17/hints.json --out work/m17/proposals
uv run heatmap-decoder automate examples/m17/figure.png --request examples/m17/request.json --out work/m17/automatic
```

The detection command exits 5 while proposals await review.

The explicit path writes one numerical matrix and source interval overlay. The
automatic path resolves geometry and counts but remains diagnostic: M19 is the
separate inferred-count export gate. Its abstention exit code is 4, with all
geometry/decision evidence retained. Inspect `automation.json`, then use its
`diagnostic_config` as a concrete reviewed extraction configuration if appropriate.

See [configuration details](../../docs/blocked-grids.md). No label spelling is
inferred or authorised by this example.
