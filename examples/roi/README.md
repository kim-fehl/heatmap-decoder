# M7 coordinate review

See [docs/progress/M7.md](../../docs/progress/M7.md) for commands, exact schemas, settings and limitations.
`hints.json` requests the heatmap in `../figure.png`, with manual colourbar
coordinates and user-supplied endpoints. `detect` exits 5 after writing review
artefacts. Inspect the overlay, then explicitly select `heatmap-001` or edit the
proposed config. No candidate is automatically inserted into extraction config.

`generate.py` recreates the six synthetic PNG fixtures and coordinate annotations.
Base layouts and splits are frozen in its `FIXTURES` table. `benchmark.py` reports
all clean, blur, JPEG, negative and multipanel attempts. `benchmark.json` records
actual results from detector `m7.1`; JPEG and blurred-border failures are retained.
These figures/code are original MIT material. Degraded variants are generated in
memory and remain in the same split as their base. No real-figure accuracy claim
or automatic hand-off is made.
