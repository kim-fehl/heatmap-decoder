# M15 bounded orientation evidence

`preregister.json` fixes two partitions of 24 base layouts each: 0/±90°, above/below,
DejaVu Sans/Mono, viridis/teal-black-red. Each base has clean/JPEG85/blur0.65
variants kept in its partition (72 ordinary attempts), plus eight challenges.
Font hashes and source LUTs accompany truth. Before recognition, layout QA checks
that all ordinary text patches fit their columns and avoid the matrix.

Numeric seeds differ between partitions. Fonts, strings and layout families are
shared; derivatives are correlated. This is another synthetic partition, not
independent text/layout or publication-image generalisation.

`preregister-initial.json` retains the initial design. Before policy freeze, the
nonlinear challenge was corrected to render non-affine tick values and matching
reference values. The capability sampling plan runs numeric, explicit-angle
labels and region/orientation on all derivatives; OCR scales, composed workflows
and unassigned-text counts run on the 24 clean bases plus all challenges. No
validation figure was recognised before `freeze.json`.

    uv run --extra ocr --group benchmark python benchmarks/m15/evaluate.py development
    uv run --extra ocr --group benchmark python benchmarks/m15/evaluate.py validation

Evaluators refuse to overwrite retained reports. The validation evaluator checks
the complete decoder/evaluator/fixture/preregistration hashes against the freeze.
Truth is consulted for scoring after runtime selection. Supplied geometry is an
explicit separate conditional mode. Full synthetic images retain distractors;
there is no silent pre-crop of whole-figure numeric/composed attempts.

Report fields separate supplied numeric, inferred columns, OCR scale, conditional
legend, explicit-angle labels, orientation proposals, count evidence without
slot fitting, and composed requested identifiers. Cap failures, conflicting
signs and text errors remain in evidence. Identifiers, inferred dimensions and
OCR scales never gain experimental export permission.

The nonlinear challenge deliberately measures an unsupported scale as well as
what a forced linear endpoint assumption would do. Such endpoint-only values
cannot establish biological numeric accuracy. Use the OCR-scale/composed review
results when assessing a supported workflow; never treat a nonlinear bar as
linear just because its endpoint numbers are known.

[The publication manifest](publications.json) records source URLs, article licence
checks and missing annotations. The user will add real-world figures directly to the repository; **zero
publication-image trials** have run. No assets are redistributed.
For those figures, retain full images, record resolution
and every crop, annotate ROIs/counts/angles/ticks/scale and audit annotations before
running separate conditional and whole-figure modes. Unavailable numeric truth
must remain explicitly unavailable.

[The reviewed example](../../examples/m15/README.md) demonstrates angle/name review
without changing numeric values. See [orientation documentation](../../docs/orientation-proposals.md).

Run local audited figure annotations with:

    uv run --extra ocr python benchmarks/m15/publication_trials.py --annotations PATH_TO_ANNOTATIONS.json --out work/m15-publication-results.json

Each trial requires `id`, `publication_url`, `licence`, `image_path`, `image_sha256`,
`second_pass_audited`, `scale_model` and a decoder `config`. Record panel ID and
preprocessing alongside these fields. Unknown/nonlinear scales are review-required,
not silently replaced by linear endpoints. The optional retrieval mode uses the
normal proxy and leaves assets in `work/`; it is unnecessary for repo-added images.

A final I/O review guard was added after synthetic recognition to reject a selected
angle conflicting with an explicit extraction angle. freeze-validation.json and
post-validation-review-guard.patch retain the exact original source; freeze.json
checks current code. This does not change OCR, orientation, counts or numbers.
[The M15 report](../../docs/progress/M15.md) reports actual coverage and the forced
linear/nonlinear challenge error instead of claiming it as a safe numeric export.
