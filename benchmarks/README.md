# Validation evidence

Benchmarks measure conditional reconstruction accuracy and abstention. They do not
establish universal OCR accuracy or recover the data transformations that preceded
plotting. Reference matrices and label transcriptions are used for evaluation only.

| Evidence | Scope |
|---|---|
| [Current M16 acceptance](m16/validation-rapidocr-v1/README.md) | Fresh, pre-registered synthetic partition; current source/runtime/model hashes; numeric and label coverage, errors, gates and runtime |
| [Current publication trial](m16/validation-rapidocr-v1/publication.json) | All five workflows on the inspected 33×8 publication crop; development evidence |
| [M16 evidence index](m16/README.md) | Initial trials and OCR follow-ups, clearly separated from current validation |
| [Automation](m12/README.md) | Earlier composed-workflow evaluation and export scope |
| [Palette and orientation investigations](../docs/progress/M14.md) | Historical ambiguity and geometry evidence |

Read [M16 status](../docs/progress/M16.md) for consolidated conclusions and the
[usage guide](../docs/usage.md) for operational review requirements. Historical
freezes bind their original implementations; source mismatch is intentional and
must not be bypassed by rewriting completed freezes. New changes require a new
registered partition or an explicitly scoped regression run.

[M19](m19/README.md) records fresh full-workflow evidence, scoped qualification and retained failures.
