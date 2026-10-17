# M13 conditional explicit-angle OCR evaluation

Run: uv run --extra ocr python benchmarks/m13/evaluate.py.

All 44 attempts remain in report.json: eight clean supported runs (±90°,
above/below, slots/sparse), and 36 underscore/hyphen challenges (±90°,
above/below, three fonts, clean/JPEG quality 85/blur sigma 0.65).

Each attempt has eight sample strings. ROI/counts/angle are explicitly supplied
to the decoder. Truth scores recognition afterwards. This is synthetic
development evidence, not automation coverage or unseen-source validation.

Raw/accepted exact strings, false accepted text, unresolved counts and geometric
slot assignment are reported separately. Geometric assignment means one candidate
inside the renderer's original column band; it does not certify spelling.
Font hashes, engine/dependency versions, generator version and timings remain
in the report. Reviewed numerical reconstruction is tested separately.
No M14/M15 palette or automatic orientation work is included.

## Padding probe requested after implementation

Run: uv run --extra ocr python benchmarks/m13/padding.py.

[Padding report](padding-report.json) retains the bundled +90° clean Sans example
with original per-slot crops expanded by 0/3/5/8 pixels on every side, and a
separate artificial white-border trial with the equivalent padding after upscale.
Each trial tests all eight underscore-containing strings. Exact raw strings
remain 0/8 for every padding choice; original-coordinate assignment stays 8/8
for every expanded-crop trial. The results are unchanged from the unpadded case.

The underscores are already inside the source crops. This finite probe indicates
a recognition/segmentation weakness rather than clipping for this fixture.
No default crop expansion or automatic spelling repair was introduced. Artificial
border trials measure text only; their shifted coordinates are not exported
as production assignment evidence.
