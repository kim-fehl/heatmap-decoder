# Problems observed on the supplied publication examples

Measured with the versions and image hashes in `evaluation.json`. Run
`build_tables.py` and `evaluate.py` to reproduce. These manually inspected fixtures
are development examples; reviewed regions and label files are supplied.

| Example | Result | Remaining problem |
|---|---|---|
| DeJesus Figure 2G, tall 20 × 3 screenshot | 60/60 numeric cells, RMSE 0.176 log2 FC, maximum error 0.325 | Manual endpoint calibration and raster color error; not exact recovery |
| Figure 2G native RapidOCR | 9/20 accepted exact; 18/20 raw exact; 11 unresolved; 0 accepted wrong | `WIPI2` → `IPI2`, `SBDS` → `BDS`; nine other correct raw strings fail confidence filtering |
| Figure 2G RapidOCR enlarged 2× / 3× | Both 10/20 accepted, 18/20 raw exact | Enlarging alone does not fix the recognition/confidence problem |
| Li Figure 5E | 618/686 tick-calibrated cells, RMSE 0.052; OCR 44/49 accepted exact | 13 ambiguous + 55 uncalibrated cells; 3 wrong names accepted and 2 unresolved |
| Bæk/Muk Figure 4E | OCR 25/25; 645/650 finite estimates | Independent numeric accuracy cannot yet be certified: 13 HIGH_SE + 14 LOW_SE source samples versus 13 + 13 displayed; omitted sample, order and transformation unspecified |
| Li supplement 3A | Categorical labels tested in 4/4/3/3 group order | Clustered matrix and gene order absent; numeric truth unavailable |

The narrow Panel G layout is a reproducible OCR failure case. Its aspect ratio,
font rendering, whitespace, JPEG blur and engine resizing are confounded; this
experiment does not prove which factor causes the errors. Raw OCR strings and
accepted labels are reported separately so confidence rejections are not confused
with recognition errors. The existing confidence threshold is retained.

The two LC3B names share an sgRNA and source values; they remain separate rows.
The additional eLife 97830 Figure 2 workbook concerns liver expression and is
inventoried, but does not substitute for the Figure 4 blood workbook. Reuploaded
CRISPR and Li workbooks are tested byte-for-byte against the preserved originals.
