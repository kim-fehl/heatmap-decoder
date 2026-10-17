# OCR language and identifier characters

Gene/sample and numeric tick OCR default to RapidOCR with **the English PP-OCRv4 recognition
model**, native-size grayscale crops and `language="eng"`. Numeric fields add two
pixels of white padding. Tesseract is selectable as a legacy/language fallback,
with its optional wrapper and separately installed engine. `en` and
`english` are aliases for the same model. The model and its embedded character
dictionary are packaged locally; no runtime downloads or remote OCR calls occur.
The detector remains the bundled RapidOCR detector. Provenance records the actual
recognition model name and SHA-256, rather than inferring it from the language string.

Earlier versions wrote `eng` in configuration but actually loaded the package's
Chinese recognizer. That model often omitted spaces in English. The
[Docling diagnosis](https://github.com/docling-project/docling/issues/2887#issuecomment-4682809758)
identifies the same model-selection problem. Selecting English now sets the model
path explicitly. We preserve whatever spacing the model actually reads; no regex
inserts spaces or repairs identifier spelling.

The paired [development comparison](../benchmarks/ocr-review/language-comparison.json)
uses identical crops/settings except for the recognizer. On the diagnostic example,
exact rotated sample names improve from 0/4 to 4/4. On the inspected publication,
exact raw genes stay 26/33, samples stay 8/8, and wrong heuristically accepted genes
drop from 6 to 3. These inspected figures do not qualify unattended identifier export.
One demo gene still loses its initial `T`; inspect the raw output in the README figure.
On the M15 quarter-turn fixture, English reads all eight upright sample names exactly
but also retains a competing opposite-angle geometry. The orientation decision
therefore requires review; recognizer changes can alter automatic geometry proposals.

The English recognition alphabet is ASCII. Allowing Greek in the character policy
does not add Greek to the model vocabulary. For Greek identifiers, use an appropriate
installed Tesseract language pack (for example `eng+ell`), or a reviewed/file label
source. For comparison with earlier recognition, set
`labels.ocr.language="chinese"` (`ch`/`chi_sim` are aliases); select
`labels.character_policy="unrestricted"` when non-Latin/Greek names are intentional.

## Unexpected characters

`\u6771` is the Unicode escape for **東** (U+6771, “east”). It was a Chinese-model
hallucination in front of `BRCA1`, not part of the gene name.

The default `labels.character_policy="latin_greek"` keeps Latin and Greek letters
(including accents and their combining marks), decimal/subscript/superscript
numbers, punctuation, ASCII symbols such as `+`, and spaces. Other scripts and
emoji are removed **from a suggested reading only**:

| Field | Example |
|---|---|
| `raw_text` | `東BRCA1` |
| `suggested_text` | `BRCA1` |
| `final_text` | empty |
| `issues` | `unexpected_label_characters` |
| `status` | `unreviewed` instead of heuristic `accepted` |

`raw_text`, tokens, crops and confidence remain intact in the evidence. Existing
ambiguity/low-confidence states remain unresolved. `suggested_text` appears in
`labels.tsv` and provenance; it is never automatically exported or prefilled as a
verified review. The reviewer supplies the intended name in `labels_review.tsv`.
Manual/file labels retain the text the user supplied.

Stripping cannot prove identity: deleting an extra character might yield another
valid gene name. English model selection fixes the underlying vocabulary choice;
the character guard makes unexpected-script proposals visible and reviewable.

Numeric RapidOCR retains unconstrained model output and applies the strict number
grammar afterwards; no letters are substituted into digits. The new pipeline reads
all 36 physical ticks in 12 paired clean/JPEG/blur development cases correctly
without Tesseract-derived crops. It also parses all five inspected publication ticks,
though three remain low-confidence review proposals. Existing experimental OCR-scale
export qualification is still specific to the earlier Tesseract settings; reviewing
RapidOCR ticks into manual scale anchors avoids that unqualified automatic path.
