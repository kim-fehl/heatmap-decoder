# Bundled English recognition model

`en_PP-OCRv4_rec_infer.onnx` is PaddleOCR's English PP-OCRv4 recognition
model converted/distributed by RapidAI. Copyright PaddlePaddle/Baidu authors;
Apache License 2.0 (see `LICENSE`). The ONNX file includes its English character
dictionary. No separate Chinese dictionary is applied.

SHA-256: `e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318`.

Upstream model/hash registry:
https://github.com/RapidAI/RapidOCR/blob/7cfd3d9ef786a2e538ca30df700325e72e85b5f5/python/rapidocr/default_models.yaml
(`en_PP-OCRv4_rec_mobile`, same model bytes under the newer filename).

Downloaded from the pinned GitHub mirror below and verified against the upstream
SHA-256 above:
https://github.com/zclucas/RMT/blob/e18f7f8512e16b173668d5f9d17e0135f840c0e7/Plugins/RapidOcr/en_models/en_PP-OCRv4_rec_infer.onnx

The worker explicitly selects this model for the default `language="eng"`.
It retains RapidOCR's bundled Chinese detector/classifier. `language="chinese"`
selects the legacy Chinese recognizer; model hashes in provenance identify the
loaded files. Model weights are packaged in the wheel/sdist; OCR needs no network.
The English recognition alphabet is ASCII; Greek identifiers require another
recognizer/language pack or a manual/file label source.
