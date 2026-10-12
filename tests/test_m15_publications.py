"""Publication trials must preserve unavailable scale/annotation evidence."""

import hashlib
import json
import runpy
from pathlib import Path

import pytest
from PIL import Image

EVALUATE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "benchmarks/m15/publication_trials.py")
)["evaluate"]


def test_unverifiable_scale_is_review_required_without_numeric_extraction(tmp_path):
    image = tmp_path / "figure.png"
    Image.new("RGB", (100, 100), "white").save(image)
    trial = {
        "id": "unverifiable",
        "image_path": "figure.png",
        "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        "publication_url": "https://example.org/publication",
        "licence": "user-recorded lawful reuse",
        "second_pass_audited": True,
        "scale_model": "unverifiable",
    }
    annotation = tmp_path / "annotation.json"
    annotation.write_text(json.dumps({"trials": [trial]}))
    r = EVALUATE(annotation)
    assert r["records"][0]["status"] == "review_required"
    assert "supplied" not in r["records"][0]


def test_unreviewed_publication_annotations_are_rejected(tmp_path):
    p = tmp_path / "annotation.json"
    p.write_text(json.dumps({"trials": [{"id": "not_audited"}]}))
    with pytest.raises(ValueError, match="second-pass audit"):
        EVALUATE(p)
