"""Compatibility contracts and fixtures."""

import importlib
import subprocess
import sys
import warnings

import heatmap_decoder


def test_legacy_imports_delegate_to_identical_implementation():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        legacy = importlib.import_module("heatmap2matrix")
    assert any("deprecated" in str(w.message) for w in caught)
    assert legacy.extract is heatmap_decoder.extract
    assert legacy.ExtractionConfig is heatmap_decoder.ExtractionConfig
    for name in ("models", "geometry", "sampling", "calibration", "extract", "io"):
        assert importlib.import_module(f"heatmap2matrix.{name}") is importlib.import_module(
            f"heatmap_decoder.{name}"
        )


def test_no_ocr_dependency_is_imported_by_core_or_geometry():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import heatmap_decoder.labels; assert 'pytesseract' not in sys.modules",
        ],
        check=True,
    )
