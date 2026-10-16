"""Publication labels contracts and fixtures."""

from heatmap_decoder.io import load_raster
from heatmap_decoder.models import ExtractionConfig
from tests.support.publication_sources import ROOT


def test_publication_ocr_against_reviewed_labels():

    from heatmap_decoder.extract import extract_labels

    for name, minimum_accepted, minimum_raw in [
        ("elife-17290-fig2", 9, 18),
        ("elife-78496-fig5e", 44, 45),
        ("elife-97830-fig4e", 25, 25),
    ]:
        folder = ROOT / name
        genes = (folder / "rows.txt").read_text().splitlines()
        config = ExtractionConfig.model_validate_json((folder / "labels-ocr.json").read_bytes())
        slots = extract_labels(load_raster(folder / "figure.jpg"), config).axes[0].slots
        assert len(slots) == len(genes)
        assert sum(g == s.final_text for g, s in zip(genes, slots, strict=True)) >= minimum_accepted
        assert sum(g == s.raw_text for g, s in zip(genes, slots, strict=True)) >= minimum_raw
        if name == "elife-17290-fig2":
            assert all(
                s.final_text is None or s.final_text == g for g, s in zip(genes, slots, strict=True)
            )
