"""Legend contracts and fixtures."""

import numpy as np
from PIL import Image

from heatmap_decoder.io import (
    extract_file,
    inspect_legend_file,
)
from heatmap_decoder.models import (
    LegendConfig,
    ScaleConfig,
)
from tests.support.legend import fixture_config
from tests.support.synthetic import legend_fixture


def test_real_engine_clean_five_ticks_and_reviewed_reconstruction(tmp_path):
    image, truth = legend_fixture()
    figure = tmp_path / "figure.png"
    Image.fromarray(image).save(figure)
    config = LegendConfig(colourbar_bbox=truth["colourbar_bbox"], colourbar_orientation="vertical")
    result = inspect_legend_file(figure, config, tmp_path / "review")
    assert len(result.ticks) == 5
    assert [t.raw_text for t in result.ticks] == [t["text"] for t in truth["tick_tokens"]]
    assert [t.position for t in result.ticks] == truth["positions"]
    assert result.development_eligible
    extracted = extract_file(
        figure,
        fixture_config(truth, ScaleConfig(source="ocr_ticks", quantity="row z-score")),
        tmp_path / "output",
        legend_review=tmp_path / "review/legend_review.tsv",
    )
    assert np.abs(extracted.values - np.array(truth["expected_matrix"])).mean() / 4 < 0.02
    assert np.isfinite(extracted.values).all()
