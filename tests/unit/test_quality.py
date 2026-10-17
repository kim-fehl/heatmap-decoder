"""Quality contracts and fixtures."""

import numpy as np
import pytest

from heatmap_decoder import ExtractionConfig
from heatmap_decoder.calibration import match_with_quality
from tests.support.quality import reference


@pytest.mark.parametrize(
    "changes",
    [
        {"colourbar_start_value": -1e308, "colourbar_end_value": 1e308},
        {"max_colour_distance": -1},
        {"max_colour_distance": float("nan")},
        {"ambiguity_colour_tolerance": -1},
        {"ambiguity_position_fraction": 0},
        {"ambiguity_position_fraction": 2},
        {"minimum_valid_fraction": 0},
        {"minimum_valid_fraction": 2},
        {"missing_colour_tolerance": -1},
        {"missing_colours": [[256, 0, 0]]},
        {"missing_colours": [[0.5, 0, 0]]},
    ],
)
def test_invalid_quality_configuration(changes):
    _, cfg = reference()
    with pytest.raises(ValueError):
        ExtractionConfig(**(cfg.model_dump() | changes))


def test_black_white_distance_is_delta_e76():
    match = match_with_quality(
        np.array([[0.0, 0.0, 0.0]]), np.array([[255.0, 255.0, 255.0], [255.0, 255.0, 255.0]])
    )
    assert match.distance[0] == pytest.approx(100, abs=0.01)
    assert match.ambiguous[0]
