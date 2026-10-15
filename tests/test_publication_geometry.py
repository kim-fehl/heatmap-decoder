"""Controlled cases exercise values, irregular spans and ambiguous input."""

import os
import shutil

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("pytesseract")
if os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1" or not shutil.which("tesseract"):
    pytest.skip(
        "Set HEATMAP_DECODER_OCR_TESTS=1 with publication OCR dependencies", allow_module_level=True
    )
pytestmark = pytest.mark.ocr_integration
publication = pytest.importorskip("heatmap_decoder.publication")
Box = publication.Box
DecodeError = publication.DecodeError
_rectangle_words = publication._rectangle_words
_values = publication._values
calibrate = publication.calibrate
extend_group_spans = publication.extend_group_spans
infer_grid = publication.infer_grid


def test_unequal_short_group_lines_snap_without_assuming_equal_sizes():
    edges = np.arange(13) * 20 + 17
    # True groups contain 2, 7, 3 columns; printed rules are inset.
    rules = [Box(23, 0, 28, 3), Box(63, 0, 128, 3), Box(203, 0, 48, 3)]
    assert extend_group_spans(rules, edges) == [(0, 2), (2, 9), (9, 12)]


def test_ambiguous_group_gap_is_rejected():
    with pytest.raises(DecodeError, match="Ambiguous"):
        extend_group_spans([Box(1, 0, 15, 3), Box(90, 0, 10, 3)], np.arange(11) * 10)


def test_nonlinear_numeric_labels_are_not_fitted_as_linear():
    with pytest.raises(DecodeError, match="linear"):
        calibrate(
            [{"pixel": 0, "value": 0}, {"pixel": 50, "value": 1}, {"pixel": 100, "value": 10}]
        )


def test_color_to_value_roundtrip_against_known_values():
    # Independent synthetic linear palette with labeled values inside its extent.
    rgb = np.full((150, 220, 3), 255, dtype=np.uint8)
    palette = np.array([[2 * t, 40, 220 - 2 * t] for t in range(101)], dtype=np.uint8)
    rgb[20:121, 195:205] = palette[:, None, :]
    truth = np.array([[-1.5, -0.5, 0.5], [1.25, 0, -1]])
    for row in range(2):
        for col in range(3):
            index = int(round((truth[row, col] + 2) * 25))
            rgb[10 + row * 30 : 40 + row * 30, 10 + col * 40 : 50 + col * 40] = palette[index]
    f = calibrate([{"pixel": 45, "value": -1}, {"pixel": 95, "value": 1}])
    recovered, quality = _values(
        rgb, np.array([10, 40, 70]), np.array([10, 50, 90, 130]), Box(195, 20, 10, 101), f
    )
    assert np.max(abs(np.array(recovered) - truth)) < 0.021
    assert quality["max_rgb_distance"] == 0


def test_grid_fit_avoids_half_and_double_count_harmonics():
    profile = np.zeros(249)
    profile[np.arange(25, 250, 25) - 1] = 20
    n, support = infer_grid(profile, 250)
    assert n == 10
    assert support > 0.99


@pytest.mark.parametrize(
    "background,foreground,polarity",
    [
        ((0, 0, 0), (255, 255, 255), "light"),
        ((220, 0, 0), (255, 255, 255), "light"),
        ((179, 179, 179), (255, 255, 255), "light"),
        ((0, 0, 255), (255, 255, 255), "light"),
        ((255, 255, 255), (0, 0, 0), "dark"),
        ((0, 180, 0), (0, 0, 0), "dark"),
    ],
)
def test_rectangle_ocr_recovers_same_text_for_both_polarities(background, foreground, polarity):
    image = np.full((70, 260, 3), background, dtype=np.uint8)
    cv2.putText(
        image, "Group 7", (35, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, foreground, 2, cv2.LINE_AA
    )
    words, color, actual_polarity = _rectangle_words(image, Box(0, 0, 260, 70))
    assert " ".join(w["text"] for w in words) == "Group 7"
    assert actual_polarity == polarity
    assert color == list(background)
