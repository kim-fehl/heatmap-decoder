"""Label characters contracts and fixtures."""

import pytest

from heatmap_decoder.labels import filter_identifier_characters


@pytest.mark.parametrize(
    "text",
    [
        "IL-1β / TNFα_2 (control)+",
        "EsIκBζ",
        "Échantillon-2",
        "α\u0301",
        "H₂O",
    ],
)
def test_scientific_names_keep_greek_accents_spaces_and_punctuation(text):
    assert filter_identifier_characters(text) == text


@pytest.mark.parametrize(
    "text, cleaned",
    [
        ("東BRCA1", "BRCA1"),
        ("株Sample 1😀", "Sample 1"),
        ("株\u0301", ""),
    ],
)
def test_other_scripts_are_only_stripped_from_the_suggestion(text, cleaned):
    assert filter_identifier_characters(text) == cleaned
