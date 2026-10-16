"""Legend contracts and fixtures."""

import numpy as np
import pytest

from heatmap_decoder import ExtractionConfig
from heatmap_decoder.calibration import fit_linear_scale, normalise_bar_positions, scale_values
from heatmap_decoder.legend import (
    evaluate_legend,
    parse_numeric_tick,
)
from heatmap_decoder.models import (
    LegendConfig,
    ScaleAnchor,
    ScaleConfig,
)
from tests.support.legend import ROOT, anchors_for, ticks_from_anchors


@pytest.mark.parametrize(
    "raw,value",
    [
        ("-2", -2),
        ("−1.25", -1.25),
        ("+.75", 0.75),
        (" 2. ", 2),
        ("- 2", -2),
        ("1.5e-3", 0.0015),
        ("-2E+2", -200),
        ("10^3", 1000),
        ("10^-3", 0.001),
    ],
)
def test_numeric_grammar_preserves_raw_and_records_normalisation(raw, value):
    parsed = parse_numeric_tick(raw)
    assert parsed.raw_text == raw
    assert parsed.value == value
    assert parsed.reason is None
    if raw != parsed.normalised_text or "^" in raw:
        assert parsed.actions


@pytest.mark.parametrize(
    "raw",
    ["O.1", "1,234", "2 mg", "10³", "1e", "--2", "NaN", "inf", "1e999", "x10^3", "1 000", "HSP70A"],
)
def test_unparsed_and_overflow_never_silently_corrected(raw):
    assert parse_numeric_tick(raw).value is None


def test_decimal_comma_requires_explicit_locale():
    assert parse_numeric_tick("-1,5").value is None
    assert parse_numeric_tick("-1,5", ",").value == -1.5
    assert parse_numeric_tick("1,2,3", ",").value is None
    assert parse_numeric_tick("1,234.5", ",").value is None


@pytest.mark.parametrize(
    "orientation,box", [("vertical", (20, 40, 34, 440)), ("horizontal", (20, 40, 420, 54))]
)
@pytest.mark.parametrize("count", [2, 3, 5, 7])
@pytest.mark.parametrize("descending", [False, True])
def test_fit_uses_actual_nonuniform_pixel_positions(orientation, box, count, descending):
    anchors = anchors_for(box, orientation, count, descending, nonuniform=True)
    fit = fit_linear_scale(anchors, box, orientation)
    assert fit.slope == pytest.approx(-4 if descending else 4)
    assert fit.intercept == pytest.approx(2 if descending else -2)
    assert fit.endpoint_coverage
    assert fit.tests_linearity == (count >= 3)
    assert max(abs(v) for v in fit.value_residuals) < 1e-12
    coordinates = np.array(
        [box[1], box[3] - 1] if orientation == "vertical" else [box[0], box[2] - 1], float
    )
    np.testing.assert_array_equal(normalise_bar_positions(coordinates, box, orientation), [0, 1])


@pytest.mark.parametrize(
    "case", ["duplicate", "zero_range", "reversal", "nonlinear", "close", "outside"]
)
def test_invalid_scale_cannot_fall_back_to_endpoints(case):
    box = (20, 40, 34, 440)
    a = list(anchors_for(box, "vertical", 5))
    if case == "duplicate":
        a[2] = a[1].model_copy(update={"value": 100})
    elif case == "zero_range":
        a = [p.model_copy(update={"value": 1}) for p in a]
    elif case == "reversal":
        a[2] = a[2].model_copy(update={"value": 3})
    elif case == "nonlinear":
        a[2] = a[2].model_copy(update={"value": 0.4})
    elif case == "close":
        a = [ScaleAnchor(position=40 + i * 0.0001, value=i) for i in range(5)]
    else:
        a[0] = a[0].model_copy(update={"position": 10})
    with pytest.raises(ValueError):
        fit_linear_scale(tuple(a), box, "vertical")


def test_legacy_endpoint_scale_is_bit_for_bit_identical():
    config = ExtractionConfig.model_validate_json((ROOT / "examples/config.json").read_text())
    expected = np.linspace(config.colourbar_start_value, config.colourbar_end_value, 320)
    np.testing.assert_array_equal(scale_values(config, 320), expected)
    explicit = config.model_copy(update={"scale": ScaleConfig(source="manual_endpoints")})
    np.testing.assert_array_equal(scale_values(explicit, 320), expected)
    payload = config.model_dump()
    payload["scale"] = {
        "source": "manual_ticks",
        "anchors": [{"position": 0, "value": 1}, {"position": 1, "value": 2}],
    }
    with pytest.raises(ValueError, match="conflicts"):
        ExtractionConfig.model_validate(payload)


@pytest.mark.parametrize("count", [2, 3, 5, 7])
def test_tick_number_endpoint_coverage_and_release_gates(count):
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    result = evaluate_legend(
        config, ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", count))
    )
    assert result.development_eligible == (count >= 3)
    assert result.status == "review_required"
    assert not result.automatic_export_allowed
    if count == 2:
        assert "two_ticks_cannot_test_linearity" in result.reasons
    interior = evaluate_legend(
        config,
        ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", count, interior=True)),
    )
    assert interior.development_eligible == (count >= 3)
    assert "interior_ticks_limit_calibration_span" in interior.reasons


def test_unparsed_tick_and_text_centre_fallback_block_eligibility():
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    ticks = list(ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 5)))
    ticks[2] = ticks[2].model_copy(
        update={"raw_text": "O", "parsed_value": None, "status": "unresolved"}
    )
    result = evaluate_legend(config, tuple(ticks))
    assert result.status == "unresolved"
    assert not result.development_eligible
    fallback = evaluate_legend(
        config, ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical"), "text_centre")
    )
    assert not fallback.development_eligible
    assert "text_centres_are_weaker_position_evidence" in fallback.reasons


def test_nonlinear_scale_keeps_residuals_and_never_removes_outlier():
    config = LegendConfig(colourbar_bbox=(20, 40, 34, 440), colourbar_orientation="vertical")
    ticks = list(ticks_from_anchors(anchors_for(config.colourbar_bbox, "vertical", 5)))
    ticks[2] = ticks[2].model_copy(update={"parsed_value": 0.4})
    result = evaluate_legend(config, tuple(ticks))
    assert result.fit is not None
    assert len(result.fit.value_residuals) == 5
    assert not result.fit.within_residual_limits
    assert "anchors_contradict_linear_scale" in result.reasons
    assert result.status == "unresolved"


@pytest.mark.parametrize("text", ["1e-999", "10^-999", "1" * 257])
def test_underflow_and_unbounded_numeric_strings_are_unresolved(text):
    assert parse_numeric_tick(text).value is None
    assert parse_numeric_tick("0e-999").value == 0


@pytest.mark.parametrize(
    "raw,value",
    [
        ("—1", -1),
        ("–1.5", -1.5),
        ("‒2", -2),
        ("‐.5", -0.5),
        ("‑ 2", -2),
        ("1e—2", 0.01),
        ("10^–3", 0.001),
    ],
)
def test_observed_dash_is_a_minus_only_in_numeric_sign_positions(raw, value):
    parsed = parse_numeric_tick(raw)
    assert parsed.raw_text == raw
    assert parsed.value == value
    assert "-" in parsed.normalised_text
    assert any(action.startswith("observed_dash_to_minus:U+") for action in parsed.actions)


@pytest.mark.parametrize("raw", ["—", "1—2", "——1", "gene—1", "O.1", "1_2", "—NaN"])
def test_numeric_sign_constraints_do_not_guess_digits_ranges_or_identifiers(raw):
    assert parse_numeric_tick(raw).value is None
