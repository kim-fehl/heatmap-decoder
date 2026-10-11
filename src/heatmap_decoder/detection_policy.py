"""Versioned detector thresholds and the fail-closed v0.3 hand-off policy.

M9 freezes the existing M8 heuristics rather than tuning on the held-out set.
These engineering constants are not probabilities or scientific uncertainty.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectorThresholds:
    border_trim_pixels: int = 4
    uniform_rgb_spread: float = 2.0
    minimum_cell_pitch: float = 3.0
    grid_slots_per_axis: int = 32
    interior_positions: tuple[float, ...] = (0.3, 0.5, 0.7)
    variation_normaliser: float = 25.0
    grid_residual_multiplier: float = 4.0
    minimum_analysis_extent: int = 6
    heatmap_score_weights: tuple[float, ...] = (0.15, 0.45, 0.25, 0.15)
    minimum_variation_score: float = 0.2
    minimum_equal_grid_score: float = 0.8
    bar_join_cross_tolerance: int = 2
    bar_join_gap_fraction: float = 0.15
    bar_refinement_reference_size: int = 512
    bar_minimum_thickness: int = 2
    bar_minimum_length: int = 24
    bar_minimum_aspect: float = 4.0
    bar_minimum_variation: float = 12.0
    bar_maximum_transverse_error: float = 8.0
    bar_maximum_step: float = 12.0
    bar_minimum_continuity: float = 0.97
    bar_minimum_changing_fraction: float = 0.2
    association_heatmap_samples: int = 20
    association_bar_samples: int = 256
    association_maximum_delta_e76: float = 10.0
    association_minimum_palette_coverage: float = 0.5
    association_minimum_axis_overlap: float = 0.5
    count_cross_samples: int = 32
    count_transition_threshold: float = 8.0
    count_maximum_transition_run: int = 4
    count_maximum_stroke_width: int = 3
    count_minimum_stroke_contrast: float = 8.0
    count_minimum_observations: int = 2
    count_position_tolerance_fraction: float = 0.08
    count_minimum_alignment: float = 0.75


THRESHOLDS = DetectorThresholds()
POLICY_VERSION = "m9-review-only.1"


def policy_snapshot() -> dict[str, Any]:
    """Return a fresh JSON-safe snapshot; saved files never enable runtime automation."""
    return {
        "version": POLICY_VERSION,
        "mode": "review_only",
        "automatic_roi_enabled": False,
        "automatic_rows_enabled": False,
        "automatic_columns_enabled": False,
        "reason": "M9 complete-layout and independent per-axis count gates are not established",
        "required_complete_layouts": 10,
        "required_layout_families": 3,
        "maximum_heatmap_edge_error_cell_pitch": 0.1,
        "required_exact_count_examples_per_axis": 10,
        "required_independent_count_signals": 2,
        "negative_auto_acceptances_allowed": 0,
    }


def threshold_snapshot() -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(THRESHOLDS).items()
    }
