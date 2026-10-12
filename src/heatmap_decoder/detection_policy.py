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
POLICY_VERSION = "review-acceptance.4"


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Preregistered M10 development checks; M12 release gates remain unestablished."""

    max_image_pixels: int = 16000000
    max_shortlisted_heatmaps: int = 3
    maximum_edge_shift_pitch: float = 0.1
    minimum_pitch: float = 6.0
    minimum_equal_grid: float = 0.9
    minimum_piecewise_constant: float = 0.95
    minimum_boundary_contrast: float = 12.0
    minimum_phase_alignment: float = 0.95
    perturbation_threshold_factors: tuple[float, ...] = (0.85, 1.0, 1.15)
    maximum_bar_noise: float = 4.0
    maximum_bar_step: float = 12.0
    bar_noise_multiplier: float = 3.0
    maximum_plateau_fraction: float = 0.12
    minimum_association_coverage: float = 0.9
    minimum_count_alignment: float = 0.98
    minimum_count_coverage: float = 0.95
    maximum_numeric_cells: int = 65536
    maximum_flagged_fraction: float = 0.01
    maximum_missing_or_ambiguous_fraction: float = 0.0
    maximum_association_pairs: int = 36
    maximum_total_ocr_pixels: int = 16000000


ACCEPTANCE_THRESHOLDS = AcceptanceThresholds()


def acceptance_threshold_snapshot() -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(ACCEPTANCE_THRESHOLDS).items()
    }


def policy_snapshot() -> dict[str, Any]:
    """Return a fresh JSON-safe snapshot; saved files never enable runtime automation."""
    return {
        "version": POLICY_VERSION,
        "mode": "review_only",
        "automatic_roi_enabled": False,
        "automatic_rows_enabled": False,
        "automatic_columns_enabled": False,
        "reason": "Stable capabilities are unestablished; scoped runtime opt-in is experimental",
        "experimental_export_scope": "numeric-only; supplied counts and supplied/reviewed scale",
        "coverage_target_is_per_image_permission": False,
        "count_model": "regular lattice; >=2 observation families, >=1 complete; extent/interiors",
        "core_search": {"long_axis_samples": 512, "cores_per_window": 16},
        "edge_profiles": {"outside_band_pixels": 12, "raster_uncertainty_pixels": 0.5},
        "required_complete_layouts": 30,
        "required_layout_families": 3,
        "maximum_heatmap_edge_error_cell_pitch": 0.1,
        "required_exact_count_examples_per_axis": 10,
        "required_independent_count_signals": 2,
        "negative_auto_acceptances_allowed": 0,
        "minimum_ordinary_acceptance_coverage": 0.7,
        "acceptance_thresholds": acceptance_threshold_snapshot(),
    }


def threshold_snapshot() -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in asdict(THRESHOLDS).items()
    }
