"""Robust original-pixel edge localisation beyond blur/JPEG halos.

Uncertainty is a measured engineering localisation bound, not a probability.
"""

from dataclasses import dataclass

import numpy as np

from .models import BoundingBox, RGBArray


@dataclass(frozen=True)
class EdgeProfile:
    contrast_rgb: float
    outside_spread_rgb: float
    location_offset_pixels: float
    uncertainty_pixels: float
    support_fraction: float
    outside_noise_rgb: float
    distances: tuple[float, ...]
    normalised_profile: tuple[float, ...]
    closed: bool


def measure_edges(
    image: RGBArray, box: BoundingBox, tolerance: float = 1.0, *, refinement: bool = False
) -> tuple[EdgeProfile, ...]:
    x, y, r, s = box
    edges = []
    for coordinate, boundary, inside_sign, first, last in (
        (0, x, 1, y, s),
        (0, r, -1, y, s),
        (1, y, 1, x, r),
        (1, s, -1, x, r),
    ):
        band = 12 if refinement else min(6, max(3, (r - x if coordinate == 0 else s - y) // 3))
        outside_band = 12
        distances = np.arange(-band, outside_band, dtype=float) + 0.5
        positions = (boundary - inside_sign * distances - 0.5).astype(int)
        extent = image.shape[1] if coordinate == 0 else image.shape[0]
        usable = (positions >= 0) & (positions < extent)
        if not np.all(usable):
            edges.append(EdgeProfile(0.0, 0.0, 0.0, float(band), 0.0, 0.0, (), (), False))
            continue
        trim = min(3, (last - first) // 8)
        along = np.unique(
            np.linspace(first + trim, last - trim - 1, min(256, last - first - 2 * trim)).astype(
                int
            )
        )
        pixels = (
            image[along[None, :], positions[:, None], :3]
            if coordinate == 0
            else image[positions[:, None], along[None, :], :3]
        ).astype(float)
        inside = np.median(pixels[: max(2, band - 2)], axis=0)
        outside = np.median(pixels[-2:], axis=0)
        delta = inside - outside
        contrasts = np.mean(np.abs(delta), axis=1)
        denominator = np.sum(delta**2, axis=1)
        valid = (contrasts >= 12) & (denominator > 100)
        if not np.any(valid):
            edges.append(
                EdgeProfile(
                    float(np.median(contrasts)),
                    0.0,
                    0.0,
                    float(band),
                    0.0,
                    0.0,
                    tuple(distances),
                    (),
                    False,
                )
            )
            continue
        alpha = (
            np.sum((pixels[:, valid] - outside[valid]) * delta[valid], axis=2) / denominator[valid]
        )
        profile = np.median(alpha, axis=1)
        locations = []
        slopes = []
        for column in range(alpha.shape[1]):
            crossings = np.flatnonzero((alpha[:-1, column] >= 0.5) & (alpha[1:, column] < 0.5))
            if len(crossings) != 1:
                continue
            k = int(crossings[0])
            slope = alpha[k, column] - alpha[k + 1, column]
            locations.append(float(distances[k] + (alpha[k, column] - 0.5) / slope))
            slopes.append(float(slope))
        far = pixels[-2:]
        spread = float(
            np.max(np.quantile(far, 0.9, axis=(0, 1)) - np.quantile(far, 0.1, axis=(0, 1)))
        )
        noise = float(np.median(np.abs(far - np.median(far, axis=(0, 1)))))
        support = len(locations) / len(along)
        if locations:
            location = float(np.median(locations))
            dispersion = float(np.quantile(np.abs(np.array(locations) - location), 0.9))
            # Raster half-pixel resolution plus observed position/noise dispersion.
            uncertainty = (
                0.5
                + dispersion
                + noise / max(float(np.median(contrasts[valid])) * float(np.median(slopes)), 1.0)
            )
        else:
            location, uncertainty = 0.0, float(band)
        closed = support >= 0.8 and spread <= 8 and abs(location) + uncertainty <= tolerance
        edges.append(
            EdgeProfile(
                float(np.median(contrasts)),
                spread,
                location,
                uncertainty,
                support,
                noise,
                tuple(map(float, distances)),
                tuple(map(float, profile)),
                closed,
            )
        )
    return tuple(edges)


def refine_measured_box(image: RGBArray, box: BoundingBox) -> BoundingBox:
    """Correct bounded component halo shifts using independently fitted transitions."""
    edges = measure_edges(image, box, 12.0, refinement=True)
    offsets = []
    for edge in edges:
        if (
            edge.support_fraction < 0.8
            or edge.outside_spread_rgb > 8
            or edge.uncertainty_pixels > 1.5
            or abs(edge.location_offset_pixels) > 8
        ):
            offsets.append(0)
        else:
            offsets.append(round(edge.location_offset_pixels))
    x, y, r, s = box
    # A uniform black outer stroke already trimmed by the component refiner
    # is distinct from a blur halo. Preserve its inner cell boundary.
    adjacent = (
        image[y:s, max(0, x - 1), :3],
        image[y:s, min(image.shape[1] - 1, r), :3],
        image[max(0, y - 1), x:r, :3],
        image[min(image.shape[0] - 1, s), x:r, :3],
    )
    for i, pixels in enumerate(adjacent):
        if offsets[i] > 0 and np.max(pixels) <= 20:
            offsets[i] = 0
    corrected = (x - offsets[0], y - offsets[2], r + offsets[1], s + offsets[3])
    if (
        0 <= corrected[0] < corrected[2] <= image.shape[1]
        and 0 <= corrected[1] < corrected[3] <= image.shape[0]
    ):
        return corrected
    return box
