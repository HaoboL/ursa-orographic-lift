#!/usr/bin/env python3
"""Deterministic URSA V10 terrain/wind variable extraction.

This module implements input variables only.  It does not read reference flow,
fit coefficients, evaluate WEMOD with an assumed wake coefficient, or select a
production architecture.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


GRID_SPACING_M = 30.0


def wrapped_angle_difference_deg(first_deg: float, second_deg: float) -> float:
    """Return first-second in [-180, 180)."""

    return (first_deg - second_deg + 180.0) % 360.0 - 180.0


def vector_direction_deg(east: float, north: float) -> float:
    return float(math.degrees(math.atan2(north, east)) % 360.0)


def sample_field(field: np.ndarray, row: float, column: float) -> float:
    return float(
        ndimage.map_coordinates(
            np.asarray(field, dtype=np.float64),
            [[row], [column]],
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )[0]
    )


def sample_wind(
    wind_east_100m: np.ndarray,
    wind_north_100m: np.ndarray,
    row: float,
    column: float,
) -> dict[str, float]:
    east = sample_field(wind_east_100m, row, column)
    north = sample_field(wind_north_100m, row, column)
    return {
        "east_100m": east,
        "north_100m": north,
        "speed_100m": math.hypot(east, north),
        "direction_deg": vector_direction_deg(east, north),
    }


def terrain_domain_diagnostics(
    dem: np.ndarray,
    *,
    spacing_m: float = GRID_SPACING_M,
) -> dict[str, float]:
    """Return coefficient-free whole-domain terrain-scale diagnostics.

    The ``mu`` and inferred ``xi`` identities follow Helbig et al. (2017)'s
    isotropic Gaussian-random-field definition.  On a natural DEM they are
    transfer diagnostics only; this function does not evaluate that paper's
    fitted wind-speed parameterization.
    """

    elevation = np.asarray(dem, dtype=np.float64)
    if elevation.ndim != 2 or not np.all(np.isfinite(elevation)):
        raise ValueError("terrain diagnostics require a finite two-dimensional DEM")
    rows, columns = np.indices(elevation.shape, dtype=np.float64)
    design = np.column_stack(
        [np.ones(elevation.size), columns.ravel(), rows.ravel()]
    )
    coefficients, *_ = np.linalg.lstsq(design, elevation.ravel(), rcond=None)
    detrended = elevation - (
        coefficients[0] + coefficients[1] * columns + coefficients[2] * rows
    )
    gradient_north, gradient_east = np.gradient(elevation, spacing_m, spacing_m)
    mean_square_slope = float(np.mean(gradient_east**2 + gradient_north**2))
    mu = math.sqrt(0.5 * mean_square_slope)
    sigma = float(np.std(detrended))
    correlation_length = math.sqrt(2.0) * sigma / mu if mu > 0.0 else math.inf
    domain_side = spacing_m * math.sqrt(float(elevation.size))
    return {
        "terrain_domain_side_equivalent_m": domain_side,
        "terrain_detrended_elevation_sigma_m": sigma,
        "terrain_mean_square_slope_native": mean_square_slope,
        "terrain_helbig_mu_native": mu,
        "terrain_helbig_inferred_correlation_length_m": correlation_length,
        "terrain_grid_over_inferred_correlation_length": spacing_m / correlation_length,
        "terrain_domain_over_inferred_correlation_length": domain_side / correlation_length,
    }


def point_from_t(
    anchor_row: float,
    anchor_column: float,
    t_m: float,
    unit_east: float,
    unit_north: float,
    *,
    spacing_m: float = GRID_SPACING_M,
) -> tuple[float, float]:
    return (
        anchor_row + t_m * unit_north / spacing_m,
        anchor_column + t_m * unit_east / spacing_m,
    )


def full_map_transect(
    field: np.ndarray,
    anchor_row: int,
    anchor_column: int,
    unit_east: float,
    unit_north: float,
    *,
    spacing_m: float = GRID_SPACING_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the complete map-line intersection at native grid spacing."""

    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("transect field must be two-dimensional")
    norm = math.hypot(unit_east, unit_north)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("transect direction must be finite and nonzero")
    east = unit_east / norm
    north = unit_north / norm
    x0 = float(anchor_column) * spacing_m
    y0 = float(anchor_row) * spacing_m
    bounds: list[tuple[float, float]] = []
    for coordinate, direction, maximum in (
        (x0, east, (array.shape[1] - 1) * spacing_m),
        (y0, north, (array.shape[0] - 1) * spacing_m),
    ):
        if abs(direction) < 1.0e-12:
            if not 0.0 <= coordinate <= maximum:
                raise ValueError("anchor lies outside map")
            bounds.append((-math.inf, math.inf))
        else:
            first = (0.0 - coordinate) / direction
            second = (maximum - coordinate) / direction
            bounds.append((min(first, second), max(first, second)))
    lower = max(value[0] for value in bounds)
    upper = min(value[1] for value in bounds)
    start_index = int(math.ceil(lower / spacing_m - 1.0e-12))
    stop_index = int(math.floor(upper / spacing_m + 1.0e-12))
    if stop_index - start_index < 4:
        raise ValueError("map-line intersection is too short")
    t_m = spacing_m * np.arange(start_index, stop_index + 1, dtype=np.float64)
    rows = anchor_row + t_m * north / spacing_m
    columns = anchor_column + t_m * east / spacing_m
    values = ndimage.map_coordinates(
        array,
        [rows, columns],
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("full-map transect unexpectedly contains nonfinite values")
    return t_m, np.asarray(values, dtype=np.float64)


def nearest_valley(
    valleys: np.ndarray,
    peak_index: int,
    *,
    upstream: bool,
) -> int | None:
    candidates = valleys[valleys < peak_index] if upstream else valleys[valleys > peak_index]
    if not len(candidates):
        return None
    return int(candidates[-1] if upstream else candidates[0])


def half_height_length(
    elevation: np.ndarray,
    t_m: np.ndarray,
    peak_index: int,
    valley_index: int,
) -> float:
    crest = float(elevation[peak_index])
    base = float(elevation[valley_index])
    target = base + 0.5 * (crest - base)
    step = 1 if valley_index > peak_index else -1
    current = peak_index
    distance_from_peak = 0.0
    while current != valley_index:
        following = current + step
        first = float(elevation[current])
        second = float(elevation[following])
        if min(first, second) <= target <= max(first, second):
            fraction = 0.0 if second == first else (target - first) / (second - first)
            segment_length = abs(float(t_m[following] - t_m[current]))
            return distance_from_peak + abs(fraction) * segment_length
        distance_from_peak += abs(float(t_m[following] - t_m[current]))
        current = following
    raise ValueError("half-height crossing was not found inside a closed side")


def _strict_extrema(elevation: np.ndarray, *, peaks: bool) -> np.ndarray:
    """Return plateau-safe extrema above the floating-point resolution floor."""

    values = np.asarray(elevation, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(values))))
    numerical_floor = np.nextafter(np.spacing(scale), math.inf)
    signed = values if peaks else -values
    return find_peaks(signed, prominence=numerical_floor)[0]


def side_metrics(
    elevation: np.ndarray,
    t_m: np.ndarray,
    peak_index: int,
    valley_index: int,
) -> dict[str, float | int]:
    crest = float(elevation[peak_index])
    base = float(elevation[valley_index])
    height = crest - base
    valley_length = abs(float(t_m[peak_index] - t_m[valley_index]))
    if height <= 0.0 or valley_length <= 0.0:
        raise ValueError("ridge side must have positive height and length")
    half_length = half_height_length(elevation, t_m, peak_index, valley_index)
    first = min(peak_index, valley_index)
    last = max(peak_index, valley_index)
    native_slopes = np.abs(
        np.diff(elevation[first : last + 1]) / np.diff(t_m[first : last + 1])
    )
    maximum = float(np.max(native_slopes))
    mean = height / valley_length
    return {
        "base_elevation_m": base,
        "height_m": height,
        "valley_length_m": valley_length,
        "half_height_length_m": half_length,
        "mean_slope_tangent": mean,
        "mean_slope_deg": math.degrees(math.atan(mean)),
        "half_height_slope_ratio": height / (2.0 * half_length),
        "max_native_slope_tangent": maximum,
        "max_native_slope_deg": math.degrees(math.atan(maximum)),
        "native_interval_count": int(last - first),
    }


def prefixed_side(prefix: str, metrics: dict[str, float | int]) -> dict[str, float | int]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _target_peak_and_valleys(
    t_m: np.ndarray,
    elevation: np.ndarray,
) -> tuple[int, int, int] | None:
    peaks = _strict_extrema(elevation, peaks=True)
    valleys = _strict_extrema(elevation, peaks=False)
    candidates: list[tuple[float, int, int, int]] = []
    for peak in peaks:
        upstream = nearest_valley(valleys, int(peak), upstream=True)
        downstream = nearest_valley(valleys, int(peak), upstream=False)
        if upstream is None or downstream is None:
            continue
        if t_m[upstream] <= 0.0 <= t_m[peak]:
            candidates.append((float(t_m[peak]), int(peak), upstream, downstream))
    if not candidates:
        return None
    _, peak, upstream, downstream = min(candidates, key=lambda value: value[0])
    return peak, upstream, downstream


def _source_peak_records(
    t_m: np.ndarray,
    elevation: np.ndarray,
    target_up_valley: int,
) -> tuple[list[tuple[int, int, int]], int]:
    peaks = _strict_extrema(elevation, peaks=True)
    valleys = _strict_extrema(elevation, peaks=False)
    sources: list[tuple[int, int, int]] = []
    truncated = 0
    for peak in peaks:
        if peak >= target_up_valley:
            continue
        upstream = nearest_valley(valleys, int(peak), upstream=True)
        downstream = nearest_valley(valleys, int(peak), upstream=False)
        if upstream is None or downstream is None:
            truncated += 1
            continue
        if downstream <= target_up_valley:
            sources.append((int(peak), upstream, downstream))
    sources.sort(key=lambda value: float(t_m[value[0]]))
    return sources, truncated


def _segment_median(
    profile: np.ndarray,
    first_index: int,
    second_index: int,
) -> float:
    first = min(first_index, second_index)
    last = max(first_index, second_index)
    return float(np.median(profile[first : last + 1]))


def _segment_range(
    profile: np.ndarray,
    first_index: int,
    second_index: int,
) -> dict[str, float]:
    first = min(first_index, second_index)
    last = max(first_index, second_index)
    values = np.asarray(profile[first : last + 1], dtype=np.float64)
    return {
        "median_m": float(np.median(values)),
        "minimum_m": float(np.min(values)),
        "maximum_m": float(np.max(values)),
    }


def _profile_curvature_native(
    elevation: np.ndarray,
    index: int,
    spacing_m: float,
) -> float:
    if index <= 0 or index >= len(elevation) - 1:
        return math.nan
    return float(
        (elevation[index + 1] - 2.0 * elevation[index] + elevation[index - 1])
        / spacing_m**2
    )


def _minimal_angular_width_deg(east: np.ndarray, north: np.ndarray) -> float:
    angles = np.sort(np.mod(np.arctan2(north, east), 2.0 * math.pi))
    if angles.size <= 1:
        return 0.0
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * math.pi]))
    return math.degrees(2.0 * math.pi - float(np.max(gaps)))


def _half_height_component(
    dem: np.ndarray,
    anchor_row: float,
    anchor_column: float,
    source_row: float,
    source_column: float,
    half_height_elevation_m: float,
    source_up_t_m: float,
    source_down_t_m: float,
    control_row: float,
    control_column: float,
    unit_east: float,
    unit_north: float,
    source_leeward_height_m: float,
    *,
    spacing_m: float,
) -> dict[str, float | int | bool]:
    row_grid, column_grid = np.indices(dem.shape, dtype=np.float64)
    along_t_m = (
        (column_grid - anchor_column) * spacing_m * unit_east
        + (row_grid - anchor_row) * spacing_m * unit_north
    )
    mask = (
        (np.asarray(dem, dtype=np.float64) >= half_height_elevation_m)
        & (along_t_m >= source_up_t_m - 1.0e-9)
        & (along_t_m <= source_down_t_m + 1.0e-9)
    )
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    rounded_row = int(np.clip(round(source_row), 0, dem.shape[0] - 1))
    rounded_column = int(np.clip(round(source_column), 0, dem.shape[1] - 1))
    label_value = int(labels[rounded_row, rounded_column])
    if label_value == 0:
        r0, r1 = max(0, rounded_row - 1), min(dem.shape[0], rounded_row + 2)
        c0, c1 = max(0, rounded_column - 1), min(dem.shape[1], rounded_column + 2)
        local = labels[r0:r1, c0:c1]
        local_dem = dem[r0:r1, c0:c1]
        candidate = np.where(local > 0, local_dem, -np.inf)
        if not np.any(np.isfinite(candidate)):
            return {
                "source_component_principal_span_m": math.nan,
                "source_component_crosswind_span_m": math.nan,
                "source_component_alongwind_thickness_m": math.nan,
                "source_half_height_component_span_m": math.nan,
                "source_half_height_component_aspect_ratio": math.nan,
                "source_crosswind_offset_m": math.nan,
                "source_wind_ridge_incidence_deg": math.nan,
                "source_component_tangent_east": math.nan,
                "source_component_tangent_north": math.nan,
                "source_component_centroid_row": math.nan,
                "source_component_centroid_column": math.nan,
                "source_component_angular_width_from_control_deg": math.nan,
                "source_component_cell_count": 0,
                "source_component_boundary_truncated": True,
            }
        local_index = np.unravel_index(int(np.argmax(candidate)), candidate.shape)
        label_value = int(local[local_index])
    rows, columns = np.where(labels == label_value)
    cell_count = int(rows.size)
    if cell_count < 2:
        return {
            "source_component_principal_span_m": spacing_m,
            "source_component_crosswind_span_m": spacing_m,
            "source_component_alongwind_thickness_m": spacing_m,
            "source_half_height_component_span_m": spacing_m,
            "source_half_height_component_aspect_ratio": (
                spacing_m / source_leeward_height_m
            ),
            "source_crosswind_offset_m": 0.0,
            "source_wind_ridge_incidence_deg": 90.0,
            "source_component_tangent_east": 1.0,
            "source_component_tangent_north": 0.0,
            "source_component_centroid_row": float(rows[0]),
            "source_component_centroid_column": float(columns[0]),
            "source_component_angular_width_from_control_deg": 0.0,
            "source_component_cell_count": cell_count,
            "source_component_boundary_truncated": True,
        }
    east_coordinates = columns.astype(np.float64) * spacing_m
    north_coordinates = rows.astype(np.float64) * spacing_m
    centered = np.column_stack(
        [east_coordinates - np.mean(east_coordinates), north_coordinates - np.mean(north_coordinates)]
    )
    covariance = centered.T @ centered / max(cell_count - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tangent = eigenvectors[:, int(np.argmax(eigenvalues))]
    if tangent[0] < 0.0 or (tangent[0] == 0.0 and tangent[1] < 0.0):
        tangent = -tangent
    projections = centered @ tangent
    principal_span = max(
        float(np.max(projections) - np.min(projections) + spacing_m), spacing_m
    )
    flow_along = (
        (east_coordinates - np.mean(east_coordinates)) * unit_east
        + (north_coordinates - np.mean(north_coordinates)) * unit_north
    )
    flow_cross = (
        -(east_coordinates - np.mean(east_coordinates)) * unit_north
        + (north_coordinates - np.mean(north_coordinates)) * unit_east
    )
    alongwind_thickness = max(
        float(np.max(flow_along) - np.min(flow_along) + spacing_m), spacing_m
    )
    crosswind_span = max(
        float(np.max(flow_cross) - np.min(flow_cross) + spacing_m), spacing_m
    )
    incidence = math.degrees(
        math.asin(np.clip(abs(unit_east * tangent[0] + unit_north * tangent[1]), 0.0, 1.0))
    )
    control_east = control_column * spacing_m
    control_north = control_row * spacing_m
    centroid_east = float(np.mean(east_coordinates))
    centroid_north = float(np.mean(north_coordinates))
    flow_cross_east = -unit_north
    flow_cross_north = unit_east
    crosswind_offset = (
        (control_east - centroid_east) * flow_cross_east
        + (control_north - centroid_north) * flow_cross_north
    )
    rays_east = east_coordinates - control_east
    rays_north = north_coordinates - control_north
    angular_width = _minimal_angular_width_deg(rays_east, rays_north)
    boundary = bool(
        np.any(rows == 0)
        or np.any(columns == 0)
        or np.any(rows == dem.shape[0] - 1)
        or np.any(columns == dem.shape[1] - 1)
    )
    return {
        "source_component_principal_span_m": principal_span,
        "source_component_crosswind_span_m": crosswind_span,
        "source_component_alongwind_thickness_m": alongwind_thickness,
        "source_half_height_component_span_m": crosswind_span,
        "source_half_height_component_aspect_ratio": (
            crosswind_span / source_leeward_height_m
        ),
        "source_crosswind_offset_m": float(crosswind_offset),
        "source_wind_ridge_incidence_deg": incidence,
        "source_component_tangent_east": float(tangent[0]),
        "source_component_tangent_north": float(tangent[1]),
        "source_component_centroid_row": float(np.mean(rows)),
        "source_component_centroid_column": float(np.mean(columns)),
        "source_component_angular_width_from_control_deg": angular_width,
        "source_component_cell_count": cell_count,
        "source_component_boundary_truncated": boundary,
    }


def extract_target_and_sources_once(
    dem: np.ndarray,
    roughness: np.ndarray,
    wind_east_100m: np.ndarray,
    wind_north_100m: np.ndarray,
    anchor_row: int,
    anchor_column: int,
    unit_east: float,
    unit_north: float,
    *,
    spacing_m: float = GRID_SPACING_M,
) -> dict[str, Any]:
    """Extract one target and every closed upstream source on its full transect."""

    t_m, elevation = full_map_transect(
        dem,
        anchor_row,
        anchor_column,
        unit_east,
        unit_north,
        spacing_m=spacing_m,
    )
    _, roughness_profile = full_map_transect(
        roughness,
        anchor_row,
        anchor_column,
        unit_east,
        unit_north,
        spacing_m=spacing_m,
    )
    target_indices = _target_peak_and_valleys(t_m, elevation)
    if target_indices is None:
        return {"status": "invalid_no_closed_target", "target": None, "sources": []}
    target_peak, target_up, target_down = target_indices
    target_wind = side_metrics(elevation, t_m, target_peak, target_up)
    target_lee = side_metrics(elevation, t_m, target_peak, target_down)
    control_t = float(t_m[target_up])
    control_row, control_column = point_from_t(
        anchor_row,
        anchor_column,
        control_t,
        unit_east,
        unit_north,
        spacing_m=spacing_m,
    )
    target_row, target_column = point_from_t(
        anchor_row,
        anchor_column,
        float(t_m[target_peak]),
        unit_east,
        unit_north,
        spacing_m=spacing_m,
    )
    control_east = sample_field(wind_east_100m, control_row, control_column)
    control_north = sample_field(wind_north_100m, control_row, control_column)
    control_speed = math.hypot(control_east, control_north)
    target_crest = float(elevation[target_peak])
    target_lee_height = float(target_lee["height_m"])
    target_z0 = _segment_median(roughness_profile, target_up, target_peak)
    target_native_geometry_resolved = bool(
        int(target_wind["native_interval_count"]) >= 2
        and int(target_lee["native_interval_count"]) >= 2
    )
    target_wemod_log_domain_valid = bool(target_lee_height > target_z0)
    target_component_applicable = bool(
        target_native_geometry_resolved and target_wemod_log_domain_valid
    )
    if target_component_applicable:
        source_named_component = _half_height_component(
            dem,
            anchor_row,
            anchor_column,
            target_row,
            target_column,
            target_crest - 0.5 * target_lee_height,
            float(t_m[target_up]),
            float(t_m[target_down]),
            control_row,
            control_column,
            unit_east,
            unit_north,
            target_lee_height,
            spacing_m=spacing_m,
        )
        target_component = {
            key.replace("source_", "target_", 1): value
            for key, value in source_named_component.items()
        }
    else:
        target_component = {
            "target_component_principal_span_m": math.nan,
            "target_component_crosswind_span_m": math.nan,
            "target_component_alongwind_thickness_m": math.nan,
            "target_half_height_component_span_m": math.nan,
            "target_half_height_component_aspect_ratio": math.nan,
            "target_crosswind_offset_m": math.nan,
            "target_wind_ridge_incidence_deg": math.nan,
            "target_component_tangent_east": math.nan,
            "target_component_tangent_north": math.nan,
            "target_component_centroid_row": math.nan,
            "target_component_centroid_column": math.nan,
            "target_component_angular_width_from_control_deg": math.nan,
            "target_component_cell_count": 0,
            "target_component_boundary_truncated": False,
        }
    target_crest_wind = sample_wind(
        wind_east_100m, wind_north_100m, target_row, target_column
    )
    target_tangent_east = float(target_component["target_component_tangent_east"])
    target_tangent_north = float(target_component["target_component_tangent_north"])
    if np.isfinite(target_tangent_east) and np.isfinite(target_tangent_north):
        target_normal_east = target_tangent_north
        target_normal_north = -target_tangent_east
        target_normal_wind = (
            float(target_crest_wind["east_100m"]) * target_normal_east
            + float(target_crest_wind["north_100m"]) * target_normal_north
        )
        if target_normal_wind < 0.0:
            target_normal_east = -target_normal_east
            target_normal_north = -target_normal_north
            target_normal_wind = -target_normal_wind
        target_tangent_wind = (
            float(target_crest_wind["east_100m"]) * target_tangent_east
            + float(target_crest_wind["north_100m"]) * target_tangent_north
        )
    else:
        target_normal_east = math.nan
        target_normal_north = math.nan
        target_normal_wind = math.nan
        target_tangent_wind = math.nan
    target_record: dict[str, Any] = {
        "target_crest_elevation_m": target_crest,
        **prefixed_side("target_windward", target_wind),
        **prefixed_side("target_leeward", target_lee),
        "target_control_t_from_anchor_m": control_t,
        "target_control_row": control_row,
        "target_control_column": control_column,
        "target_crest_row": target_row,
        "target_crest_column": target_column,
        "control_wind_east_100m": control_east,
        "control_wind_north_100m": control_north,
        "control_wind_speed_100m": control_speed,
        "control_wind_direction_deg": vector_direction_deg(control_east, control_north),
        "control_z0_m": sample_field(roughness, control_row, control_column),
        "target_z0_m": target_z0,
        **{
            f"target_windward_z0_{name}": value
            for name, value in _segment_range(
                roughness_profile, target_up, target_peak
            ).items()
        },
        **{
            f"target_leeward_z0_{name}": value
            for name, value in _segment_range(
                roughness_profile, target_peak, target_down
            ).items()
        },
        **{
            f"target_crest_wind_{name}": value
            for name, value in target_crest_wind.items()
        },
        "target_native_geometry_resolved": target_native_geometry_resolved,
        "target_wemod_log_domain_valid": target_wemod_log_domain_valid,
        "target_component_applicable": target_component_applicable,
        "target_ridge_normal_east": target_normal_east,
        "target_ridge_normal_north": target_normal_north,
        "target_wind_normal_to_ridge_mps": target_normal_wind,
        "target_wind_tangent_to_ridge_mps": target_tangent_wind,
        **target_component,
        "target_profile_curvature_native_per_m": _profile_curvature_native(
            elevation, target_peak, spacing_m
        ),
        "target_profile_mean_square_slope_native": float(
            np.mean(
                (
                    np.diff(elevation[target_up : target_down + 1])
                    / np.diff(t_m[target_up : target_down + 1])
                )
                ** 2
            )
        ),
        "target_local_prominence_m": min(
            float(target_wind["height_m"]), float(target_lee["height_m"])
        ),
        "profile_upwind_boundary_truncated": False,
        "profile_downwind_boundary_truncated": False,
    }
    source_indices, truncated_count = _source_peak_records(
        t_m,
        elevation,
        target_up,
    )
    target_record["profile_upwind_boundary_truncated"] = bool(truncated_count)
    sources: list[dict[str, Any]] = []
    target_crest = float(elevation[target_peak])
    target_wind_height = float(target_wind["height_m"])
    for source_index, (source_peak, source_up, source_down) in enumerate(source_indices):
        source_windward = side_metrics(elevation, t_m, source_peak, source_up)
        source_leeward = side_metrics(elevation, t_m, source_peak, source_down)
        local_prominence = min(
            float(source_windward["height_m"]),
            float(source_leeward["height_m"]),
        )
        native_geometry_resolved = bool(
            int(source_windward["native_interval_count"]) >= 2
            and int(source_leeward["native_interval_count"]) >= 2
        )
        source_t = float(t_m[source_peak])
        source_up_t = float(t_m[source_up])
        source_down_t = float(t_m[source_down])
        source_row, source_column = point_from_t(
            anchor_row,
            anchor_column,
            source_t,
            unit_east,
            unit_north,
            spacing_m=spacing_m,
        )
        source_east = sample_field(wind_east_100m, source_row, source_column)
        source_north = sample_field(wind_north_100m, source_row, source_column)
        source_speed = math.hypot(source_east, source_north)
        source_direction = vector_direction_deg(source_east, source_north)
        source_crest = float(elevation[source_peak])
        source_lee_height = float(source_leeward["height_m"])
        source_z0 = _segment_median(roughness_profile, source_peak, source_down)
        wemod_log_domain_valid = bool(source_lee_height > source_z0)
        component_applicable = bool(
            native_geometry_resolved and wemod_log_domain_valid
        )
        if component_applicable:
            component = _half_height_component(
                dem,
                anchor_row,
                anchor_column,
                source_row,
                source_column,
                source_crest - 0.5 * source_lee_height,
                source_up_t,
                source_down_t,
                control_row,
                control_column,
                unit_east,
                unit_north,
                source_lee_height,
                spacing_m=spacing_m,
            )
        else:
            component = {
                "source_component_principal_span_m": math.nan,
                "source_component_crosswind_span_m": math.nan,
                "source_component_alongwind_thickness_m": math.nan,
                "source_half_height_component_span_m": math.nan,
                "source_half_height_component_aspect_ratio": math.nan,
                "source_crosswind_offset_m": math.nan,
                "source_wind_ridge_incidence_deg": math.nan,
                "source_component_tangent_east": math.nan,
                "source_component_tangent_north": math.nan,
                "source_component_centroid_row": math.nan,
                "source_component_centroid_column": math.nan,
                "source_component_angular_width_from_control_deg": math.nan,
                "source_component_cell_count": 0,
                "source_component_boundary_truncated": False,
            }
        tangent_east = float(component["source_component_tangent_east"])
        tangent_north = float(component["source_component_tangent_north"])
        if np.isfinite(tangent_east) and np.isfinite(tangent_north):
            normal_east = tangent_north
            normal_north = -tangent_east
            normal_wind = source_east * normal_east + source_north * normal_north
            if normal_wind < 0.0:
                normal_east = -normal_east
                normal_north = -normal_north
                normal_wind = -normal_wind
            tangent_wind = source_east * tangent_east + source_north * tangent_north
        else:
            normal_east = math.nan
            normal_north = math.nan
            normal_wind = math.nan
            tangent_wind = math.nan
        source_up_row, source_up_column = point_from_t(
            anchor_row,
            anchor_column,
            source_up_t,
            unit_east,
            unit_north,
            spacing_m=spacing_m,
        )
        source_down_row, source_down_column = point_from_t(
            anchor_row,
            anchor_column,
            source_down_t,
            unit_east,
            unit_north,
            spacing_m=spacing_m,
        )
        between_first = min(source_peak, target_peak)
        between_last = max(source_peak, target_peak)
        between_valley = float(np.min(elevation[between_first : between_last + 1]))
        intervening_source_count = len(source_indices) - 1 - source_index
        nearest_source = intervening_source_count == 0
        source_record: dict[str, Any] = {
            "source_index": source_index,
            "source_order_upwind_to_downwind": source_index,
            "source_crest_elevation_m": source_crest,
            **prefixed_side("source_windward", source_windward),
            **prefixed_side("source_leeward", source_leeward),
            "source_crest_t_from_anchor_m": source_t,
            "source_lee_base_t_from_anchor_m": source_down_t,
            "source_crest_row": source_row,
            "source_crest_column": source_column,
            "source_lee_over_target_wind_height_ratio": source_lee_height / target_wind_height,
            "source_windward_over_target_windward_height_ratio": (
                float(source_windward["height_m"]) / float(target_wind["height_m"])
            ),
            "source_windward_over_target_leeward_height_ratio": (
                float(source_windward["height_m"]) / float(target_lee["height_m"])
            ),
            "source_leeward_over_target_windward_height_ratio": (
                source_lee_height / float(target_wind["height_m"])
            ),
            "source_leeward_over_target_leeward_height_ratio": (
                source_lee_height / float(target_lee["height_m"])
            ),
            "source_local_prominence_m": local_prominence,
            "source_native_geometry_resolved": native_geometry_resolved,
            "source_wemod_log_domain_valid": wemod_log_domain_valid,
            "source_component_applicable": component_applicable,
            "relative_crest_height_m": source_crest - target_crest,
            "crest_gap_m": float(t_m[target_peak] - source_t),
            "clear_valley_gap_m": (
                max(control_t - source_down_t, 0.0) if nearest_source else math.nan
            ),
            "valley_depth_from_lower_crest_m": (
                min(source_crest, target_crest) - between_valley
            ),
            "intervening_source_count": intervening_source_count,
            "source_crest_to_control_m": control_t - source_t,
            "source_lee_base_to_control_m": control_t - source_down_t,
            "source_crest_to_control_over_source_lee_h": (control_t - source_t) / source_lee_height,
            "source_z0_m": source_z0,
            **{
                f"source_windward_z0_{name}": value
                for name, value in _segment_range(
                    roughness_profile, source_up, source_peak
                ).items()
            },
            **{
                f"source_leeward_z0_{name}": value
                for name, value in _segment_range(
                    roughness_profile, source_peak, source_down
                ).items()
            },
            **{
                f"source_to_control_corridor_z0_{name}": value
                for name, value in _segment_range(
                    roughness_profile, source_peak, target_up
                ).items()
            },
            "source_z0_over_source_lee_h": source_z0 / source_lee_height,
            "source_wind_east_100m": source_east,
            "source_wind_north_100m": source_north,
            "source_wind_speed_100m": source_speed,
            "source_wind_direction_deg": source_direction,
            "source_wind_along_control_mps": source_east * unit_east + source_north * unit_north,
            "source_wind_cross_control_mps": -source_east * unit_north + source_north * unit_east,
            **{
                f"source_windward_base_wind_{name}": value
                for name, value in sample_wind(
                    wind_east_100m,
                    wind_north_100m,
                    source_up_row,
                    source_up_column,
                ).items()
            },
            **{
                f"source_leeward_base_wind_{name}": value
                for name, value in sample_wind(
                    wind_east_100m,
                    wind_north_100m,
                    source_down_row,
                    source_down_column,
                ).items()
            },
            "source_ridge_normal_east": normal_east,
            "source_ridge_normal_north": normal_north,
            "source_wind_normal_to_ridge_mps": normal_wind,
            "source_wind_tangent_to_ridge_mps": tangent_wind,
            "source_to_control_speed_change_mps": control_speed - source_speed,
            "source_to_control_direction_turn_deg": wrapped_angle_difference_deg(
                vector_direction_deg(control_east, control_north), source_direction
            ),
            "source_profile_curvature_native_per_m": _profile_curvature_native(
                elevation, source_peak, spacing_m
            ),
            "source_profile_mean_square_slope_native": float(
                np.mean(
                    (
                        np.diff(elevation[source_up : source_down + 1])
                        / np.diff(t_m[source_up : source_down + 1])
                    )
                    ** 2
                )
            ),
            **component,
        }
        sources.append(source_record)
    for index, source in enumerate(sources):
        source["previous_source_crest_gap_m"] = (
            math.nan
            if index == 0
            else float(source["source_crest_t_from_anchor_m"])
            - float(sources[index - 1]["source_crest_t_from_anchor_m"])
        )
        source["next_source_crest_gap_m"] = (
            math.nan
            if index == len(sources) - 1
            else float(sources[index + 1]["source_crest_t_from_anchor_m"])
            - float(source["source_crest_t_from_anchor_m"])
        )
    target_record["closed_upstream_source_count"] = len(sources)
    target_record["native_resolved_source_count"] = int(
        sum(bool(source["source_native_geometry_resolved"]) for source in sources)
    )
    target_record["wemod_log_domain_source_count"] = int(
        sum(bool(source["source_wemod_log_domain_valid"]) for source in sources)
    )
    return {"status": "valid", "target": target_record, "sources": sources}


def extract_target_and_sources(
    dem: np.ndarray,
    roughness: np.ndarray,
    wind_east_100m: np.ndarray,
    wind_north_100m: np.ndarray,
    anchor_row: int,
    anchor_column: int,
    *,
    spacing_m: float = GRID_SPACING_M,
) -> dict[str, Any]:
    """Two-pass extraction using the spatial wind at the target control surface."""

    initial_east = sample_field(wind_east_100m, anchor_row, anchor_column)
    initial_north = sample_field(wind_north_100m, anchor_row, anchor_column)
    initial_speed = math.hypot(initial_east, initial_north)
    if not np.isfinite(initial_speed) or initial_speed <= 0.0:
        return {"status": "invalid_zero_anchor_wind", "target": None, "sources": []}
    first = extract_target_and_sources_once(
        dem,
        roughness,
        wind_east_100m,
        wind_north_100m,
        anchor_row,
        anchor_column,
        initial_east / initial_speed,
        initial_north / initial_speed,
        spacing_m=spacing_m,
    )
    if first["status"] != "valid":
        return first
    control_speed = float(first["target"]["control_wind_speed_100m"])
    if not np.isfinite(control_speed) or control_speed <= 0.0:
        return {"status": "invalid_zero_control_wind", "target": None, "sources": []}
    return extract_target_and_sources_once(
        dem,
        roughness,
        wind_east_100m,
        wind_north_100m,
        anchor_row,
        anchor_column,
        float(first["target"]["control_wind_east_100m"]) / control_speed,
        float(first["target"]["control_wind_north_100m"]) / control_speed,
        spacing_m=spacing_m,
    )
