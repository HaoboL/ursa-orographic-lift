#!/usr/bin/env python3
"""Literature-anchored URSA V10 S/E model primitives frozen before output access.

The functions are vectorized and contain no file I/O.  They deliberately keep
the attenuation head S and the separation/exposure head E parallel: E never
multiplies the center vertical-wind estimate.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


MODEL_SCHEMA = "ursa.v10.dual-head-d1-model.v1"
LEVEL_TIED = "D1_SE_TIED_5P"
LEVEL_SPLIT = "D1_SE_SPLIT_6P"
PERERA_FAR_WAKE_X_OVER_H = 7.5
VON_KARMAN = 0.4
PERERA_PROFILE_COEFFICIENT = 0.67
PERERA_PROFILE_EXPONENT = 1.5
LIU_SLOPE_ANCHORS = np.asarray((0.32, 0.63, 1.26), dtype=np.float64)
LIU_SEPARATION_X_OVER_L_3D = np.asarray((0.21, 0.12, 0.03), dtype=np.float64)
LIU_REATTACHMENT_X_OVER_L_3D = np.asarray((0.8, 1.1, 1.4), dtype=np.float64)
SEPARATION_RAMP_SLOPE = (0.20, 0.32)
E_EVENT_MEMBERSHIP_THRESHOLD = 0.5
PARAMETER_BOUNDS = {
    "beta": (0.0, 5.0),
    "c_s": (0.0, 1.0),
    "c_w": (0.0, 1.0),
    "c_l": (0.0, 1.0),
    "c_x": (1.0, 5.1 / 1.4),
    "c_z": (0.10, 0.30),
    "c_n": (0.0, 0.20),
}


def _array(value: np.ndarray | float) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _require_finite(name: str, value: np.ndarray | float) -> np.ndarray:
    array = _array(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def validate_parameters(level: str, parameters: Mapping[str, float]) -> None:
    required = {
        LEVEL_TIED: ("beta", "c_s", "c_x", "c_z", "c_n"),
        LEVEL_SPLIT: ("beta", "c_w", "c_l", "c_x", "c_z", "c_n"),
    }
    if level not in required:
        raise ValueError(f"unknown model level: {level}")
    if set(parameters) != set(required[level]):
        raise ValueError(f"parameter names differ for {level}")
    for name, value in parameters.items():
        lower, upper = PARAMETER_BOUNDS[name]
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} lies outside its frozen bound")


def linear_anchor_interpolation(
    value: np.ndarray | float,
    anchors: np.ndarray,
    responses: np.ndarray,
) -> np.ndarray:
    x = _require_finite("anchor coordinate", value)
    return np.interp(x, anchors, responses, left=responses[0], right=responses[-1])


def slope_shape_coordinate(max_slope_tangent: np.ndarray | float) -> np.ndarray:
    """Natural-shape coordinate anchored to Liu's 0.32--1.26 slope family."""

    slope = _require_finite("maximum slope", max_slope_tangent)
    if np.any(slope < 0.0):
        raise ValueError("maximum slope must be nonnegative")
    return np.clip(
        (slope - LIU_SLOPE_ANCHORS[0])
        / (LIU_SLOPE_ANCHORS[-1] - LIU_SLOPE_ANCHORS[0]),
        0.0,
        1.0,
    )


def separation_activation(max_leeward_slope_tangent: np.ndarray | float) -> np.ndarray:
    """Conservative uncertainty ramp around the literature separation boundary."""

    slope = _require_finite("maximum leeward slope", max_leeward_slope_tangent)
    if np.any(slope < 0.0):
        raise ValueError("maximum leeward slope must be nonnegative")
    lower, upper = SEPARATION_RAMP_SLOPE
    return np.clip((slope - lower) / (upper - lower), 0.0, 1.0)


def liu_separation_x_over_l(max_leeward_slope_tangent: np.ndarray | float) -> np.ndarray:
    return linear_anchor_interpolation(
        max_leeward_slope_tangent,
        LIU_SLOPE_ANCHORS,
        LIU_SEPARATION_X_OVER_L_3D,
    )


def liu_reattachment_x_over_l_3d(
    max_leeward_slope_tangent: np.ndarray | float,
) -> np.ndarray:
    return linear_anchor_interpolation(
        max_leeward_slope_tangent,
        LIU_SLOPE_ANCHORS,
        LIU_REATTACHMENT_X_OVER_L_3D,
    )


def lateral_membership(
    crosswind_from_component_m: np.ndarray | float,
    component_crosswind_span_m: np.ndarray | float,
    downwind_from_crest_m: np.ndarray | float,
    c_n: float,
) -> np.ndarray:
    """Taylor--Salmon-style finite-obstacle Gaussian lateral membership."""

    lower, upper = PARAMETER_BOUNDS["c_n"]
    if not math.isfinite(c_n) or not lower <= c_n <= upper:
        raise ValueError("c_n lies outside its frozen bound")
    n = _require_finite("crosswind offset", crosswind_from_component_m)
    span = _array(component_crosswind_span_m)
    x = _require_finite("downwind distance", downwind_from_crest_m)
    if np.any(span <= 0.0) or np.any(np.isnan(span)) or np.any(x < 0.0):
        raise ValueError("span must be positive and downwind distance nonnegative")
    sigma = 0.5 * span + c_n * x
    return np.exp(-0.5 * np.square(n / sigma))


def perera_normalized_far_wake_recovery(
    downwind_from_crest_m: np.ndarray | float,
    source_leeward_height_m: np.ndarray | float,
    query_h_agl_m: np.ndarray | float,
    source_z0_m: np.ndarray | float,
) -> np.ndarray:
    """Perera Eq. (1) relative tail, normalized to one at x/H=7.5.

    The original fence displacement height is set to zero for the declared
    natural-terrain transfer.  The local exponent of the neutral log profile,
    1/log(H/z0), supplies Perera's approach-profile exponent.  This routine is
    only a relative recovery factor; it does not transfer fence wake amplitude.
    """

    x = _require_finite("downwind distance", downwind_from_crest_m)
    height = _require_finite("source height", source_leeward_height_m)
    h_agl = _require_finite("query AGL", query_h_agl_m)
    z0 = _require_finite("source roughness", source_z0_m)
    if np.any(x < 0.0) or np.any(height <= z0) or np.any(z0 <= 0.0) or np.any(h_agl < 0.0):
        raise ValueError("Perera transfer requires x>=0, H>z0>0 and AGL>=0")
    x, height, h_agl, z0 = np.broadcast_arrays(x, height, h_agl, z0)
    x_over_h = x / height
    result = np.ones_like(x_over_h)
    far = x_over_h >= PERERA_FAR_WAKE_X_OVER_H
    if not np.any(far):
        return result
    log_ratio = np.log(height[far] / z0[far])
    k_value = 2.0 * VON_KARMAN**2 / log_ratio
    approach_exponent = 1.0 / log_ratio
    xh = x_over_h[far]
    eta = (h_agl[far] / height[far]) * np.power(
        1.0 / (k_value * xh), 1.0 / (approach_exponent + 2.0)
    )
    eta_boundary = (h_agl[far] / height[far]) * np.power(
        1.0 / (k_value * PERERA_FAR_WAKE_X_OVER_H),
        1.0 / (approach_exponent + 2.0),
    )
    log_relative = np.log(PERERA_FAR_WAKE_X_OVER_H / xh) - (
        PERERA_PROFILE_COEFFICIENT
        * (
            np.power(eta, PERERA_PROFILE_EXPONENT)
            - np.power(eta_boundary, PERERA_PROFILE_EXPONENT)
        )
    )
    result[far] = np.exp(np.minimum(log_relative, 0.0))
    return result


def horizon_angle_rad(
    source_crest_elevation_m: np.ndarray | float,
    query_absolute_elevation_m: np.ndarray | float,
    source_to_query_horizontal_distance_m: np.ndarray | float,
) -> np.ndarray:
    crest = _require_finite("source crest elevation", source_crest_elevation_m)
    query = _require_finite("query absolute elevation", query_absolute_elevation_m)
    distance = _require_finite(
        "source-query horizontal distance", source_to_query_horizontal_distance_m
    )
    if np.any(distance <= 0.0):
        raise ValueError("source-query horizontal distance must be positive")
    return np.maximum(np.arctan2(crest - query, distance), 0.0)


def source_s_deficit(
    *,
    level: str,
    parameters: Mapping[str, float],
    source_crest_elevation_m: np.ndarray | float,
    query_absolute_elevation_m: np.ndarray | float,
    source_to_query_horizontal_distance_m: np.ndarray | float,
    downwind_from_crest_m: np.ndarray | float,
    crosswind_from_component_m: np.ndarray | float,
    component_crosswind_span_m: np.ndarray | float,
    source_wind_normal_to_ridge_mps: np.ndarray | float,
    source_wind_speed_mps: np.ndarray | float,
    source_windward_max_slope_tangent: np.ndarray | float,
    source_leeward_max_slope_tangent: np.ndarray | float,
    source_leeward_height_m: np.ndarray | float,
    source_z0_m: np.ndarray | float,
    query_h_agl_m: np.ndarray | float,
) -> np.ndarray:
    validate_parameters(level, parameters)
    x = _require_finite("downwind distance", downwind_from_crest_m)
    if np.any(x <= 0.0):
        raise ValueError("S sources must be strictly upstream of the query")
    speed = _require_finite("source wind speed", source_wind_speed_mps)
    normal = _require_finite("source normal wind", source_wind_normal_to_ridge_mps)
    if np.any(speed <= 0.0) or np.any(normal < 0.0):
        raise ValueError("source speed must be positive and normal wind nonnegative")
    incidence = np.clip(normal / speed, 0.0, 1.0)
    phi_w = slope_shape_coordinate(source_windward_max_slope_tangent)
    phi_l = slope_shape_coordinate(source_leeward_max_slope_tangent)
    if level == LEVEL_TIED:
        shape = 1.0 + parameters["c_s"] * (phi_w + phi_l)
    else:
        shape = 1.0 + parameters["c_w"] * phi_w + parameters["c_l"] * phi_l
    deficit = (
        parameters["beta"]
        * horizon_angle_rad(
            source_crest_elevation_m,
            query_absolute_elevation_m,
            source_to_query_horizontal_distance_m,
        )
        * incidence
        * shape
        * perera_normalized_far_wake_recovery(
            x, source_leeward_height_m, query_h_agl_m, source_z0_m
        )
        * lateral_membership(
            crosswind_from_component_m,
            component_crosswind_span_m,
            x,
            parameters["c_n"],
        )
    )
    return np.clip(deficit, 0.0, 1.0)


def aggregate_s_parallel(source_deficits: np.ndarray, axis: int = -1) -> np.ndarray:
    deficits = _require_finite("source deficits", source_deficits)
    if np.any((deficits < 0.0) | (deficits > 1.0)):
        raise ValueError("source deficits must lie in [0,1]")
    return 1.0 - np.minimum(1.0, np.sum(deficits, axis=axis))


def source_e_exposure(
    *,
    parameters: Mapping[str, float],
    level: str,
    downwind_from_crest_m: np.ndarray | float,
    query_h_agl_m: np.ndarray | float,
    crosswind_from_component_m: np.ndarray | float,
    component_crosswind_span_m: np.ndarray | float,
    leeward_max_slope_tangent: np.ndarray | float,
    leeward_valley_length_m: np.ndarray | float,
) -> np.ndarray:
    validate_parameters(level, parameters)
    x = _require_finite("downwind distance", downwind_from_crest_m)
    h_agl = _require_finite("query AGL", query_h_agl_m)
    length = _require_finite("leeward valley length", leeward_valley_length_m)
    slope = _require_finite("leeward maximum slope", leeward_max_slope_tangent)
    if np.any(x < 0.0) or np.any(h_agl < 0.0) or np.any(length <= 0.0):
        raise ValueError("E requires x>=0, AGL>=0 and positive leeward length")
    x, h_agl, length, slope = np.broadcast_arrays(x, h_agl, length, slope)
    x_separation = liu_separation_x_over_l(slope) * length
    x_reattachment = (
        parameters["c_x"] * liu_reattachment_x_over_l_3d(slope) * length
    )
    streamwise = ((x >= x_separation) & (x <= x_reattachment)).astype(np.float64)
    mixing_depth = parameters["c_z"] * np.maximum(x - x_separation, 0.0)
    vertical = np.zeros_like(mixing_depth)
    active = mixing_depth > 0.0
    vertical[active] = np.clip(1.0 - h_agl[active] / mixing_depth[active], 0.0, 1.0)
    lateral = lateral_membership(
        crosswind_from_component_m,
        component_crosswind_span_m,
        x,
        parameters["c_n"],
    )
    return np.minimum.reduce(
        (separation_activation(slope), streamwise, vertical, lateral)
    )


def aggregate_e(source_exposures: np.ndarray, axis: int = -1) -> np.ndarray:
    exposures = _require_finite("source exposures", source_exposures)
    if np.any((exposures < 0.0) | (exposures > 1.0)):
        raise ValueError("source exposures must lie in [0,1]")
    return np.max(exposures, axis=axis)


def e_event(exposure: np.ndarray | float) -> np.ndarray:
    value = _require_finite("E", exposure)
    if np.any((value < 0.0) | (value > 1.0)):
        raise ValueError("E must lie in [0,1]")
    return value >= E_EVENT_MEMBERSHIP_THRESHOLD


def apply_s_to_base_vertical_wind(
    base_vertical_wind_mps: np.ndarray | float,
    s_parallel: np.ndarray | float,
) -> np.ndarray:
    w0 = _require_finite("base vertical wind", base_vertical_wind_mps)
    shelter = _require_finite("S", s_parallel)
    if np.any((shelter < 0.0) | (shelter > 1.0)):
        raise ValueError("S must lie in [0,1]")
    return np.minimum(w0, 0.0) + shelter * np.maximum(w0, 0.0)
