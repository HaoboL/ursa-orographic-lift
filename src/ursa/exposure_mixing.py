#!/usr/bin/env python3
"""Literature-anchored ML1 slope, mixing-layer, and recovery relations."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import erf


SLOPE_ATTACHED = 0.20
SLOPE_SEPARATED = 0.30
MIXING_GROWTH = 0.10
VELOCITY_ERF_SCALE = 1.40


def separation_activation(
    leeward_mean_slope: float | np.ndarray,
    incidence_ratio: float | np.ndarray,
) -> float | np.ndarray:
    """Finnigan/Karim slope transition times ridge-normal dynamic fraction."""

    slope = np.asarray(leeward_mean_slope, dtype=np.float64)
    incidence = np.asarray(incidence_ratio, dtype=np.float64)
    if np.any(~np.isfinite(slope)) or np.any(slope < 0.0):
        raise ValueError("leeward slope must be finite and nonnegative")
    if np.any(~np.isfinite(incidence)) or np.any((incidence < 0.0) | (incidence > 1.0)):
        raise ValueError("incidence ratio must lie in [0,1]")
    state = np.clip(
        (slope - SLOPE_ATTACHED) / (SLOPE_SEPARATED - SLOPE_ATTACHED),
        0.0,
        1.0,
    ) * np.square(incidence)
    return float(state) if state.ndim == 0 else state


def mixing_layer_thickness_m(
    downstream_distance_m: float | np.ndarray,
    upper_velocity_mps: float | np.ndarray,
    lower_velocity_mps: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Karim/Dimotakis 10--90% thickness: delta=0.1*(Us/Uc)*x."""

    distance = np.asarray(downstream_distance_m, dtype=np.float64)
    upper = np.asarray(upper_velocity_mps, dtype=np.float64)
    lower = np.asarray(lower_velocity_mps, dtype=np.float64)
    if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
        raise ValueError("downstream distance must be finite and nonnegative")
    if np.any(~np.isfinite(upper)) or np.any(~np.isfinite(lower)):
        raise ValueError("mixing-layer velocities must be finite")
    distance, upper, lower = np.broadcast_arrays(distance, upper, lower)
    shear = upper - lower
    convective = 0.5 * (upper + lower)
    if np.any(shear < 0.0) or np.any(convective <= 0.0):
        raise ValueError("mixing-layer velocities require Ut>=Ub and Uc>0")
    thickness = MIXING_GROWTH * (shear / convective) * distance
    return float(thickness) if thickness.ndim == 0 else thickness


def mixing_layer_zeta(
    query_elevation_m: float | np.ndarray,
    lower_edge_elevation_m: float | np.ndarray,
    thickness_m: float | np.ndarray,
) -> float | np.ndarray:
    """Similarity coordinate with zc at the midpoint of the 10--90% layer."""

    query = np.asarray(query_elevation_m, dtype=np.float64)
    lower = np.asarray(lower_edge_elevation_m, dtype=np.float64)
    thickness = np.asarray(thickness_m, dtype=np.float64)
    if any(np.any(~np.isfinite(value)) for value in (query, lower, thickness)):
        raise ValueError("mixing-layer coordinates must be finite")
    if np.any(thickness <= 0.0):
        raise ValueError("mixing-layer thickness must be positive")
    query, lower, thickness = np.broadcast_arrays(query, lower, thickness)
    zeta = (query - (lower + 0.5 * thickness)) / thickness
    return float(zeta) if zeta.ndim == 0 else zeta


def mixing_layer_velocity_mps(
    zeta: float | np.ndarray,
    upper_velocity_mps: float | np.ndarray,
    lower_velocity_mps: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Bell--Mehta profile as fitted by Karim et al., clipped to stream bounds."""

    coordinate = np.asarray(zeta, dtype=np.float64)
    upper = np.asarray(upper_velocity_mps, dtype=np.float64)
    lower = np.asarray(lower_velocity_mps, dtype=np.float64)
    if any(np.any(~np.isfinite(value)) for value in (coordinate, upper, lower)):
        raise ValueError("velocity-profile inputs must be finite")
    if np.any(upper < lower):
        raise ValueError("upper velocity must be no smaller than lower velocity")
    coordinate, upper, lower = np.broadcast_arrays(coordinate, upper, lower)
    convective = 0.5 * (upper + lower)
    shear = upper - lower
    velocity = np.clip(
        convective + shear * erf(coordinate) / VELOCITY_ERF_SCALE,
        lower,
        upper,
    )
    return float(velocity) if velocity.ndim == 0 else velocity


def post_reattachment_decay(
    downstream_from_reattachment_m: float | np.ndarray,
    component_height_m: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Jackson mean-deficit and excess-turbulence relaxation anchors."""

    distance = np.asarray(downstream_from_reattachment_m, dtype=np.float64)
    height = np.asarray(component_height_m, dtype=np.float64)
    if np.any(~np.isfinite(distance)) or np.any(distance < 0.0):
        raise ValueError("post-reattachment distance must be finite and nonnegative")
    if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("component height must be finite and positive")
    distance, height = np.broadcast_arrays(distance, height)
    scaled = 1.0 + distance / height
    mean_deficit = np.power(scaled, -1.0)
    turbulence_excess = np.power(scaled, -1.5)
    if mean_deficit.ndim == 0:
        return float(mean_deficit), float(turbulence_excess)
    return mean_deficit, turbulence_excess


def verify_reference_identities() -> dict[str, float | bool]:
    """Machine-readable identities used by the input-only firewall."""

    delta = mixing_layer_thickness_m(5.0, 10.0, 0.0)
    centre = mixing_layer_velocity_mps(0.0, 10.0, 0.0)
    mean, turbulence = post_reattachment_decay(10.0, 10.0)
    result = {
        "attached_endpoint": separation_activation(0.20, 1.0),
        "separated_endpoint": separation_activation(0.30, 1.0),
        "delta_over_x_Ub0": delta / 5.0,
        "centre_velocity_fraction_Ub0": centre / 10.0,
        "mean_deficit_at_one_H": mean,
        "turbulence_excess_at_one_H": turbulence,
    }
    result["all_pass"] = bool(
        math.isclose(float(result["attached_endpoint"]), 0.0, abs_tol=1.0e-12)
        and math.isclose(float(result["separated_endpoint"]), 1.0, abs_tol=1.0e-12)
        and math.isclose(float(result["delta_over_x_Ub0"]), 0.2, abs_tol=1.0e-12)
        and math.isclose(
            float(result["centre_velocity_fraction_Ub0"]), 0.5, abs_tol=1.0e-12
        )
        and math.isclose(float(result["mean_deficit_at_one_H"]), 0.5, abs_tol=1.0e-12)
        and math.isclose(
            float(result["turbulence_excess_at_one_H"]),
            2.0 ** -1.5,
            abs_tol=1.0e-12,
        )
    )
    return result

