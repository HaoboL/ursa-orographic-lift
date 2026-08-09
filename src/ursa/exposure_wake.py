#!/usr/bin/env python3
"""Counihan--Jackson rough-wall relaxing-wake similarity coordinates."""

from __future__ import annotations

import numpy as np


VON_KARMAN = 0.40


def rough_wall_similarity_parameters(
    roughness_over_height: float | np.ndarray,
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Return K=2*kappa^2/log(H/z0) and local log-law power exponent n."""

    ratio = np.asarray(roughness_over_height, dtype=np.float64)
    if np.any(~np.isfinite(ratio)) or np.any((ratio <= 0.0) | (ratio >= 1.0)):
        raise ValueError("roughness/height must lie strictly between zero and one")
    logarithm = np.log(1.0 / ratio)
    eddy_viscosity = 2.0 * VON_KARMAN**2 / logarithm
    exponent = 1.0 / logarithm
    if ratio.ndim == 0:
        return float(eddy_viscosity), float(exponent)
    return eddy_viscosity, exponent


def relaxing_wake_depth_m(
    downstream_from_obstacle_m: float | np.ndarray,
    component_height_m: float | np.ndarray,
    roughness_over_height: float | np.ndarray,
) -> float | np.ndarray:
    """Characteristic mixing-region height H[K(x/H)]^(1/(n+2))."""

    distance = np.asarray(downstream_from_obstacle_m, dtype=np.float64)
    height = np.asarray(component_height_m, dtype=np.float64)
    ratio = np.asarray(roughness_over_height, dtype=np.float64)
    if np.any(~np.isfinite(distance)) or np.any(distance <= 0.0):
        raise ValueError("wake distance must be finite and positive")
    if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("component height must be finite and positive")
    distance, height, ratio = np.broadcast_arrays(distance, height, ratio)
    eddy_viscosity, exponent = rough_wall_similarity_parameters(ratio)
    depth = height * np.power(
        eddy_viscosity * distance / height,
        1.0 / (exponent + 2.0),
    )
    return float(depth) if depth.ndim == 0 else depth


def relaxing_wake_eta(
    query_agl_m: float | np.ndarray,
    characteristic_depth_m: float | np.ndarray,
) -> float | np.ndarray:
    """Wall-wake similarity height eta=z_AGL/l_z."""

    agl = np.asarray(query_agl_m, dtype=np.float64)
    depth = np.asarray(characteristic_depth_m, dtype=np.float64)
    if np.any(~np.isfinite(agl)) or np.any(agl < 0.0):
        raise ValueError("query AGL must be finite and nonnegative")
    if np.any(~np.isfinite(depth)) or np.any(depth <= 0.0):
        raise ValueError("wake depth must be finite and positive")
    agl, depth = np.broadcast_arrays(agl, depth)
    eta = agl / depth
    return float(eta) if eta.ndim == 0 else eta


def verify_relaxing_wake_identities() -> dict[str, float | bool]:
    """Machine-readable rough-wall identities for the v1p1 firewall."""

    ratio = 1.0e-3
    eddy_viscosity, exponent = rough_wall_similarity_parameters(ratio)
    depth = relaxing_wake_depth_m(1000.0, 100.0, ratio)
    eta = relaxing_wake_eta(depth, depth)
    expected_log = np.log(1000.0)
    result = {
        "K": eddy_viscosity,
        "n": exponent,
        "depth_over_H": depth / 100.0,
        "eta_at_characteristic_depth": eta,
    }
    result["all_pass"] = bool(
        np.isclose(eddy_viscosity, 2.0 * VON_KARMAN**2 / expected_log)
        and np.isclose(exponent, 1.0 / expected_log)
        and np.isclose(eta, 1.0)
    )
    return result

