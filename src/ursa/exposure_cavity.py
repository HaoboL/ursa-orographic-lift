#!/usr/bin/env python3
"""URSA V10 D4 directional-state extension of the D3 dual-head model."""

from __future__ import annotations

import math

import numpy as np


LEVEL_D4 = "D4_SE_DIRECTIONAL_STATE_10P"
BLOCKAGE_AMPLITUDE_GRID = np.linspace(0.0, 1.0, 5)
TARGET_INCIDENCE_EXPONENT_GRID = np.asarray((0.50, 0.75, 1.00))
E_DIRECTIONAL_MIX_GRID = np.linspace(0.0, 1.0, 5)


def target_directional_incidence_response(
    incidence_ratio: float | np.ndarray,
    incidence_exponent: float,
) -> float | np.ndarray:
    """Nested B/C target response: D3 is recovered at exponent one."""

    incidence = np.asarray(incidence_ratio, dtype=np.float64)
    if np.any(~np.isfinite(incidence)) or np.any((incidence < 0.0) | (incidence > 1.0)):
        raise ValueError("incidence ratio must lie in [0,1]")
    if (
        not math.isfinite(incidence_exponent)
        or incidence_exponent < 0.50
        or incidence_exponent > 1.00
    ):
        raise ValueError("target incidence exponent lies outside D4 bounds")
    response = np.power(incidence, incidence_exponent)
    return float(response) if response.ndim == 0 else response


def e_directional_separation_state(
    incidence_ratio: float | np.ndarray,
    directional_mix: float,
) -> float | np.ndarray:
    """Blend D3 activation with the ridge-normal dynamic-pressure fraction."""

    incidence = np.asarray(incidence_ratio, dtype=np.float64)
    if np.any(~np.isfinite(incidence)) or np.any((incidence < 0.0) | (incidence > 1.0)):
        raise ValueError("incidence ratio must lie in [0,1]")
    if not math.isfinite(directional_mix) or not 0.0 <= directional_mix <= 1.0:
        raise ValueError("E directional mix lies outside D4 bounds")
    state = (1.0 - directional_mix) + directional_mix * np.square(incidence)
    return float(state) if state.ndim == 0 else state


def target_spectral_blockage_core(
    *,
    target_windward_height_m: float | np.ndarray,
    target_windward_length_m: float | np.ndarray,
    query_h_agl_m: float | np.ndarray,
    incidence_ratio: float | np.ndarray,
    upstream_recovery: float | np.ndarray,
    incidence_exponent: float = 1.0,
) -> float | np.ndarray:
    """Bounded B/C hill-wavenumber response used by the D4 target S branch."""

    height = np.asarray(target_windward_height_m, dtype=np.float64)
    length = np.asarray(target_windward_length_m, dtype=np.float64)
    h_agl = np.asarray(query_h_agl_m, dtype=np.float64)
    incidence = np.asarray(incidence_ratio, dtype=np.float64)
    recovery = np.asarray(upstream_recovery, dtype=np.float64)
    if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("target windward height must be finite and positive")
    if np.any(~np.isfinite(length)) or np.any(length <= 0.0):
        raise ValueError("target windward length must be finite and positive")
    if np.any(~np.isfinite(h_agl)) or np.any(h_agl < 0.0):
        raise ValueError("query AGL must be finite and nonnegative")
    if np.any(~np.isfinite(recovery)) or np.any((recovery < 0.0) | (recovery > 1.0)):
        raise ValueError("upstream recovery must lie in [0,1]")
    height, length, h_agl, incidence, recovery = np.broadcast_arrays(
        height, length, h_agl, incidence, recovery
    )
    wavenumber = np.pi / length
    response = (
        target_directional_incidence_response(incidence, incidence_exponent)
        * recovery
        * np.minimum(wavenumber * height, 1.0)
        * np.exp(-wavenumber * h_agl)
    )
    return float(response) if response.ndim == 0 else response


def cosine_squared_separation_elevation_m(
    *,
    leeward_base_elevation_m: float | np.ndarray,
    component_height_m: float | np.ndarray,
    leeward_length_m: float | np.ndarray,
    separation_distance_m: float | np.ndarray,
) -> float | np.ndarray:
    """Liu-family cos-squared equivalent elevation at the separation point."""

    base = np.asarray(leeward_base_elevation_m, dtype=np.float64)
    height = np.asarray(component_height_m, dtype=np.float64)
    length = np.asarray(leeward_length_m, dtype=np.float64)
    separation = np.asarray(separation_distance_m, dtype=np.float64)
    if np.any(~np.isfinite(base)):
        raise ValueError("leeward base elevation must be finite")
    if np.any(~np.isfinite(height)) or np.any(height <= 0.0):
        raise ValueError("component height must be finite and positive")
    if np.any(~np.isfinite(length)) or np.any(length <= 0.0):
        raise ValueError("leeward length must be finite and positive")
    if np.any(~np.isfinite(separation)) or np.any(separation < 0.0):
        raise ValueError("separation distance must be finite and nonnegative")
    base, height, length, separation = np.broadcast_arrays(
        base, height, length, separation
    )
    fraction = np.clip(separation / length, 0.0, 1.0)
    elevation = base + height * np.square(np.cos(0.5 * np.pi * fraction))
    return float(elevation) if elevation.ndim == 0 else elevation


def terrain_cavity_vertical_membership(
    *,
    downwind_distance_m: float | np.ndarray,
    separation_distance_m: float | np.ndarray,
    reattachment_distance_m: float | np.ndarray,
    separation_elevation_m: float | np.ndarray,
    leeward_base_elevation_m: float | np.ndarray,
    query_absolute_elevation_m: float | np.ndarray,
    c_z: float,
) -> float | np.ndarray:
    """Old-article z10--z90 membership evaluated in absolute DEM elevation."""

    x = np.asarray(downwind_distance_m, dtype=np.float64)
    x_s = np.asarray(separation_distance_m, dtype=np.float64)
    x_r = np.asarray(reattachment_distance_m, dtype=np.float64)
    z_s = np.asarray(separation_elevation_m, dtype=np.float64)
    z_b = np.asarray(leeward_base_elevation_m, dtype=np.float64)
    z_q = np.asarray(query_absolute_elevation_m, dtype=np.float64)
    arrays = (x, x_s, x_r, z_s, z_b, z_q)
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise ValueError("terrain-cavity coordinates must be finite")
    if np.any(x < 0.0) or np.any(x_s < 0.0) or np.any(x_r <= x_s):
        raise ValueError("terrain-cavity distances require x>=0 and x_r>x_s")
    if not math.isfinite(c_z) or c_z < 0.10 or c_z > 0.30:
        raise ValueError("c_z lies outside the frozen D1 bounds")
    x, x_s, x_r, z_s, z_b, z_q = np.broadcast_arrays(
        x, x_s, x_r, z_s, z_b, z_q
    )
    progress = np.clip((x - x_s) / (x_r - x_s), 0.0, 1.0)
    z10 = z_s + (z_b - z_s) * progress
    thickness = c_z * np.maximum(x - x_s, 0.0)
    z90 = z10 + thickness
    membership = np.zeros_like(z90)
    active = thickness > 0.0
    membership[active] = np.clip(
        (z90[active] - z_q[active]) / thickness[active], 0.0, 1.0
    )
    return float(membership) if membership.ndim == 0 else membership


def validate_d4_parameters(parameters: dict[str, float]) -> None:
    expected = {
        "beta",
        "c_w",
        "c_l",
        "c_t",
        "c_b",
        "p_s",
        "c_x",
        "c_z",
        "c_n",
        "c_i",
    }
    if set(parameters) != expected:
        raise ValueError("D4 parameter inventory changed")
    bounds = {
        "beta": (0.0, 5.0),
        "c_w": (0.0, 1.0),
        "c_l": (0.0, 1.0),
        "c_t": (0.0, 1.0),
        "c_b": (0.0, 1.0),
        "p_s": (0.50, 1.00),
        "c_x": (1.0, 5.1 / 1.4),
        "c_z": (0.10, 0.30),
        "c_n": (0.0, 0.20),
        "c_i": (0.0, 1.0),
    }
    for name, (lower, upper) in bounds.items():
        value = float(parameters[name])
        if not math.isfinite(value) or value < lower or value > upper:
            raise ValueError(f"D4 parameter outside frozen bounds: {name}")
