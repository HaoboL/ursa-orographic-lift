from __future__ import annotations

import numpy as np

from ursa import build_ridge_segment_inventory, evaluate_wemod_incident_ratios


def synthetic_inventory():
    resolution_m = 30.0
    x_m = np.arange(0.0, 2100.0, resolution_m)
    profile_m = (
        120.0 * np.exp(-0.5 * ((x_m - 600.0) / 120.0) ** 2)
        + 80.0 * np.exp(-0.5 * ((x_m - 1200.0) / 150.0) ** 2)
    )
    dem_m = np.repeat(profile_m[None, :], 41, axis=0)
    return build_ridge_segment_inventory(
        dem_m,
        np.full_like(dem_m, 0.1),
        flow_to_math_deg=0.0,
        source_resolution_m=resolution_m,
    )


def test_synthetic_inventory_and_far_ratio():
    inventory = synthetic_inventory()
    assert inventory.segment_count > 0
    assert inventory.component_count == 2

    result = evaluate_wemod_incident_ratios(
        inventory,
        query_x_m=np.asarray([1800.0]),
        query_y_m=np.asarray([600.0]),
        query_ground_elevation_m=np.asarray([0.0]),
        height_agl_m=np.asarray([100.0]),
        effective_moment_coefficient=0.8,
        combination="linear_sum",
    )
    physical = float(result.physical_far_ratio_hq[0, 0])
    assert 0.0 <= physical <= 1.0
    assert int(result.far_segment_count_hq[0, 0]) > 0


def test_no_ridge_is_identity():
    dem_m = np.zeros((21, 31), dtype=np.float64)
    inventory = build_ridge_segment_inventory(
        dem_m,
        np.full_like(dem_m, 0.1),
        flow_to_math_deg=0.0,
        source_resolution_m=30.0,
    )
    result = evaluate_wemod_incident_ratios(
        inventory,
        query_x_m=np.asarray([300.0]),
        query_y_m=np.asarray([300.0]),
        query_ground_elevation_m=np.asarray([0.0]),
        height_agl_m=np.asarray([100.0]),
        effective_moment_coefficient=0.8,
    )
    assert inventory.segment_count == 0
    assert float(result.physical_far_ratio_hq[0, 0]) == 1.0
    assert float(result.planner_safe_ratio_hq[0, 0]) == 1.0
