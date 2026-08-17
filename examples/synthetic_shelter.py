#!/usr/bin/env python3
"""Data-free smoke example for the public URSA terrain-shelter API."""

from __future__ import annotations

import json

import numpy as np

from ursa import build_ridge_segment_inventory, evaluate_wemod_incident_ratios


def main() -> None:
    resolution_m = 30.0
    x_m = np.arange(0.0, 2100.0, resolution_m)
    first_ridge = 120.0 * np.exp(-0.5 * ((x_m - 600.0) / 120.0) ** 2)
    second_ridge = 80.0 * np.exp(-0.5 * ((x_m - 1200.0) / 150.0) ** 2)
    profile_m = first_ridge + second_ridge
    dem_m = np.repeat(profile_m[None, :], 41, axis=0)
    roughness_m = np.full_like(dem_m, 0.1)

    inventory = build_ridge_segment_inventory(
        dem_m,
        roughness_m,
        flow_to_math_deg=0.0,
        source_resolution_m=resolution_m,
    )
    result = evaluate_wemod_incident_ratios(
        inventory,
        query_x_m=np.asarray([1800.0]),
        query_y_m=np.asarray([600.0]),
        query_ground_elevation_m=np.asarray([0.0]),
        height_agl_m=np.asarray([100.0]),
        effective_moment_coefficient=0.8,
        combination="linear_sum",
    )
    output = {
        "status": "pass",
        "ridge_segment_count": inventory.segment_count,
        "ridge_component_count": inventory.component_count,
        "physical_far_ratio": float(result.physical_far_ratio_hq[0, 0]),
        "planner_safe_ratio": float(result.planner_safe_ratio_hq[0, 0]),
        "far_segment_count": int(result.far_segment_count_hq[0, 0]),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
