"""URSA terrain-sheltering primitives."""

from .terrain_shelter import (
    FAR_MINIMUM_X_OVER_H,
    build_ridge_segment_inventory,
    evaluate_wemod_incident_ratios,
)

__all__ = [
    "FAR_MINIMUM_X_OVER_H",
    "build_ridge_segment_inventory",
    "evaluate_wemod_incident_ratios",
]

