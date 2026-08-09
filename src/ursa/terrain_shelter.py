"""Cellwise, flow-aligned terrain-shelter operator for EVVE V7.

This module accepts only terrain, roughness, wind direction, query geometry,
and AGL.  It has no FuXi-reference or EVVE-carrier input.  Ridge segments are
extracted on a native-resolution flow-aligned grid without smoothing,
prominence thresholds, fetch windows, or top-k selection.

The far-wake kernel follows the Taylor--Salmon/WEMOD equations as reproduced
in DTU Wind Energy E-Report-0092.  Terrain segmentation and the explicit
near-field applicability state are project adaptations documented in
``docs/EVVE_CELLWISE_TERRAIN_SHELTER_V7_PROTOCOL_ZH.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.ndimage import label, map_coordinates


BaseVariant = Literal["downwind_valley", "topographic_prominence"]
ShearVariant = Literal["one_seventh", "neutral_inverse_log"]
Combination = Literal[
    "linear_sum",
    "ordered_local_inflow",
    "strongest_component",
]

KAPPA = 0.4
GAMMA = 12.1875
LATERAL_SPREAD_AF = 0.5
FAR_MINIMUM_X_OVER_H = 7.5
PLANNER_NEAR_MAXIMUM_Z_OVER_H = 3.0


@dataclass(frozen=True, eq=False)
class FlowAlignedGrid:
    """A flow-aligned, bilinearly resampled input terrain."""

    dem_sn_m: np.ndarray
    roughness_sn_m: np.ndarray
    valid_sn: np.ndarray
    s_m: np.ndarray
    n_m: np.ndarray
    flow_to_math_deg: float
    source_resolution_m: float
    spacing_m: float
    s_phase_m: float
    n_phase_m: float
    source_shape_yx: tuple[int, int]


@dataclass(frozen=True, eq=False)
class RidgeSegmentInventory:
    """All resolvable closed-valley ridge segments for one terrain/wind."""

    flow_to_math_deg: float
    source_resolution_m: float
    flow_grid_spacing_m: float
    source_shape_yx: tuple[int, int]
    segment_s_m: np.ndarray
    segment_n_m: np.ndarray
    segment_width_m: np.ndarray
    crest_elevation_m: np.ndarray
    downwind_base_elevation_m: np.ndarray
    prominence_base_elevation_m: np.ndarray
    roughness_length_m: np.ndarray
    component_index: np.ndarray
    component_segment_starts: np.ndarray
    component_min_n_m: np.ndarray
    component_max_n_m: np.ndarray
    component_mean_s_m: np.ndarray
    windward_half_height_width_m: np.ndarray
    source_slope_index: np.ndarray
    open_boundary_peak_count: int
    flow_grid_shape_sn: tuple[int, int]

    @property
    def segment_count(self) -> int:
        return int(self.segment_s_m.size)

    @property
    def component_count(self) -> int:
        return int(self.component_min_n_m.size)


@dataclass(frozen=True, eq=False)
class IncidentRatioResult:
    """Far-field physical ratio and planner fail-closed ratio."""

    height_agl_m: np.ndarray
    physical_far_ratio_hq: np.ndarray
    planner_safe_ratio_hq: np.ndarray
    near_or_reattachment_ood_hq: np.ndarray
    vertical_or_roughness_ood_hq: np.ndarray
    vertical_extrapolation_hq: np.ndarray
    far_segment_count_hq: np.ndarray
    clipped_to_zero_hq: np.ndarray


def build_flow_aligned_grid(
    dem_yx_m: object,
    roughness_yx_m: object,
    *,
    flow_to_math_deg: float,
    source_resolution_m: float,
    spacing_m: float | None = None,
    s_phase_m: float = 0.0,
    n_phase_m: float = 0.0,
) -> FlowAlignedGrid:
    """Rotate inputs into a downwind/crosswind grid without smoothing.

    ``s_phase_m`` and ``n_phase_m`` expose the otherwise implicit sampling
    origin for the pre-label numerical-sensitivity gate.  Their default is
    the frozen zero-phase implementation, and each phase must lie in
    ``[0, spacing_m)``.
    """

    dem = _finite_2d(dem_yx_m, name="dem_yx_m")
    roughness = _finite_2d(roughness_yx_m, name="roughness_yx_m")
    if roughness.shape != dem.shape or np.any(roughness <= 0.0):
        raise ValueError(
            "roughness_yx_m must match DEM and be strictly positive"
        )
    source_resolution = _positive_scalar(
        source_resolution_m, name="source_resolution_m"
    )
    spacing = (
        source_resolution
        if spacing_m is None
        else _positive_scalar(spacing_m, name="spacing_m")
    )
    s_phase = _grid_phase(
        s_phase_m,
        spacing=spacing,
        name="s_phase_m",
    )
    n_phase = _grid_phase(
        n_phase_m,
        spacing=spacing,
        name="n_phase_m",
    )
    flow_deg = _finite_scalar(
        flow_to_math_deg, name="flow_to_math_deg"
    ) % 360.0
    theta = math.radians(flow_deg)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    rows, columns = dem.shape
    xmax = (columns - 1) * source_resolution
    ymax = (rows - 1) * source_resolution
    corner_x = np.asarray([0.0, xmax, xmax, 0.0], dtype=np.float64)
    corner_y = np.asarray([0.0, 0.0, ymax, ymax], dtype=np.float64)
    corner_s = corner_x * cosine + corner_y * sine
    corner_n = -corner_x * sine + corner_y * cosine

    s_m = _covering_axis(corner_s, spacing, phase=s_phase)
    n_m = _covering_axis(corner_n, spacing, phase=n_phase)
    ss, nn = np.meshgrid(s_m, n_m, indexing="xy")
    xx = ss * cosine - nn * sine
    yy = ss * sine + nn * cosine
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, xmax, ymax
    )
    valid = (
        (xx >= -tolerance)
        & (xx <= xmax + tolerance)
        & (yy >= -tolerance)
        & (yy <= ymax + tolerance)
    )
    sample_x = np.clip(xx, 0.0, xmax) / source_resolution
    sample_y = np.clip(yy, 0.0, ymax) / source_resolution
    dem_sn = map_coordinates(
        dem,
        [sample_y, sample_x],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    roughness_sn = map_coordinates(
        roughness,
        [sample_y, sample_x],
        order=1,
        mode="nearest",
        prefilter=False,
    )
    dem_sn = np.where(valid, dem_sn, np.nan)
    roughness_sn = np.where(valid, roughness_sn, np.nan)
    return FlowAlignedGrid(
        dem_sn_m=_readonly(dem_sn, np.float64),
        roughness_sn_m=_readonly(roughness_sn, np.float64),
        valid_sn=_readonly(valid, np.bool_),
        s_m=_readonly(s_m, np.float64),
        n_m=_readonly(n_m, np.float64),
        flow_to_math_deg=flow_deg,
        source_resolution_m=source_resolution,
        spacing_m=spacing,
        s_phase_m=s_phase,
        n_phase_m=n_phase,
        source_shape_yx=(rows, columns),
    )


def extract_ridge_segment_inventory(
    grid: FlowAlignedGrid,
) -> RidgeSegmentInventory:
    """Extract every closed-valley peak segment on the flow grid."""

    if not isinstance(grid, FlowAlignedGrid):
        raise TypeError("grid must be a FlowAlignedGrid")
    dem = np.asarray(grid.dem_sn_m, dtype=np.float64)
    roughness = np.asarray(grid.roughness_sn_m, dtype=np.float64)
    valid = np.asarray(grid.valid_sn, dtype=bool)
    if (
        dem.shape != roughness.shape
        or dem.shape != valid.shape
        or dem.ndim != 2
    ):
        raise ValueError("flow grid arrays are not aligned")

    peak_mask = np.zeros(dem.shape, dtype=bool)
    records: list[dict[str, float | int]] = []
    open_boundary_peak_count = 0
    for n_index in range(dem.shape[0]):
        row_valid = valid[n_index]
        for start, stop in _contiguous_true_runs(row_valid):
            profile = dem[n_index, start:stop]
            if profile.size < 3:
                continue
            peaks, valleys = _strict_local_extrema(profile)
            for peak_local in peaks:
                upstream = valleys[valleys < peak_local]
                downwind = valleys[valleys > peak_local]
                if upstream.size == 0 or downwind.size == 0:
                    open_boundary_peak_count += 1
                    continue
                upstream_local = int(upstream[-1])
                downwind_local = int(downwind[0])
                peak_index = start + int(peak_local)
                upstream_index = start + upstream_local
                downwind_index = start + downwind_local
                crest = float(dem[n_index, peak_index])
                upstream_base = float(dem[n_index, upstream_index])
                downwind_base = float(dem[n_index, downwind_index])
                prominence_base = max(upstream_base, downwind_base)
                h_downwind = crest - downwind_base
                h_prominence = crest - prominence_base
                tolerance = _height_tolerance(profile)
                if h_downwind <= tolerance or h_prominence <= tolerance:
                    continue
                windward_width = _windward_half_height_width(
                    profile,
                    peak_index=int(peak_local),
                    upstream_valley_index=upstream_local,
                    base_elevation_m=prominence_base,
                    crest_elevation_m=crest,
                    spacing_m=grid.spacing_m,
                )
                slope_index = (
                    h_prominence / windward_width
                    if windward_width > 0.0
                    else math.inf
                )
                peak_mask[n_index, peak_index] = True
                records.append(
                    {
                        "n_index": n_index,
                        "s_index": peak_index,
                        "s_m": float(grid.s_m[peak_index]),
                        "n_m": float(grid.n_m[n_index]),
                        "crest_m": crest,
                        "downwind_base_m": downwind_base,
                        "prominence_base_m": prominence_base,
                        "roughness_m": float(
                            roughness[n_index, peak_index]
                        ),
                        "windward_half_height_width_m": windward_width,
                        "source_slope_index": slope_index,
                    }
                )

    if not records:
        return _empty_inventory(grid, open_boundary_peak_count)

    component_labels, _ = label(
        peak_mask,
        structure=np.ones((3, 3), dtype=np.int8),
    )
    for record in records:
        record["raw_component"] = int(
            component_labels[
                int(record["n_index"]), int(record["s_index"])
            ]
        )
    records.sort(
        key=lambda item: (
            int(item["raw_component"]),
            float(item["n_m"]),
            float(item["s_m"]),
        )
    )
    raw_components = np.asarray(
        [int(item["raw_component"]) for item in records],
        dtype=np.int64,
    )
    _, component_index = np.unique(
        raw_components, return_inverse=True
    )
    component_index = np.asarray(component_index, dtype=np.int32)
    component_count = int(component_index.max()) + 1
    starts = np.r_[
        0,
        np.flatnonzero(np.diff(component_index)) + 1,
    ].astype(np.int64)

    def values(name: str) -> np.ndarray:
        return np.asarray(
            [float(item[name]) for item in records],
            dtype=np.float64,
        )

    segment_s = values("s_m")
    segment_n = values("n_m")
    component_min_n = np.minimum.reduceat(segment_n, starts)
    component_max_n = np.maximum.reduceat(segment_n, starts)
    component_sum_s = np.add.reduceat(segment_s, starts)
    component_sizes = np.diff(np.r_[starts, len(records)])
    component_mean_s = component_sum_s / component_sizes
    if (
        starts.size != component_count
        or not np.array_equal(
            component_index,
            np.repeat(np.arange(component_count), component_sizes),
        )
    ):
        raise RuntimeError("ridge-component relabeling became inconsistent")
    return RidgeSegmentInventory(
        flow_to_math_deg=grid.flow_to_math_deg,
        source_resolution_m=grid.source_resolution_m,
        flow_grid_spacing_m=grid.spacing_m,
        source_shape_yx=grid.source_shape_yx,
        segment_s_m=_readonly(segment_s, np.float64),
        segment_n_m=_readonly(segment_n, np.float64),
        segment_width_m=_readonly(
            np.full(len(records), grid.spacing_m), np.float64
        ),
        crest_elevation_m=_readonly(values("crest_m"), np.float64),
        downwind_base_elevation_m=_readonly(
            values("downwind_base_m"), np.float64
        ),
        prominence_base_elevation_m=_readonly(
            values("prominence_base_m"), np.float64
        ),
        roughness_length_m=_readonly(
            values("roughness_m"), np.float64
        ),
        component_index=_readonly(component_index, np.int32),
        component_segment_starts=_readonly(starts, np.int64),
        component_min_n_m=_readonly(component_min_n, np.float64),
        component_max_n_m=_readonly(component_max_n, np.float64),
        component_mean_s_m=_readonly(component_mean_s, np.float64),
        windward_half_height_width_m=_readonly(
            values("windward_half_height_width_m"), np.float64
        ),
        source_slope_index=_readonly(
            values("source_slope_index"), np.float64
        ),
        open_boundary_peak_count=open_boundary_peak_count,
        flow_grid_shape_sn=dem.shape,
    )


def build_ridge_segment_inventory(
    dem_yx_m: object,
    roughness_yx_m: object,
    *,
    flow_to_math_deg: float,
    source_resolution_m: float,
    spacing_m: float | None = None,
    s_phase_m: float = 0.0,
    n_phase_m: float = 0.0,
) -> RidgeSegmentInventory:
    """Convenience wrapper for resampling plus ridge extraction."""

    grid = build_flow_aligned_grid(
        dem_yx_m,
        roughness_yx_m,
        flow_to_math_deg=flow_to_math_deg,
        source_resolution_m=source_resolution_m,
        spacing_m=spacing_m,
        s_phase_m=s_phase_m,
        n_phase_m=n_phase_m,
    )
    return extract_ridge_segment_inventory(grid)


def evaluate_wemod_incident_ratios(
    inventory: RidgeSegmentInventory,
    *,
    query_x_m: object,
    query_y_m: object,
    query_ground_elevation_m: object,
    height_agl_m: object,
    effective_moment_coefficient: float,
    base_variant: BaseVariant = "downwind_valley",
    shear_variant: ShearVariant = "one_seventh",
    combination: Combination = "linear_sum",
    query_chunk_size: int = 128,
) -> IncidentRatioResult:
    """Evaluate WEMOD far shelter and the separate planner-safe ratio."""

    if not isinstance(inventory, RidgeSegmentInventory):
        raise TypeError("inventory must be a RidgeSegmentInventory")
    if base_variant not in {
        "downwind_valley",
        "topographic_prominence",
    }:
        raise ValueError("unknown base_variant")
    if shear_variant not in {
        "one_seventh",
        "neutral_inverse_log",
    }:
        raise ValueError("unknown shear_variant")
    if combination not in {
        "linear_sum",
        "ordered_local_inflow",
        "strongest_component",
    }:
        raise ValueError("unknown combination")
    coefficient = _finite_scalar(
        effective_moment_coefficient,
        name="effective_moment_coefficient",
    )
    if not 0.0 <= coefficient <= 0.8:
        raise ValueError(
            "effective_moment_coefficient must lie in [0, 0.8]"
        )
    if (
        isinstance(query_chunk_size, bool)
        or int(query_chunk_size) != query_chunk_size
        or int(query_chunk_size) <= 0
    ):
        raise ValueError("query_chunk_size must be a positive integer")
    chunk_size = int(query_chunk_size)
    query_x = _finite_1d(query_x_m, name="query_x_m")
    query_y = _finite_1d(query_y_m, name="query_y_m")
    query_ground = _finite_1d(
        query_ground_elevation_m,
        name="query_ground_elevation_m",
    )
    if query_y.shape != query_x.shape or query_ground.shape != query_x.shape:
        raise ValueError("query coordinate arrays must have equal length")
    heights = _finite_1d(height_agl_m, name="height_agl_m")
    if np.any(heights < 0.0):
        raise ValueError("height_agl_m must be nonnegative")
    height_count = heights.size
    query_count = query_x.size
    output_shape = (height_count, query_count)
    physical = np.ones(output_shape, dtype=np.float32)
    planner = np.ones(output_shape, dtype=np.float32)
    near_ood = np.zeros(output_shape, dtype=bool)
    vertical_ood = np.zeros(output_shape, dtype=bool)
    vertical_extrapolation = np.zeros(output_shape, dtype=bool)
    far_count = np.zeros(output_shape, dtype=np.int32)
    clipped = np.zeros(output_shape, dtype=bool)
    if inventory.segment_count == 0 or query_count == 0:
        return _freeze_result(
            heights,
            physical,
            planner,
            near_ood,
            vertical_ood,
            vertical_extrapolation,
            far_count,
            clipped,
        )

    theta = math.radians(inventory.flow_to_math_deg)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    query_s = query_x * cosine + query_y * sine
    query_n = -query_x * sine + query_y * cosine
    segment_s = inventory.segment_s_m
    segment_n = inventory.segment_n_m
    segment_width = inventory.segment_width_m
    roughness = inventory.roughness_length_m
    base = (
        inventory.downwind_base_elevation_m
        if base_variant == "downwind_valley"
        else inventory.prominence_base_elevation_m
    )
    source_height = inventory.crest_elevation_m - base
    starts = inventory.component_segment_starts
    component_min_n = inventory.component_min_n_m
    component_max_n = inventory.component_max_n_m

    for begin in range(0, query_count, chunk_size):
        end = min(query_count, begin + chunk_size)
        qs = query_s[begin:end]
        qn = query_n[begin:end]
        qground = query_ground[begin:end]
        alongwind = qs[:, None] - segment_s[None, :]
        crosswind = qn[:, None] - segment_n[None, :]
        upstream = alongwind > 0.0
        xi = np.divide(
            alongwind,
            source_height[None, :],
            out=np.full_like(alongwind, np.nan),
            where=source_height[None, :] > 0.0,
        )
        far_geometry = upstream & (xi >= FAR_MINIMUM_X_OVER_H)
        component_covers_query = (
            (qn[:, None] >= component_min_n[None, :])
            & (qn[:, None] <= component_max_n[None, :])
        )
        lambda_value = np.zeros_like(alongwind)
        np.divide(
            crosswind,
            source_height[None, :],
            out=lambda_value,
            where=source_height[None, :] > 0.0,
        )
        inverse_sqrt_xi = np.zeros_like(xi)
        inverse_sqrt_xi[far_geometry] = xi[far_geometry] ** -0.5
        lambda_value *= inverse_sqrt_xi
        lateral = (
            np.exp(
                -lambda_value**2
                / (2.0 * LATERAL_SPREAD_AF**2)
            )
            / (
                math.sqrt(2.0 * math.pi)
                * LATERAL_SPREAD_AF
            )
        )
        geometry = np.zeros_like(alongwind)
        inverse_xi_three_halves = np.zeros_like(xi)
        inverse_xi_three_halves[far_geometry] = (
            xi[far_geometry] ** -1.5
        )
        geometry = (
            (segment_width[None, :] / source_height[None, :])
            * inverse_xi_three_halves
            * lateral
        )

        for height_index, height in enumerate(heights):
            z_query = (
                qground[:, None] + float(height) - base[None, :]
            )
            log_valid = (
                (source_height[None, :] > roughness[None, :])
                & (z_query > roughness[None, :])
            )
            far_valid = far_geometry & log_valid
            z_over_h = np.divide(
                z_query,
                source_height[None, :],
                out=np.full_like(z_query, np.nan),
                where=source_height[None, :] > 0.0,
            )
            near_segment = (
                upstream
                & (xi < FAR_MINIMUM_X_OVER_H)
                & (z_over_h > 0.0)
                & (
                    z_over_h
                    < PLANNER_NEAR_MAXIMUM_Z_OVER_H
                )
            )
            near_component = np.maximum.reduceat(
                near_segment.astype(np.int8),
                starts,
                axis=1,
            ).astype(bool)
            near_here = np.any(
                near_component & component_covers_query,
                axis=1,
            )
            any_far_geometry = np.any(far_geometry, axis=1)
            any_far_valid = np.any(far_valid, axis=1)
            invalid_here = any_far_geometry & ~any_far_valid
            within_primary_vertical_band = far_valid & (
                z_over_h
                <= PLANNER_NEAR_MAXIMUM_Z_OVER_H
            )
            vertical_here = any_far_valid & ~np.any(
                within_primary_vertical_band,
                axis=1,
            )
            if shear_variant == "one_seventh":
                shear_exponent = np.full_like(
                    z_query, 1.0 / 7.0
                )
            else:
                shear_exponent = np.zeros_like(z_query)
                shear_exponent[log_valid] = 1.0 / np.log(
                    z_query[log_valid]
                    / np.broadcast_to(
                        roughness[None, :], z_query.shape
                    )[log_valid]
                )
            ca = np.zeros_like(z_query)
            ca[log_valid] = (
                np.log(
                    np.broadcast_to(
                        source_height[None, :], z_query.shape
                    )[log_valid]
                    / np.broadcast_to(
                        roughness[None, :], z_query.shape
                    )[log_valid]
                )
                / (2.0 * KAPPA**2)
                ** (1.0 / (shear_exponent[log_valid] + 2.0))
            )
            zeta = np.zeros_like(z_query)
            zeta[far_valid] = (
                z_over_h[far_valid]
                * xi[far_valid]
                ** (
                    -1.0
                    / (shear_exponent[far_valid] + 2.0)
                )
            )
            vertical_shape = np.zeros_like(z_query)
            vertical_shape[far_valid] = (
                ca[far_valid]
                * zeta[far_valid]
                * np.exp(
                    -0.67
                    * ca[far_valid] ** 1.5
                    * zeta[far_valid] ** 1.5
                )
            )
            speed_ratio = np.zeros_like(z_query)
            speed_ratio[log_valid] = np.log(
                np.broadcast_to(
                    source_height[None, :], z_query.shape
                )[log_valid]
                / np.broadcast_to(
                    roughness[None, :], z_query.shape
                )[log_valid]
            ) / np.log(
                z_query[log_valid]
                / np.broadcast_to(
                    roughness[None, :], z_query.shape
                )[log_valid]
            )
            segment_deficit = (
                GAMMA
                * coefficient
                * geometry
                * vertical_shape
                * speed_ratio
            )
            segment_deficit = np.where(
                far_valid,
                np.maximum(segment_deficit, 0.0),
                0.0,
            )
            component_deficit = np.add.reduceat(
                segment_deficit,
                starts,
                axis=1,
            )
            if combination == "linear_sum":
                unbounded_deficit = np.sum(
                    component_deficit, axis=1
                )
                ratio = np.maximum(0.0, 1.0 - unbounded_deficit)
                clipped_here = unbounded_deficit >= 1.0
            elif combination == "ordered_local_inflow":
                bounded_component = np.clip(
                    component_deficit, 0.0, 1.0
                )
                ratio = np.prod(1.0 - bounded_component, axis=1)
                clipped_here = np.any(
                    component_deficit >= 1.0, axis=1
                )
            else:
                maximum = np.max(component_deficit, axis=1)
                ratio = 1.0 - np.clip(maximum, 0.0, 1.0)
                clipped_here = maximum >= 1.0
            physical[height_index, begin:end] = ratio.astype(
                np.float32
            )
            planner[height_index, begin:end] = np.where(
                near_here, 0.0, ratio
            ).astype(np.float32)
            near_ood[height_index, begin:end] = near_here
            vertical_ood[height_index, begin:end] = invalid_here
            vertical_extrapolation[
                height_index, begin:end
            ] = vertical_here
            far_count[height_index, begin:end] = np.count_nonzero(
                far_valid, axis=1
            )
            clipped[height_index, begin:end] = clipped_here
    return _freeze_result(
        heights,
        physical,
        planner,
        near_ood,
        vertical_ood,
        vertical_extrapolation,
        far_count,
        clipped,
    )


def _freeze_result(
    heights: np.ndarray,
    physical: np.ndarray,
    planner: np.ndarray,
    near_ood: np.ndarray,
    vertical_ood: np.ndarray,
    vertical_extrapolation: np.ndarray,
    far_count: np.ndarray,
    clipped: np.ndarray,
) -> IncidentRatioResult:
    return IncidentRatioResult(
        height_agl_m=_readonly(heights, np.float64),
        physical_far_ratio_hq=_readonly(physical, np.float32),
        planner_safe_ratio_hq=_readonly(planner, np.float32),
        near_or_reattachment_ood_hq=_readonly(
            near_ood, np.bool_
        ),
        vertical_or_roughness_ood_hq=_readonly(
            vertical_ood, np.bool_
        ),
        vertical_extrapolation_hq=_readonly(
            vertical_extrapolation, np.bool_
        ),
        far_segment_count_hq=_readonly(far_count, np.int32),
        clipped_to_zero_hq=_readonly(clipped, np.bool_),
    )


def _empty_inventory(
    grid: FlowAlignedGrid,
    open_boundary_peak_count: int,
) -> RidgeSegmentInventory:
    empty_float = np.empty(0, dtype=np.float64)
    empty_int = np.empty(0, dtype=np.int32)
    empty_start = np.empty(0, dtype=np.int64)
    return RidgeSegmentInventory(
        flow_to_math_deg=grid.flow_to_math_deg,
        source_resolution_m=grid.source_resolution_m,
        flow_grid_spacing_m=grid.spacing_m,
        source_shape_yx=grid.source_shape_yx,
        segment_s_m=_readonly(empty_float, np.float64),
        segment_n_m=_readonly(empty_float, np.float64),
        segment_width_m=_readonly(empty_float, np.float64),
        crest_elevation_m=_readonly(empty_float, np.float64),
        downwind_base_elevation_m=_readonly(empty_float, np.float64),
        prominence_base_elevation_m=_readonly(
            empty_float, np.float64
        ),
        roughness_length_m=_readonly(empty_float, np.float64),
        component_index=_readonly(empty_int, np.int32),
        component_segment_starts=_readonly(
            empty_start, np.int64
        ),
        component_min_n_m=_readonly(empty_float, np.float64),
        component_max_n_m=_readonly(empty_float, np.float64),
        component_mean_s_m=_readonly(empty_float, np.float64),
        windward_half_height_width_m=_readonly(
            empty_float, np.float64
        ),
        source_slope_index=_readonly(empty_float, np.float64),
        open_boundary_peak_count=open_boundary_peak_count,
        flow_grid_shape_sn=grid.dem_sn_m.shape,
    )


def _covering_axis(
    values: np.ndarray,
    spacing: float,
    *,
    phase: float,
) -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float64) - phase) / spacing
    tolerance = 128.0 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(scaled)))
    )
    lower = math.floor(float(np.min(scaled)) + tolerance)
    upper = math.ceil(float(np.max(scaled)) - tolerance)
    return (
        np.arange(lower, upper + 1, dtype=np.float64) * spacing
        + phase
    )


def _grid_phase(value: object, *, spacing: float, name: str) -> float:
    phase = _finite_scalar(value, name=name)
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, spacing)
    if phase < -tolerance or phase >= spacing - tolerance:
        raise ValueError(f"{name} must lie in [0, spacing_m)")
    return 0.0 if abs(phase) <= tolerance else phase


def _contiguous_true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    padded = np.r_[False, values, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _strict_local_extrema(
    profile: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return plateau-safe peak and valley indices in O(N)."""

    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 1 or values.size < 3 or not np.all(np.isfinite(values)):
        raise ValueError("profile must be a finite one-dimensional array")
    tolerance = _height_tolerance(values)
    starts = [0]
    ends: list[int] = []
    run_values: list[float] = []
    for index in range(1, values.size):
        if abs(float(values[index] - values[index - 1])) > tolerance:
            ends.append(index - 1)
            run_values.append(
                float(np.mean(values[starts[-1] : index]))
            )
            starts.append(index)
    ends.append(values.size - 1)
    run_values.append(float(np.mean(values[starts[-1] :])))
    representatives = np.asarray(
        [
            (start + end) // 2
            for start, end in zip(starts, ends, strict=True)
        ],
        dtype=np.int32,
    )
    run = np.asarray(run_values, dtype=np.float64)
    if run.size < 3:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    peaks = representatives[1:-1][
        (run[1:-1] > run[:-2]) & (run[1:-1] > run[2:])
    ]
    interior_valleys = representatives[1:-1][
        (run[1:-1] < run[:-2]) & (run[1:-1] < run[2:])
    ]
    endpoint_valleys: list[int] = []
    # A flat low ground run that reaches the DEM boundary is still a
    # resolved obstacle base.  This does not admit an edge peak: endpoint
    # maxima remain excluded because their far-side descent is unknown.
    if run[0] < run[1]:
        endpoint_valleys.append(int(representatives[0]))
    if run[-1] < run[-2]:
        endpoint_valleys.append(int(representatives[-1]))
    valleys = np.sort(
        np.r_[
            interior_valleys,
            np.asarray(endpoint_valleys, dtype=np.int32),
        ]
    ).astype(np.int32)
    return peaks, valleys


def _windward_half_height_width(
    profile: np.ndarray,
    *,
    peak_index: int,
    upstream_valley_index: int,
    base_elevation_m: float,
    crest_elevation_m: float,
    spacing_m: float,
) -> float:
    target = base_elevation_m + 0.5 * (
        crest_elevation_m - base_elevation_m
    )
    values = np.asarray(profile, dtype=np.float64)
    for right in range(peak_index, upstream_valley_index, -1):
        left = right - 1
        low = float(values[left])
        high = float(values[right])
        if (low <= target <= high) or (high <= target <= low):
            if high == low:
                crossing = float(left)
            else:
                crossing = left + (target - low) / (high - low)
            return max((peak_index - crossing) * spacing_m, 0.0)
    return max(
        (peak_index - upstream_valley_index) * spacing_m,
        0.0,
    )


def _height_tolerance(values: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(values))))
    return 1.0e-10 * scale


def _finite_2d(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (
        result.ndim != 2
        or min(result.shape) < 3
        or not np.all(np.isfinite(result))
    ):
        raise ValueError(f"{name} must be a finite 2-D array")
    return result


def _finite_1d(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 1-D array")
    return result


def _finite_scalar(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: object, *, name: str) -> float:
    result = _finite_scalar(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _readonly(value: object, dtype: np.dtype | type) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    result.setflags(write=False)
    return result


__all__ = [
    "FAR_MINIMUM_X_OVER_H",
    "GAMMA",
    "KAPPA",
    "LATERAL_SPREAD_AF",
    "PLANNER_NEAR_MAXIMUM_Z_OVER_H",
    "FlowAlignedGrid",
    "IncidentRatioResult",
    "RidgeSegmentInventory",
    "build_flow_aligned_grid",
    "build_ridge_segment_inventory",
    "evaluate_wemod_incident_ratios",
    "extract_ridge_segment_inventory",
]
