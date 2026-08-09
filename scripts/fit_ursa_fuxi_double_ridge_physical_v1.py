#!/usr/bin/env python3
"""Fit a small, explicit fluid correction on the frozen FuXi two-ridge panel.

No machine-learning package is used.  The candidate family only modulates the
original URSA incident deficit with source/target geometry, ridge incidence,
finite-span overlap, and normalized query height.  The separated core is
withheld using the frozen J001--J015 slope/reattachment relation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(os.environ.get("URSA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ACTIVE = ROOT / "experiments/harm_w_active_worktree"
PANEL = ACTIVE / "results/ursa_fuxi_explicit_two_ridge_large_v1/input_planner_v2"
ARCHIVE = Path(os.environ.get("FUXI_CFD_DATA_ROOT", ROOT / "third_party_data" / "fuxi-cfd"))
DEFAULT_OUTPUT = ROOT / "results/ursa_fuxi_double_ridge_physical_v1/development.json"
for path in (ROOT / "experiments/ursa_shelter_v10/scripts", ACTIVE / "scripts"):
    sys.path.insert(0, str(path))

from audit_ursa_v10_input_variables_v1 import _fine_corrected_wind  # noqa: E402
from ursa_v10_variables_v2 import extract_target_and_sources, full_map_transect  # noqa: E402


THRESHOLDS = (0.2, 0.5, 1.0)
SPLITS = {"train", "development", "confirmation"}


def spectral_pressure_history(
    elevation_m: np.ndarray,
    t_m: np.ndarray,
    crest_t_m: float,
    lee_base_t_m: float,
) -> float:
    """Hunt linear pressure proxy integrated over adverse lee-side history."""
    elevation = np.asarray(elevation_m, dtype=np.float64)
    spacing = float(np.median(np.diff(t_m)))
    coordinate = spacing * np.arange(elevation.size, dtype=np.float64)
    design = np.column_stack((np.ones(elevation.size), coordinate))
    coefficient, *_ = np.linalg.lstsq(design, elevation, rcond=None)
    detrended = elevation - design @ coefficient
    pad = elevation.size - 1
    padded = np.pad(detrended, (pad, pad), mode="reflect")
    phase = np.arange(1, pad + 1, dtype=np.float64) / (pad + 1.0)
    ramp = 0.5 * (1.0 - np.cos(math.pi * phase))
    window = np.ones_like(padded)
    window[:pad] = ramp
    window[-pad:] = ramp[::-1]
    wave_number = 2.0 * math.pi * np.fft.fftfreq(padded.size, d=spacing)
    speed_ratio = np.fft.ifft(np.abs(wave_number) * np.fft.fft(padded * window)).real
    pressure = -2.0 * speed_ratio[pad : pad + elevation.size]
    first = int(np.argmin(np.abs(t_m - crest_t_m)))
    last = int(np.argmin(np.abs(t_m - lee_base_t_m)))
    if last <= first:
        return math.nan
    gradient = np.gradient(pressure, t_m)
    segment_t = t_m[first : last + 1]
    weight = (segment_t - segment_t[0]) / max(segment_t[-1] - segment_t[0], 1.0e-9)
    return float(np.trapz(weight * np.maximum(gradient[first : last + 1], 0.0), segment_t))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def separation_limits(lee_slope: float) -> tuple[float, float] | None:
    """J001--J015 max-slope interpolation; no separation below 0.4."""
    if lee_slope < 0.4:
        return None
    knots = np.asarray([0.4, 0.5, 0.6])
    x_sep = np.asarray([1.4842929452712592, 0.8132589012468169, 0.5211376473933503])
    x_re = np.asarray([4.206618837271164, 4.399335997676735, 4.593917960582103])
    return (
        float(np.interp(lee_slope, knots, x_sep)),
        float(np.interp(lee_slope, knots, x_re)),
    )


def select_pair_geometry(record: dict, dem, roughness, east, north) -> dict:
    tx, ty = map(float, record["pair"]["target_xy_m"])
    sx, sy = map(float, record["pair"]["source_xy_m"])
    inventory = extract_target_and_sources(
        dem, roughness, east, north, int(round(ty / 30.0)), int(round(tx / 30.0))
    )
    if inventory["status"] != "valid" or not inventory["sources"]:
        raise ValueError(f"no V10 geometry for {record['case_id']}")
    source = min(
        inventory["sources"],
        key=lambda row: (
            (30.0 * float(row["source_crest_column"]) - sx) ** 2
            + (30.0 * float(row["source_crest_row"]) - sy) ** 2
        ),
    )
    target = inventory["target"]
    source_match_distance = math.hypot(
        30.0 * float(source["source_crest_column"]) - sx,
        30.0 * float(source["source_crest_row"]) - sy,
    )
    return {
        "source": source,
        "sources": inventory["sources"],
        "target": target,
        "source_match_distance_m": source_match_distance,
    }


def load_case(
    path: Path,
    *,
    global_wind_only: bool = False,
    reference_free: bool = False,
) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    case_id = record["case_id"]
    candidate_path = PANEL / "candidates" / f"{case_id}.npz"
    input_path = ARCHIVE / "cases" / case_id / "inputs.npz"
    output_path = ARCHIVE / "cases" / case_id / "outputs.npz"
    with np.load(candidate_path, allow_pickle=False) as z:
        heights = np.asarray(z["height_agl_m"], dtype=np.float64)
        flat = np.asarray(z["flat_indices"], dtype=np.int64)
        raw = np.asarray(z["evve_raw_hc_mps"], dtype=np.float64)
        old = np.asarray(z["ursa_s_only_c080_hc_mps"], dtype=np.float64)
        old_retention = np.asarray(z["physical_far_ratio_c080_hc"], dtype=np.float64)
    with np.load(input_path, allow_pickle=False) as z:
        dem = np.asarray(z["dem"], dtype=np.float64)
        roughness = np.asarray(z["roughness"], dtype=np.float64)
        east, north = _fine_corrected_wind(z["u_100m"], z["v_100m"])
    if global_wind_only:
        global_east, global_north = map(float, record["input_horizontal_uv_mps"])
        east = np.full_like(east, global_east)
        north = np.full_like(north, global_north)
    if reference_free:
        truth = np.full_like(raw, np.nan, dtype=np.float64)
        reference_east = np.zeros_like(raw, dtype=np.float64)
        reference_north = np.zeros_like(raw, dtype=np.float64)
    else:
        with np.load(output_path, allow_pickle=False) as z:
            truth = np.asarray(z["w"], dtype=np.float64).reshape(27, -1)[:, flat]
            reference_east = np.asarray(z["v"], dtype=np.float64).reshape(27, -1)[:, flat]
            reference_north = np.asarray(z["u"], dtype=np.float64).reshape(27, -1)[:, flat]
    if raw.shape != truth.shape or old.shape != truth.shape:
        raise ValueError(f"shape mismatch: {case_id}")
    geometry = select_pair_geometry(record, dem, roughness, east, north)
    source, target = geometry["source"], geometry["target"]
    hu = float(source["source_leeward_height_m"])
    ht = float(target["target_windward_height_m"])
    width = float(source["source_component_crosswind_span_m"])
    incidence = float(np.clip(
        abs(float(source["source_wind_normal_to_ridge_mps"])) / max(
            float(source["source_wind_speed_100m"]), 1.0e-9
        ), 0.0, 1.0
    ))
    lateral = abs(float(source["source_crosswind_offset_m"]))
    lee = float(source["source_leeward_max_native_slope_tangent"])
    windward = float(source["source_windward_mean_slope_tangent"])
    target_windward = float(target["target_windward_mean_slope_tangent"])

    rows, cols = np.unravel_index(flat, dem.shape)
    flow = np.asarray(record["pair"]["flow_unit_xy"], dtype=np.float64)
    cross = np.asarray(record["pair"]["cross_unit_xy"], dtype=np.float64)
    # Anchor the wake coordinate to the extracted physical crest, not to the
    # route-pair proxy point used only for choosing the matching ridge.
    sx = 30.0 * float(source["source_crest_column"])
    sy = 30.0 * float(source["source_crest_row"])
    x_m = (cols * 30.0 - sx) * flow[0] + (rows * 30.0 - sy) * flow[1]
    n_m = (cols * 30.0 - sx) * cross[0] + (rows * 30.0 - sy) * cross[1]
    reference_aligned = reference_east * flow[0] + reference_north * flow[1]
    target_anchor_x, target_anchor_y = map(float, record["pair"]["target_xy_m"])
    pressure_t, pressure_z = full_map_transect(
        dem,
        int(round(target_anchor_y / 30.0)),
        int(round(target_anchor_x / 30.0)),
        float(flow[0]),
        float(flow[1]),
        spacing_m=30.0,
    )
    e_components = []
    for item in geometry["sources"]:
        component_sx = 30.0 * float(item["source_crest_column"])
        component_sy = 30.0 * float(item["source_crest_row"])
        component_x = (
            (cols * 30.0 - component_sx) * flow[0]
            + (rows * 30.0 - component_sy) * flow[1]
        )
        component_n = (
            (cols * 30.0 - component_sx) * cross[0]
            + (rows * 30.0 - component_sy) * cross[1]
        )
        e_components.append({
            "role": "upstream_source",
            "crest_along_flow_m": component_sx * flow[0] + component_sy * flow[1],
            "downwind_from_crest_m": component_x,
            "crosswind_from_component_m": component_n,
            "component_crosswind_span_m": float(item["source_component_crosswind_span_m"]),
            "leeward_max_slope_tangent": float(item["source_leeward_max_native_slope_tangent"]),
            "leeward_mean_slope_tangent": float(item["source_leeward_mean_slope_tangent"]),
            "incidence_ratio": float(np.clip(
                abs(float(item["source_wind_normal_to_ridge_mps"])) / max(
                    float(item["source_wind_speed_100m"]), 1.0e-9
                ), 0.0, 1.0
            )),
            "z0_over_height": float(item["source_z0_m"]) / max(
                float(item["source_leeward_height_m"]), 1.0e-9
            ),
            "leeward_valley_length_m": float(item["source_leeward_valley_length_m"]),
            "leeward_base_elevation_m": float(item["source_leeward_base_elevation_m"]),
            "leeward_height_m": float(item["source_leeward_height_m"]),
            "profile_rms_slope": math.sqrt(max(
                float(item["source_profile_mean_square_slope_native"]), 0.0
            )),
            "profile_curvature_H": abs(
                float(item["source_profile_curvature_native_per_m"])
            ) * float(item["source_leeward_height_m"]),
            "pressure_history": spectral_pressure_history(
                pressure_z,
                pressure_t,
                float(item["source_crest_t_from_anchor_m"]),
                float(item["source_lee_base_t_from_anchor_m"]),
            ),
        })
    target_sx = 30.0 * float(target["target_crest_column"])
    target_sy = 30.0 * float(target["target_crest_row"])
    e_components.append({
        "role": "target_self_wake",
        "crest_along_flow_m": target_sx * flow[0] + target_sy * flow[1],
        "downwind_from_crest_m": (
            (cols * 30.0 - target_sx) * flow[0]
            + (rows * 30.0 - target_sy) * flow[1]
        ),
        "crosswind_from_component_m": (
            (cols * 30.0 - target_sx) * cross[0]
            + (rows * 30.0 - target_sy) * cross[1]
        ),
        "component_crosswind_span_m": float(target["target_component_crosswind_span_m"]),
        "leeward_max_slope_tangent": float(target["target_leeward_max_native_slope_tangent"]),
        "leeward_mean_slope_tangent": float(target["target_leeward_mean_slope_tangent"]),
        "incidence_ratio": incidence,
        "z0_over_height": float(target["target_z0_m"]) / max(
            float(target["target_leeward_height_m"]), 1.0e-9
        ),
        "leeward_valley_length_m": float(target["target_leeward_valley_length_m"]),
        "leeward_base_elevation_m": float(target["target_leeward_base_elevation_m"]),
        "leeward_height_m": float(target["target_leeward_height_m"]),
        "profile_rms_slope": math.sqrt(max(
            float(target["target_profile_mean_square_slope_native"]), 0.0
        )),
        "profile_curvature_H": abs(
            float(target["target_profile_curvature_native_per_m"])
        ) * float(target["target_leeward_height_m"]),
        "pressure_history": spectral_pressure_history(
            pressure_z,
            pressure_t,
            (target_sx - target_anchor_x) * flow[0]
            + (target_sy - target_anchor_y) * flow[1],
            (target_sx - target_anchor_x) * flow[0]
            + (target_sy - target_anchor_y) * flow[1]
            + float(target["target_leeward_valley_length_m"]),
        ),
    })
    source_base = float(source["source_leeward_base_elevation_m"])
    ground = dem.ravel()[flat]
    mask = np.zeros_like(truth, dtype=bool)
    limits = separation_limits(lee)
    if limits is not None and hu > 0.0:
        x = x_m / hu
        x_sep, x_re = limits
        centre = np.maximum(0.0, 1.0 - (x - x_sep) / (x_re - x_sep))
        half = 0.1 * (x - x_sep)
        for hi, agl in enumerate(heights):
            z = (ground - source_base + agl) / hu
            mask[hi] = (
                (x >= x_sep)
                & (x <= x_re)
                & (z >= np.maximum(0.0, centre - half))
                & (z <= centre + half)
            )
    features = {
        "hu": hu,
        "ht": ht,
        "height_ratio": hu / max(ht, 1.0e-9),
        "distance_ratio": float(source["crest_gap_m"]) / max(hu, 1.0e-9),
        "source_windward_slope": windward,
        "source_leeward_slope": lee,
        "target_windward_slope": target_windward,
        "incidence": incidence,
        "width": width,
        "lateral_ratio": lateral / max(0.5 * width, 1.0e-9),
        "z0_over_hu": float(source["source_z0_m"]) / max(hu, 1.0e-9),
        "source_match_distance_m": geometry["source_match_distance_m"],
        "source_profile_rms_slope": math.sqrt(max(
            float(source["source_profile_mean_square_slope_native"]), 0.0
        )),
        "source_profile_curvature_H": abs(
            float(source["source_profile_curvature_native_per_m"])
        ) * hu,
        "source_wind_speed_100m": float(source["source_wind_speed_100m"]),
        "source_wind_normal_to_ridge_mps": float(
            source["source_wind_normal_to_ridge_mps"]
        ),
        "source_wind_tangent_to_ridge_mps": float(
            source["source_wind_tangent_to_ridge_mps"]
        ),
        "target_wind_speed_100m": float(target["target_crest_wind_speed_100m"]),
        "target_wind_normal_to_ridge_mps": float(
            target["target_wind_normal_to_ridge_mps"]
        ),
        "target_wind_tangent_to_ridge_mps": float(
            target["target_wind_tangent_to_ridge_mps"]
        ),
    }
    formula_required = (
        "hu",
        "ht",
        "height_ratio",
        "source_leeward_slope",
        "target_windward_slope",
    )
    nonfinite = sorted(
        key for key in formula_required if not math.isfinite(features[key])
    )
    if nonfinite:
        raise ValueError(f"non-finite physical geometry fields for {case_id}: {nonfinite}")
    return {
        "case_id": case_id,
        "input_contract": (
            "DEM+single_wind_vector" if global_wind_only
            else "DEM+spatial_100m_wind_field"
        ),
        "reference_free": reference_free,
        "split": record["split"],
        "group": record["terrain_group_sha256"],
        "heights": heights,
        "truth": truth,
        "raw": raw,
        "old": old,
        "old_retention": old_retention,
        "mask": mask,
        "features": features,
        "e_inputs": {
            "downwind_from_crest_m": x_m,
            "crosswind_from_component_m": n_m,
            "component_crosswind_span_m": width,
            "leeward_max_slope_tangent": lee,
            "leeward_valley_length_m": float(source["source_leeward_valley_length_m"]),
            "reference_aligned_mps": reference_aligned,
            "query_ground_elevation_m": ground,
            "components": e_components,
        },
    }


def candidate_prediction(case: dict, params: tuple[float, float, float, float]) -> np.ndarray:
    scale, slope_power, ratio_power, target_restore = params
    f = case["features"]
    # The frozen old retention already carries distance, incidence, lateral
    # overlap and query-height physics.  Do not count those effects twice.
    lee_state = np.clip(f["source_leeward_slope"] / 0.4, 0.5, 1.5) ** slope_power
    ratio_state = np.clip(f["height_ratio"], 0.5, 2.0) ** ratio_power
    # A stronger second-hill windward face restores its own ridge lift; this
    # prevents an upstream shelter term from erasing the second ridge globally.
    second_hill_recovery = math.exp(
        -target_restore * np.clip(f["target_windward_slope"] / 0.4, 0.0, 1.5)
    )
    deficit = (
        (1.0 - case["old_retention"])
        * scale
        * lee_state
        * ratio_state
        * second_hill_recovery
    )
    retention = np.clip(1.0 - deficit, 0.0, 1.0)
    return np.minimum(case["raw"], 0.0) + retention * np.maximum(case["raw"], 0.0)


def case_metrics(prediction: np.ndarray, case: dict) -> dict:
    valid = ~case["mask"] & np.isfinite(prediction) & np.isfinite(case["truth"])
    pred = prediction[valid]
    truth = case["truth"][valid]
    row = {
        "mae": float(np.mean(np.abs(pred - truth))),
        "positive_overprediction_mae": float(np.mean(np.maximum(pred - truth, 0.0))),
        "masked_fraction": float(np.mean(case["mask"])),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        predicted = pred >= threshold
        actual = truth >= threshold
        tp = int(np.count_nonzero(predicted & actual))
        fp = int(np.count_nonzero(predicted & ~actual))
        fn = int(np.count_nonzero(~predicted & actual))
        tn = int(np.count_nonzero(~predicted & ~actual))
        row["thresholds"][str(threshold)] = {
            "false_lift_joint_probability": fp / pred.size,
            "lift_classification_accuracy": (tp + tn) / pred.size,
            "recall": tp / (tp + fn) if tp + fn else None,
        }
    return row


def aggregate(cases: list[dict], variant: str, params=None) -> dict:
    rows = []
    for case in cases:
        if variant == "raw":
            prediction = case["raw"]
        elif variant == "old_ursa":
            prediction = case["old"]
        else:
            prediction = candidate_prediction(case, params)
        rows.append(case_metrics(prediction, case))
    result = {
        "case_count": len(rows),
        "mae": float(np.mean([r["mae"] for r in rows])),
        "positive_overprediction_mae": float(
            np.mean([r["positive_overprediction_mae"] for r in rows])
        ),
        "masked_fraction": float(np.mean([r["masked_fraction"] for r in rows])),
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        key = str(threshold)
        result["thresholds"][key] = {}
        for metric in (
            "false_lift_joint_probability",
            "lift_classification_accuracy",
            "recall",
        ):
            values = [r["thresholds"][key][metric] for r in rows]
            values = [v for v in values if v is not None]
            result["thresholds"][key][metric] = float(np.mean(values)) if values else None
    return result


def dominates(candidate: dict, old: dict) -> bool:
    return (
        candidate["mae"] < old["mae"]
        and candidate["positive_overprediction_mae"] < old["positive_overprediction_mae"]
        and all(
            candidate["thresholds"][str(t)]["false_lift_joint_probability"]
            <= old["thresholds"][str(t)]["false_lift_joint_probability"]
            for t in THRESHOLDS
        )
        and all(
            candidate["thresholds"][str(t)]["lift_classification_accuracy"]
            >= old["thresholds"][str(t)]["lift_classification_accuracy"]
            for t in THRESHOLDS
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--global-wind-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    wanted = {"train", "development"} if args.phase == "development" else {"confirmation"}
    paths = []
    for path in sorted((PANEL / "candidates").glob("case_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["split"] in wanted:
            paths.append(path)
    if args.max_cases is not None:
        paths = paths[: args.max_cases]
    cases = []
    extraction_exclusions = []
    for index, path in enumerate(paths, 1):
        try:
            cases.append(load_case(path, global_wind_only=args.global_wind_only))
        except ValueError as exc:
            # Preserve panel membership and the exact deterministic reason.
            # A newer geometry extractor is not allowed to redefine the
            # already-frozen FuXi two-ridge panel by silently dropping rows.
            extraction_exclusions.append(
                {"case_id": path.stem, "reason": str(exc), "receipt_path": str(path)}
            )
        elapsed = time.monotonic() - started
        rate = index / elapsed
        print(
            json.dumps(
                {
                    "stage": "load",
                    "completed": index,
                    "total": len(paths),
                    "percent": 100.0 * index / len(paths),
                    "elapsed_s": elapsed,
                    "eta_s": (len(paths) - index) / rate,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.phase == "development":
        train_cases = [case for case in cases if case["split"] == "train"]
        development_cases = [case for case in cases if case["split"] == "development"]
        if not train_cases or not development_cases:
            raise ValueError("development phase requires non-empty train and development splits")
        train_raw = aggregate(train_cases, "raw")
        train_old = aggregate(train_cases, "old_ursa")
        development_raw = aggregate(development_cases, "raw")
        development_old = aggregate(development_cases, "old_ursa")
        grid = [
            (scale, slope_power, ratio_power, target_restore)
            for scale in (0.75, 1.0, 1.25, 1.5)
            for slope_power in (0.0, 0.25, 0.5, 1.0)
            for ratio_power in (0.0, 0.25, 0.5)
            for target_restore in (0.0, 0.25, 0.5, 1.0)
        ]
        scored = []
        for index, params in enumerate(grid, 1):
            # Freeze constants only from the predeclared development split.
            metrics = aggregate(development_cases, "candidate", params)
            scored.append({"parameters": params, "metrics": metrics})
            if index % 12 == 0:
                print(json.dumps({"stage": "formula_grid", "completed": index, "total": len(grid)}), flush=True)
        admissible = [row for row in scored if dominates(row["metrics"], development_old)]
        pool = admissible if admissible else scored
        selected = min(pool, key=lambda row: row["metrics"]["mae"])
        result = {
            "schema": "ursa.fuxi-double-ridge-physical-development.v1",
            "status": "complete",
            "phase": args.phase,
            "selection_rule": "require lower MAE and positive-overprediction, no worse false-lift probability, and no worse lift-classification accuracy in all three bins; then minimum MAE",
            "candidate_count": len(grid),
            "componentwise_improving_candidate_count": len(admissible),
            "selected_parameters": {
                "deficit_scale": selected["parameters"][0],
                "source_leeward_slope_power": selected["parameters"][1],
                "source_to_target_height_ratio_power": selected["parameters"][2],
                "second_hill_windward_lift_recovery": selected["parameters"][3],
            },
            "train": {
                "raw": train_raw,
                "old_ursa": train_old,
                "selected": aggregate(train_cases, "candidate", selected["parameters"]),
            },
            "development": {
                "raw": development_raw,
                "old_ursa": development_old,
                "selected": selected["metrics"],
            },
        }
    else:
        if args.freeze is None:
            raise ValueError("--freeze is required for confirmation")
        frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
        p = frozen["selected_parameters"]
        params = (
            p["deficit_scale"],
            p["source_leeward_slope_power"],
            p["source_to_target_height_ratio_power"],
            p["second_hill_windward_lift_recovery"],
        )
        raw = aggregate(cases, "raw")
        old = aggregate(cases, "old_ursa")
        result = {
            "schema": "ursa.fuxi-double-ridge-physical-confirmation.v1",
            "status": "complete",
            "phase": args.phase,
            "freeze_path": str(args.freeze),
            "freeze_sha256": sha256(args.freeze),
            "selected_parameters": p,
            "raw": raw,
            "old_ursa": old,
            "selected": aggregate(cases, "candidate", params),
        }
    result["runtime_s"] = time.monotonic() - started
    result["openfoam_runs"] = 0
    result["machine_learning_model"] = False
    result["input_contract"] = (
        "DEM+single_wind_vector" if args.global_wind_only
        else "legacy_DEM+spatial_100m_wind_field"
    )
    result["case_ids"] = [case["case_id"] for case in cases]
    result["requested_case_count"] = len(paths)
    result["extraction_exclusion_count"] = len(extraction_exclusions)
    result["extraction_exclusions"] = extraction_exclusions
    result["script_sha256"] = sha256(Path(__file__))
    # The result file itself is protected above; an existing parent directory
    # (including /tmp for smoke tests) is not an overwrite.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "runtime_s": result["runtime_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
