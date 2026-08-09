#!/usr/bin/env python3
"""Validate frozen URSA S+E V2 on saved EVVE/BO04/WindNinja 45-case fields."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("URSA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ACTIVE = ROOT / "experiments/harm_w_active_worktree"
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "experiments/ursa_shelter_v10/scripts"), str(ACTIVE / "scripts")]

import prepare_ursa_fuxi_axis_corrected_confirmation328_candidates_v3 as s_base
import prepare_evve_v7_confirmation_inputs as evve_prepare
import run_ursa_bo04_windninja_staged_confirmation_v1 as legacy
import run_ursa_v10_fast_engineering_fuxi_reuse_v1 as old_runner
import evaluate_ursa_fuxi_core_mix_engineering_v1 as e_v1
import evaluate_ursa_global_wind_pressure_post_increment_v1 as v2
from fit_ursa_fuxi_double_ridge_physical_v1 import spectral_pressure_history
from ursa_v10_variables_v2 import extract_target_and_sources, full_map_transect

LEGACY_ROOT = ACTIVE / "results/ursa_multi_estimator_transfer_v2/bo04_windninja_staged_confirmation_v1"
MODELS = ("evve", "bo04", "windninja")
THRESHOLDS = (0.2, 0.5, 1.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def component(item: dict, role: str, rows: np.ndarray, cols: np.ndarray, flow: np.ndarray,
              cross: np.ndarray, pressure_t: np.ndarray, pressure_z: np.ndarray) -> dict:
    sx = 30.0 * float(item[f"{'target' if role == 'target_self_wake' else 'source'}_crest_column"])
    sy = 30.0 * float(item[f"{'target' if role == 'target_self_wake' else 'source'}_crest_row"])
    prefix = "target" if role == "target_self_wake" else "source"
    height = float(item[f"{prefix}_leeward_height_m"])
    crest_t = float(item.get(f"{prefix}_crest_t_from_anchor_m", 0.0))
    lee_t = float(item.get(f"{prefix}_lee_base_t_from_anchor_m", crest_t + item[f"{prefix}_leeward_valley_length_m"]))
    return {
        "role": role,
        "crest_along_flow_m": sx * flow[0] + sy * flow[1],
        "downwind_from_crest_m": (cols * 30.0 - sx) * flow[0] + (rows * 30.0 - sy) * flow[1],
        "crosswind_from_component_m": (cols * 30.0 - sx) * cross[0] + (rows * 30.0 - sy) * cross[1],
        "component_crosswind_span_m": float(item[f"{prefix}_component_crosswind_span_m"]),
        "leeward_max_slope_tangent": float(item[f"{prefix}_leeward_max_native_slope_tangent"]),
        "leeward_mean_slope_tangent": float(item[f"{prefix}_leeward_mean_slope_tangent"]),
        "incidence_ratio": float(np.clip(abs(float(item[f"{prefix}_wind_normal_to_ridge_mps"])) / max(float(item.get(f"{prefix}_wind_speed_100m", item.get("target_crest_wind_speed_100m", 0.0))), 1e-9), 0, 1)),
        "z0_over_height": float(item[f"{prefix}_z0_m"]) / max(height, 1e-9),
        "leeward_valley_length_m": float(item[f"{prefix}_leeward_valley_length_m"]),
        "leeward_base_elevation_m": float(item[f"{prefix}_leeward_base_elevation_m"]),
        "leeward_height_m": height,
        "pressure_history": spectral_pressure_history(pressure_z, pressure_t, crest_t, lee_t),
    }


def frozen_fields(spec, candidate: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    inputs = legacy.incident_base._load_inputs(spec)
    dem = np.asarray(inputs.dem_yx_m, dtype=np.float64)
    roughness = np.asarray(inputs.roughness_yx_m, dtype=np.float64)
    east_spatial, north_spatial = old_runner._fine_corrected_wind(inputs.u100_9x9_mps, inputs.v100_9x9_mps)
    centre = (dem.shape[0] // 2, dem.shape[1] // 2)
    east0, north0 = float(east_spatial[centre]), float(north_spatial[centre])
    speed = math.hypot(east0, north0)
    if speed <= 0:
        raise ValueError(f"zero global wind: {spec.case_id}")
    east = np.full_like(dem, east0)
    north = np.full_like(dem, north0)
    flow = np.asarray([east0 / speed, north0 / speed])
    cross = np.asarray([-flow[1], flow[0]])
    heights = np.asarray(candidate["height_agl_m"], dtype=np.float64)
    flat = np.asarray(candidate["flat_indices"], dtype=np.int64)
    offsets = np.asarray(candidate["cell_offsets"], dtype=np.int64)
    with np.load(LEGACY_ROOT / "ursa_input_geometry/v1_compact/preflight_cases" / f"{spec.case_id}.npz", allow_pickle=False) as z:
        anchor_rows = np.asarray(z["anchor_row"], dtype=np.int64)
        anchor_cols = np.asarray(z["anchor_column"], dtype=np.int64)
    old_ratio, old_record = s_base.physical_far_ratio(inputs, flat, query_chunk_size=131072)
    retention = np.ones_like(old_ratio)
    exposure = np.zeros_like(old_ratio)
    counts = {"anchors": len(anchor_rows), "valid_anchors": 0, "no_source": 0}
    for ai, (ar, ac) in enumerate(zip(anchor_rows, anchor_cols, strict=True)):
        begin, end = int(offsets[ai]), int(offsets[ai + 1])
        if end <= begin:
            continue
        inventory = extract_target_and_sources(dem, roughness, east, north, int(ar), int(ac))
        if inventory["status"] != "valid" or not inventory["sources"]:
            counts["no_source"] += 1
            continue
        counts["valid_anchors"] += 1
        target = inventory["target"]
        source = inventory["sources"][-1]
        hu = float(source["source_leeward_height_m"])
        ht = float(target["target_windward_height_m"])
        factor = 1.30 * np.clip(hu / max(ht, 1e-9), 0.5, 2.0) ** 0.25
        retention[:, begin:end] = np.clip(1.0 - (1.0 - old_ratio[:, begin:end]) * factor, 0.0, 1.0)
        local_flat = flat[begin:end]
        rows, cols = np.unravel_index(local_flat, dem.shape)
        pressure_t, pressure_z = full_map_transect(dem, int(ar), int(ac), flow[0], flow[1], spacing_m=30.0)
        components = [component(s, "upstream_source", rows, cols, flow, cross, pressure_t, pressure_z) for s in inventory["sources"]]
        components.append(component(target, "target_self_wake", rows, cols, flow, cross, pressure_t, pressure_z))
        local_case = {
            "heights": heights,
            "e_inputs": {
                "reference_aligned_mps": np.zeros((len(heights), end - begin)),
                "query_ground_elevation_m": dem.ravel()[local_flat],
                "components": components,
            },
        }
        exposure[:, begin:end] = v2.field(local_case, 0.55)
    return retention, exposure, {**counts, "global_wind_east_north_mps": [east0, north0], "old_ratio": old_record}


def metrics(pred: np.ndarray, truth: np.ndarray, weights: np.ndarray, valid: np.ndarray) -> dict:
    w = np.where(valid, weights, 0.0)
    support = float(w.sum())
    w /= support
    out = {"mae": float(np.sum(w * abs(pred - truth))), "positive_overprediction_mae": float(np.sum(w * np.maximum(pred - truth, 0))), "thresholds": {}}
    for t in THRESHOLDS:
        pp, aa = pred >= t, truth >= t
        out["thresholds"][f"{t:.1f}"] = {"false_lift": float(np.sum(w * (pp & ~aa))), "accuracy": float(np.sum(w * (pp == aa))), "recall": float(np.sum(w * (pp & aa)) / max(np.sum(w * aa), 1e-15))}
    return out


def mean(rows: list[dict]) -> dict:
    out = {"count": len(rows), "mae": float(np.mean([x["mae"] for x in rows])), "positive_overprediction_mae": float(np.mean([x["positive_overprediction_mae"] for x in rows])), "thresholds": {}}
    for t in THRESHOLDS:
        k = f"{t:.1f}"; out["thresholds"][k] = {m: float(np.mean([x["thresholds"][k][m] for x in rows])) for m in ("false_lift", "accuracy", "recall")}
    return out


def score_one(payload) -> dict:
    spec, group = payload
    cand=legacy.load_candidate(LEGACY_ROOT/"candidates"/f"{spec.case_id}.npz")
    with np.load(LEGACY_ROOT/"reference_cases"/f"{spec.case_id}.npz",allow_pickle=False) as z: truth=np.asarray(z["reference_w_hc_mps"],float)
    rec=json.loads((LEGACY_ROOT/"ursa_input_geometry/v1_compact/preflight_cases"/f"{spec.case_id}.json").read_text())
    _,ev=evve_prepare.active_carrier_components(spec,v1_record=rec)
    bases={"evve":np.asarray(ev["carrier_hc_mps"],float),"bo04":np.asarray(cand["bo04_raw_hc_mps"],float),"windninja":np.asarray(cand["windninja_raw_hc_mps"],float)}
    retention,exposure,audit=frozen_fields(spec,cand); valid=exposure<0.05
    weights=legacy.hierarchy_weights(np.asarray(cand["anchor_strong"],bool),np.asarray(cand["cell_offsets"],int),truth.shape)
    model={}
    for name,base in bases.items():
        corrected=np.minimum(base,0)+retention*np.maximum(base,0)
        model[name]={"raw":metrics(base,truth,weights,valid),"v2":metrics(corrected,truth,weights,valid)}
    return {"case_id":spec.case_id,"group":group,"masked_fraction":float(np.mean(~valid)),"audit":audit,"models":model}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--max-cases", type=int); ap.add_argument("--workers", type=int, default=1); a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    legacy.bind_input_contract_compatibility(); panel, specs = legacy.checked_panel(); specs = specs[:a.max_cases] if a.max_cases else specs
    groups = {c["case_id"]: g["terrain_group_sha256"] for g in panel["selection"]["selected_groups"] for c in g["cases"]}
    rows=[]; started=time.monotonic(); payloads=[(s,groups[s.case_id]) for s in specs]
    if a.workers == 1:
        iterator = ((score_one(p), None) for p in payloads)
        for i,(row,_) in enumerate(iterator,1):
            rows.append(row); elapsed=time.monotonic()-started; print(json.dumps({"completed":i,"total":len(specs),"percent":100*i/len(specs),"elapsed_s":elapsed,"eta_s":elapsed/i*(len(specs)-i),"case_id":row["case_id"]}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={pool.submit(score_one,p):p[0].case_id for p in payloads}
            for i,future in enumerate(as_completed(futures),1):
                row=future.result(); rows.append(row); elapsed=time.monotonic()-started; print(json.dumps({"completed":i,"total":len(specs),"percent":100*i/len(specs),"elapsed_s":elapsed,"eta_s":elapsed/i*(len(specs)-i),"case_id":row["case_id"]}),flush=True)
    rows.sort(key=lambda x:x["case_id"])
    by_group={}; [by_group.setdefault(r["group"],[]).append(r) for r in rows]
    aggregate={}
    for name in MODELS:
        aggregate[name]={}
        for variant in ("raw","v2"):
            aggregate[name][variant]=mean([mean([r["models"][name][variant] for r in rs]) for rs in by_group.values()])
    out={"schema":"ursa.v2-three-base-45case-validation.v1","status":"complete" if len(specs)==45 else "dry_run","model":"frozen S+E V2","input_contract":"DEM + single global wind vector","case_count":len(specs),"terrain_group_count":len(by_group),"workers":a.workers,"aggregate":aggregate,"case_rows":rows,"bindings":{"script_sha256":sha256(Path(__file__)),"legacy_result_sha256":sha256(LEGACY_ROOT/"result_v1.json")},"runtime_s":time.monotonic()-started}
    a.output.parent.mkdir(parents=True,exist_ok=False); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"stage":"complete","output":str(a.output)})); return 0


if __name__ == "__main__": raise SystemExit(main())
