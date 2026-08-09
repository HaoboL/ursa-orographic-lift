#!/usr/bin/env python3
"""Evaluate literature-based URSA E_core/E_mix and task-relevant S metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(os.environ.get("URSA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/ursa_shelter_v10/scripts"))

from fit_ursa_fuxi_double_ridge_physical_v1 import (  # noqa: E402
    PANEL, candidate_prediction, load_case,
)
from fit_ursa_fuxi_double_ridge_e_physical_v1 import predict_e as predict_core  # noqa: E402
from ursa_v10_dual_head_model_d4_v1 import cosine_squared_separation_elevation_m  # noqa: E402
from ursa_v10_dual_head_model_v1 import (  # noqa: E402
    LIU_REATTACHMENT_X_OVER_L_3D, LIU_SEPARATION_X_OVER_L_3D, LIU_SLOPE_ANCHORS,
)
from ursa_v10_ml1_mixing_layer_v1 import separation_activation, post_reattachment_decay  # noqa: E402
from ursa_v10_ml1_relaxing_wake_v1 import relaxing_wake_depth_m, relaxing_wake_eta  # noqa: E402

S_PARAMS = (1.25, 0.0, 0.25, 0.0)
CORE_PARAMS = (1.0, 0.10, 0.0)
THRESHOLDS = (0.05, 0.10, 0.25, 0.50, 0.75)
FALSE_LIFT_BINS = (0.2, 0.5, 1.0)


def load_cases(
    max_cases: int | None,
    wanted: set[str],
    *,
    global_wind_only: bool = False,
) -> list[dict]:
    paths = []
    for path in sorted((PANEL / "candidates").glob("case_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["split"] in wanted:
            paths.append(path)
    if max_cases is not None:
        paths = paths[:max_cases]
    cases = []
    started = time.monotonic()
    for i, path in enumerate(paths, 1):
        try:
            cases.append(load_case(path, global_wind_only=global_wind_only))
        except ValueError:
            pass
        elapsed = time.monotonic() - started
        print(json.dumps({"stage": "load", "completed": i, "total": len(paths),
            "percent": 100*i/len(paths), "elapsed_s": elapsed,
            "eta_s": elapsed/i*(len(paths)-i)}), flush=True)
    return cases


def predict_mix(case: dict) -> tuple[np.ndarray, np.ndarray]:
    e = case["e_inputs"]
    ground = np.asarray(e["query_ground_elevation_m"], dtype=np.float64)
    out_direct = np.zeros_like(np.asarray(e["reference_aligned_mps"], dtype=np.float64))
    out_post = np.zeros_like(out_direct)
    for component in e["components"]:
        x = np.asarray(component["downwind_from_crest_m"], dtype=np.float64)
        n = np.asarray(component["crosswind_from_component_m"], dtype=np.float64)
        slope_max = float(component["leeward_max_slope_tangent"])
        slope_mean = float(component["leeward_mean_slope_tangent"])
        length = float(component["leeward_valley_length_m"])
        height = float(component["leeward_height_m"])
        base = float(component["leeward_base_elevation_m"])
        span = float(component["component_crosswind_span_m"])
        incidence = float(component["incidence_ratio"])
        if not all(np.isfinite(v) for v in (slope_max, slope_mean, length, height, base, incidence)) or length <= 0 or height <= 0:
            continue
        x_sep = float(np.interp(slope_max, LIU_SLOPE_ANCHORS, LIU_SEPARATION_X_OVER_L_3D))*length
        x_re = float(np.interp(slope_max, LIU_SLOPE_ANCHORS, LIU_REATTACHMENT_X_OVER_L_3D))*length
        ds = np.maximum(x-x_sep, 0.0)
        thickness = 0.20*ds  # Karim: 0.1*(Us/Uc)*x and Ub=0 => Us/Uc=2.
        sep_z = cosine_squared_separation_elevation_m(
            leeward_base_elevation_m=base,
            component_height_m=height,
            leeward_length_m=length,
            separation_distance_m=x_sep,
        )
        progress = np.clip(ds/max(x_re-x_sep, 1e-9), 0.0, 1.0)
        lower = sep_z + (base-sep_z)*progress
        lateral = np.exp(-0.5*np.square(n/(0.5*span))) if np.isfinite(span) and span > 0 else np.ones_like(x)
        activation = float(separation_activation(slope_mean, incidence))
        component_direct = np.zeros_like(out_direct)
        component_post = np.zeros_like(out_post)
        direct = (x > x_sep) & (x <= x_re) & (thickness > 0)
        post = x > x_re
        _, tke_decay = post_reattachment_decay(np.maximum(x-x_re, 0.0), height)
        ratio = float(component["z0_over_height"])
        for hi, agl in enumerate(case["heights"]):
            zeta = np.divide(ground+agl-(lower+0.5*thickness), thickness,
                out=np.full_like(x, np.inf), where=thickness > 0)
            direct_shape = np.where(direct, np.clip(1.0-2.0*np.abs(zeta), 0.0, 1.0), 0.0)
            post_shape = np.zeros_like(x)
            if 0.0 < ratio < 1.0 and np.any(post):
                depth = relaxing_wake_depth_m(np.maximum(ds, 1e-9), height, ratio)
                eta = relaxing_wake_eta(np.full_like(x, agl), depth)
                post_shape = np.where(post & (eta <= 1.0), tke_decay, 0.0)
            component_direct[hi] = activation*lateral*direct_shape
            component_post[hi] = activation*lateral*post_shape
        out_direct = np.maximum(out_direct, component_direct)
        out_post = np.maximum(out_post, component_post)
    return out_direct, out_post


def summarize(cases: list[dict], s_params: tuple[float, float, float, float] = S_PARAMS) -> dict:
    rows = {threshold: [] for threshold in THRESHOLDS}
    core_rows = []
    baselines = {"S_v4_unmasked": [], "old_URSA_unmasked": []}
    ablation_rows = {name: [] for name in ("E_core", "Karim_direct", "Jackson_post", "E_mix", "E_total")}

    def task_row(prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray) -> dict:
        pred = prediction[valid]
        ref = truth[valid]
        row = {
            "total": int(valid.size), "outside": int(pred.size),
            "mae": float(np.mean(np.abs(pred-ref))),
            "positive_overprediction_mae": float(np.mean(np.maximum(pred-ref, 0.0))),
        }
        for level in FALSE_LIFT_BINS:
            predicted = pred >= level
            actual = ref >= level
            tp = np.count_nonzero(predicted & actual)
            fp = np.count_nonzero(predicted & ~actual)
            fn = np.count_nonzero(~predicted & actual)
            tn = np.count_nonzero(~predicted & ~actual)
            row[f"false_lift_{level:g}"] = float(fp/pred.size)
            row[f"accuracy_{level:g}"] = float((tp+tn)/pred.size)
            row[f"recall_{level:g}"] = float(tp/(tp+fn)) if tp+fn else None
        return row
    for case in cases:
        core = predict_core(case, CORE_PARAMS)
        direct, post = predict_mix(case)
        mix = np.maximum(direct, post)
        exposure = np.maximum(core, mix)
        aligned = np.asarray(case["e_inputs"]["reference_aligned_mps"])
        valid_e = np.isfinite(aligned)
        truth_e = aligned < 0
        event = core >= 0.5
        tp = np.count_nonzero(valid_e & event & truth_e)
        fn = np.count_nonzero(valid_e & ~event & truth_e)
        fp = np.count_nonzero(valid_e & event & ~truth_e)
        core_rows.append({"recall": tp/max(tp+fn,1), "precision": tp/max(tp+fp,1)})
        s = candidate_prediction(case, s_params)
        finite = np.isfinite(s) & np.isfinite(case["truth"])
        baselines["S_v4_unmasked"].append(task_row(s, case["truth"], finite))
        baselines["old_URSA_unmasked"].append(task_row(case["old"], case["truth"], finite))
        for name, field in {
            "E_core": core, "Karim_direct": direct, "Jackson_post": post,
            "E_mix": mix, "E_total": exposure,
        }.items():
            valid = finite & (field < 0.05)
            if np.any(valid):
                ablation_rows[name].append(task_row(s, case["truth"], valid))
        for threshold in THRESHOLDS:
            valid = np.isfinite(s) & np.isfinite(case["truth"]) & (exposure < threshold)
            if not np.any(valid):
                continue
            rows[threshold].append(task_row(s, case["truth"], valid))
    result = {"case_count": len(cases), "E_core_reverse_flow": {
        "macro_recall": float(np.mean([r["recall"] for r in core_rows])),
        "macro_precision": float(np.mean([r["precision"] for r in core_rows])),
    }, "baselines": {}, "ablation_at_0.05": {}, "thresholds": {}}
    for name, values in baselines.items():
        result["baselines"][name] = {
            "mae": float(np.mean([r["mae"] for r in values])),
            "positive_overprediction_mae": float(np.mean([r["positive_overprediction_mae"] for r in values])),
            **{metric: float(np.mean([r[metric] for r in values if r[metric] is not None]))
               for level in FALSE_LIFT_BINS for metric in
               (f"false_lift_{level:g}", f"accuracy_{level:g}", f"recall_{level:g}")},
        }
    for name, values in ablation_rows.items():
        result["ablation_at_0.05"][name] = {
            "scored_case_count": len(values),
            "masked_fraction": float(np.mean([1-r["outside"]/r["total"] for r in values])),
            "outside_mae": float(np.mean([r["mae"] for r in values])),
            **{metric: float(np.mean([r[metric] for r in values if r[metric] is not None]))
               for level in FALSE_LIFT_BINS for metric in
               (f"false_lift_{level:g}", f"accuracy_{level:g}", f"recall_{level:g}")},
        }
    for threshold, values in rows.items():
        result["thresholds"][str(threshold)] = {
            "scored_case_count": len(values),
            "masked_fraction": float(np.mean([1-r["outside"]/r["total"] for r in values])),
            "outside_mae": float(np.mean([r["mae"] for r in values])),
            "outside_positive_overprediction_mae": float(np.mean([r["positive_overprediction_mae"] for r in values])),
            **{metric: float(np.mean([r[metric] for r in values if r[metric] is not None]))
               for level in FALSE_LIFT_BINS for metric in
               (f"false_lift_{level:g}", f"accuracy_{level:g}", f"recall_{level:g}")},
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--global-wind-only", action="store_true")
    parser.add_argument("--s-params", type=float, nargs=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    wanted = {"train", "development"} if args.phase == "development" else {"confirmation"}
    cases = load_cases(
        args.max_cases, wanted, global_wind_only=args.global_wind_only
    )
    s_params = S_PARAMS if args.s_params is None else tuple(args.s_params)
    result = summarize(cases, s_params)
    result["by_split"] = {
        split: summarize(
            [case for case in cases if case["split"] == split], s_params
        )
        for split in sorted({case["split"] for case in cases})
        if any(case["split"] == split for case in cases)
    }
    result.update({
        "schema": "ursa.fuxi-core-mix-engineering-evaluation.v1",
        "machine_learning_model": False,
        "E_definition": "max(reverse-flow core, Karim direct mixing layer, Jackson post-reattachment relaxing wake)",
        "selection": "no threshold selected; report literature-membership frontier on task metrics",
        "phase": args.phase,
        "input_contract": (
            "DEM+single_wind_vector" if args.global_wind_only
            else "legacy_DEM+spatial_100m_wind_field"
        ),
        "S_parameters": list(s_params),
        "runtime_s": time.monotonic()-started,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"status":"complete","output":str(args.output),"runtime_s":result["runtime_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
