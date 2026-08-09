#!/usr/bin/env python3
"""Retrospective engineering rerun for the frozen URSA S+E V2."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(os.environ.get("URSA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ACTIVE = ROOT / "experiments/harm_w_active_worktree"
for path in (ROOT / "src", ROOT / "scripts", ROOT / "experiments/ursa_shelter_v10/scripts", ACTIVE / "scripts"):
    sys.path.insert(0, str(path))

import evaluate_ursa_fuxi_core_mix_engineering_v1 as e_v1
import evaluate_ursa_global_wind_pressure_post_increment_v1 as e_v2
import run_ursa_fuxi_explicit_two_ridge_large_input_v1 as old_input
from fit_ursa_fuxi_double_ridge_physical_v1 import candidate_prediction, load_case
from orographic_cfd.aircraft import AircraftParameters, QuasiSteadyAircraft
from orographic_cfd.lift_hazard_volume import zero_thrust_sink_rate_mps

PROTOCOL = ROOT / "URSA_V2_ROUTE_ENERGY_RERUN_PROTOCOL_20260808_ZH.md"
PANEL = ACTIVE / "results/ursa_fuxi_explicit_two_ridge_large_v1/input_planner_v2"
REFERENCE = ACTIVE / "results/ursa_fuxi_explicit_two_ridge_large_v1/reference_score_v1/reference_score.json"
FEATURES = ACTIVE / "results/ursa_fuxi_explicit_two_ridge_large_v1/high_benefit_feature_analysis_v1/raw_downstream_96_feature_table_v1.csv"
ARCHIVE = ROOT.parent / "fuxicfd-obstructed-npz/data/obstructed_npz_all_parts_v2/cases"
FREEZE = ROOT / "results/ursa_fuxi_double_ridge_physical_v1/global_wind_pressure_post_freeze_v1.json"
ARMS = ("mask_gated_se_v2", "continuous_se_v2", "s_only_v2")
BOOTSTRAP_SEED = 2026080817
BOOTSTRAP_RESAMPLES = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exposure(case: dict) -> np.ndarray:
    core = e_v1.predict_core(case, e_v1.CORE_PARAMS)
    direct, post = e_v1.predict_mix(case)
    target = next(row for row in case["e_inputs"]["components"] if row["role"] == "target_self_wake")
    pressure = np.clip(
        np.log1p(max(float(target["pressure_history"]), 0.0)) / np.log1p(e_v2.P90),
        0.0,
        1.0,
    )
    post = np.clip(post * (1.0 + 0.55 * pressure), 0.0, 1.0)
    return np.maximum(core, np.maximum(direct, post))


def run_case(path_text: str) -> dict:
    started = time.monotonic()
    path = Path(path_text)
    receipt = json.loads(path.read_text(encoding="utf-8"))
    case_id = str(receipt["case_id"])
    with np.load(PANEL / "candidates" / f"{case_id}.npz", allow_pickle=False) as z:
        flat = np.asarray(z["flat_indices"], dtype=np.int64)
        heights = np.asarray(z["height_agl_m"], dtype=np.float64)
        raw = np.asarray(z["evve_raw_hc_mps"], dtype=np.float64)
    with np.load(ARCHIVE / case_id / "inputs.npz", allow_pickle=False) as z:
        dem = np.asarray(z["dem"], dtype=np.float64)

    formula_domain = True
    ood_reason = None
    try:
        case = load_case(path, global_wind_only=True, reference_free=True)
    except ValueError as exc:
        if not str(exc).startswith("no V10 geometry for "):
            raise
        formula_domain = False
        ood_reason = str(exc)
        fields = {arm: raw.copy() for arm in ARMS}
        e_field = None
        s_retention = None
    else:
        if not case.get("reference_free"):
            raise AssertionError("input map unexpectedly opened reference")
        s_field = candidate_prediction(case, (1.30, 0.0, 0.25, 0.0))
        e_field = exposure(case)
        if s_field.shape != e_field.shape or not np.all(np.isfinite(s_field)) or not np.all(np.isfinite(e_field)):
            raise ValueError("non-finite or mismatched frozen V2 fields")
        positive = np.maximum(s_field, 0.0)
        negative = np.minimum(s_field, 0.0)
        fields = {
            "mask_gated_se_v2": negative + np.where(e_field < 0.05, positive, 0.0),
            "continuous_se_v2": negative + (1.0 - e_field) * positive,
            "s_only_v2": s_field,
        }
        s_retention = np.divide(
            np.maximum(s_field, 0.0), np.maximum(raw, 1.0e-12),
            out=np.ones_like(raw), where=raw > 0.0,
        )

    axes = 30.0 * np.arange(300, dtype=np.float64)
    terrain = RegularGridInterpolator((axes, axes), dem, bounds_error=False, fill_value=np.nan)
    horizontal = np.asarray(receipt["input_horizontal_uv_mps"], dtype=np.float64)
    aircraft = QuasiSteadyAircraft(AircraftParameters(
        reference_cruise_airspeed_mps=old_input.planner.FIXED_AIRSPEED_MPS,
        max_bank_deg=old_input.planner.MAXIMUM_BANK_DEG,
    ))
    zero_updraft = zero_thrust_sink_rate_mps(aircraft, old_input.planner.FIXED_AIRSPEED_MPS)
    worlds = {}
    original_transition = old_input.route_screen.TRANSITION_LENGTH_M
    try:
        old_input.route_screen.TRANSITION_LENGTH_M = float(receipt["transition_length_m"])
        for arm, values in fields.items():
            wind = old_input.build_wind(
                terrain=terrain, flat=flat, heights_m=heights,
                vertical_hc_mps=old_input.scaled_positive_credit(values),
                horizontal_uv_mps=horizontal,
                zero_propulsion_updraft_mps=zero_updraft,
            )
            worlds[arm] = old_input.evaluate_world(
                mission=receipt["mission"], wind=wind, aircraft=aircraft, map_world=arm
            )
    finally:
        old_input.route_screen.TRANSITION_LENGTH_M = original_transition

    return {
        "case_id": case_id, "split": receipt["split"],
        "terrain_group_sha256": receipt["terrain_group_sha256"],
        "raw_selected_route": receipt["raw_selected_route"],
        "old_s_only_selected_route": receipt["ursa_selected_route"],
        "formula_domain": formula_domain,
        "ood_reason": ood_reason,
        "arms": {arm: {
            "selected_route": worlds[arm]["fine_selected_route"],
            "coarse_selected_route": worlds[arm]["coarse_selected_route"],
            "selection_stable": worlds[arm]["selection_stable"],
            "feasibility_stable": worlds[arm]["feasibility_stable"],
            "fine_routes": worlds[arm]["by_spacing"]["1.5"]["routes"],
        } for arm in ARMS},
        "v2_diagnostics": {
            "masked_fraction_e_ge_0p05": float(np.mean(e_field >= 0.05)) if e_field is not None else None,
            "mean_e": float(np.mean(e_field)) if e_field is not None else None,
            "mean_s_retention_on_raw_positive": (
                float(np.mean(s_retention[raw > 0.0]))
                if s_retention is not None and np.any(raw > 0.0) else None
            ),
        },
        "reference_output_numeric_open_count": 0,
        "elapsed_s": time.monotonic() - started,
    }


def route_by_id(rows: list[dict], route_id: str) -> dict:
    return next(row for row in rows if row["route_id"] == route_id)


def distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    a = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "minimum": float(np.min(a)), "maximum": float(np.max(a))}


def summarize(rows: list[dict], arm: str, reference_by_case: dict[str, dict]) -> dict:
    scored = []
    for row in rows:
        selected = row["arms"][arm]["selected_route"]
        ref = reference_by_case[row["case_id"]]
        raw_route = row["raw_selected_route"]
        ref_rows = ref["worlds"]["fuxi_w"]["full_w_route_rows"]
        raw_ref = route_by_id(ref_rows, raw_route)
        selected_ref = None if selected is None else route_by_id(ref_rows, selected)
        raw_energy = float(raw_ref["predicted_energy_j"]) if raw_ref["feasible"] else None
        selected_energy = float(selected_ref["predicted_energy_j"]) if selected_ref is not None and selected_ref["feasible"] else None
        delta = None if raw_energy is None or selected_energy is None else raw_energy - selected_energy
        scored.append({
            "case_id": row["case_id"], "split": row["split"], "raw_route": raw_route,
            "selected_route": selected, "raw_reference_energy_j": raw_energy,
            "selected_reference_energy_j": selected_energy, "delta_e_select_j": delta,
            "saving_percent": None if delta is None else 100.0 * delta / raw_energy,
            "changed_route": selected is not None and selected != raw_route,
        })
    finite = [r for r in scored if r["delta_e_select_j"] is not None]
    deltas = np.asarray([r["delta_e_select_j"] for r in finite], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.mean(rng.choice(deltas, size=(BOOTSTRAP_RESAMPLES, len(deltas)), replace=True), axis=1) if len(deltas) else np.asarray([])
    changed = [r for r in finite if r["changed_route"]]
    transitions = Counter(f"{r['raw_route']}->{r['selected_route']}" for r in scored)
    outcome = Counter(
        "better" if r["delta_e_select_j"] > 1e-9 else "worse" if r["delta_e_select_j"] < -1e-9 else "same"
        for r in finite
    )
    return {
        "case_count": len(scored), "finite_case_count": len(finite),
        "route_transition_matrix": dict(sorted(transitions.items())),
        "outcome_counts": dict(sorted(outcome.items())),
        "changed_route_count": len(changed),
        "delta_e_select_j": distribution(deltas.tolist()),
        "paired_bootstrap_mean_delta_e_j": {
            "seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
            "ci95_j": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))] if len(means) else [None, None],
        },
        "ratio_of_sums_saving_percent": 100.0 * float(np.sum(deltas)) / float(np.sum([r["raw_reference_energy_j"] for r in finite])) if finite else None,
        "changed_route_saving_percent": distribution([float(r["saving_percent"]) for r in changed]),
        "case_rows": scored,
    }


def cliff_delta(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    a, b = np.asarray(left), np.asarray(right)
    return float(np.mean(a[:, None] > b[None, :]) - np.mean(a[:, None] < b[None, :]))


def feature_analysis(summary: dict, diagnostics: dict[str, dict]) -> dict:
    with FEATURES.open(newline="", encoding="utf-8") as stream:
        table = {row["case_id"]: row for row in csv.DictReader(stream)}
    scored = [row for row in summary["case_rows"] if row["case_id"] in table]
    changed = [row for row in scored if row["changed_route"]]
    unchanged = [row for row in scored if not row["changed_route"]]
    result = {}
    for field in (
        "raw_predicted_downstream_advantage_vs_direct_j",
        "raw_downstream_extra_distance_vs_direct_m",
        "raw_downstream_extra_time_vs_direct_s",
        "ridge_along_flow_separation_m",
        "central_speed_mps",
    ):
        a = [float(table[row["case_id"]][field]) for row in changed]
        b = [float(table[row["case_id"]][field]) for row in unchanged]
        result[field] = {"changed_median": float(np.median(a)) if a else None,
                         "unchanged_median": float(np.median(b)) if b else None,
                         "cliffs_delta": cliff_delta(a, b)}
    for field in ("masked_fraction_e_ge_0p05", "mean_e", "mean_s_retention_on_raw_positive"):
        a = [diagnostics[row["case_id"]][field] for row in changed if diagnostics[row["case_id"]][field] is not None]
        b = [diagnostics[row["case_id"]][field] for row in unchanged if diagnostics[row["case_id"]][field] is not None]
        result[f"v2_{field}"] = {"changed_median": float(np.median(a)) if a else None,
                                 "unchanged_median": float(np.median(b)) if b else None,
                                 "cliffs_delta": cliff_delta(a, b)}
    return {"changed_count": len(changed), "unchanged_count": len(unchanged), "features": result}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_before_confirmation_reuse" or freeze.get("S_deficit_scale") != 1.3 or freeze.get("post_pressure_gamma") != 0.55:
        raise RuntimeError("current V2 freeze contract changed")
    paths = sorted((PANEL / "candidates").glob("case_*.json"))
    if args.limit:
        paths = paths[:args.limit]
    started = time.monotonic()
    rows, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, str(path)): path.stem for path in paths}
        for completed, future in enumerate(as_completed(futures), 1):
            case_id = futures[future]
            try:
                rows.append(future.result())
            except BaseException as exc:
                errors.append({"case_id": case_id, "error_type": type(exc).__name__, "error": str(exc)[:1000]})
            elapsed = time.monotonic() - started
            rate = completed / max(elapsed, 1e-12)
            print("URSA_V2_ROUTE_PROGRESS " + json.dumps({
                "completed": completed, "total": len(paths), "percent": 100.0 * completed / len(paths),
                "elapsed_s": elapsed, "throughput_cases_per_s": rate,
                "eta_s": (len(paths) - completed) / rate, "successful": len(rows), "errors": len(errors),
            }, sort_keys=True), flush=True)
    rows.sort(key=lambda row: row["case_id"])
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference_by_case = {row["case_id"]: row for row in reference["results"]}
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {
            "all_evaluable": summarize(rows, arm, reference_by_case),
            "raw_downstream_main": summarize([r for r in rows if r["raw_selected_route"] == "ridge_downstream"], arm, reference_by_case),
        }
    diagnostics = {row["case_id"]: row["v2_diagnostics"] for row in rows}
    for arm in ARMS:
        summaries[arm]["raw_downstream_feature_analysis"] = feature_analysis(summaries[arm]["raw_downstream_main"], diagnostics)
    out = {
        "schema": "ursa.v2-route-energy-rerun.v1",
        "status": "complete" if not errors and len(paths) == 325 else "partial_or_failed",
        "evidence_class": "retrospective_frozen_formula_engineering_rerun",
        "primary_arm": "mask_gated_se_v2", "sensitivity_arms": ["continuous_se_v2", "s_only_v2"],
        "requested_case_count": len(paths), "successful_case_count": len(rows), "errors": errors,
        "formula_domain_case_count": sum(bool(row["formula_domain"]) for row in rows),
        "ood_fallback_case_count": sum(not bool(row["formula_domain"]) for row in rows),
        "ood_fallback_reason_counts": dict(sorted(Counter(
            str(row["ood_reason"]) for row in rows if not row["formula_domain"]
        ).items())),
        "reference_output_numeric_open_count_during_map_selection": 0,
        "bindings": {str(path): sha256(path) for path in (PROTOCOL, FREEZE, REFERENCE, FEATURES, Path(__file__))},
        "summaries": summaries, "input_rows": rows,
        "runtime_s": time.monotonic() - started,
        "resource_configuration": {"processes": args.workers, "blas_openmp_threads_per_process": 1},
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("URSA_V2_ROUTE_COMPLETE " + json.dumps({"status": out["status"], "successful": len(rows), "errors": len(errors), "runtime_s": out["runtime_s"], "output": str(args.output)}, sort_keys=True), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
