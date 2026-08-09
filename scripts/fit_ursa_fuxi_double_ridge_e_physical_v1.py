#!/usr/bin/env python3
"""Select and confirm the continuous physics-based URSA E exposure head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(os.environ.get("URSA_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "scripts"))
from fit_ursa_fuxi_double_ridge_physical_v1 import PANEL, load_case  # noqa: E402
from ursa_v10_dual_head_model_v1 import (  # noqa: E402
    LIU_REATTACHMENT_X_OVER_L_3D,
    LIU_SEPARATION_X_OVER_L_3D,
    LIU_SLOPE_ANCHORS,
    separation_activation,
)
from ursa_v10_dual_head_model_d4_v1 import (  # noqa: E402
    cosine_squared_separation_elevation_m,
    terrain_cavity_vertical_membership,
)


DEFAULT_OUTPUT = ROOT / "results/ursa_fuxi_double_ridge_physical_v1/e_development_v1.json"
EVENT_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(wanted: set[str], max_cases: int | None) -> tuple[list[dict], list[dict]]:
    paths = []
    for path in sorted((PANEL / "candidates").glob("case_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["split"] in wanted:
            paths.append(path)
    if max_cases is not None:
        paths = paths[:max_cases]
    cases, exclusions = [], []
    started = time.monotonic()
    for index, path in enumerate(paths, 1):
        try:
            cases.append(load_case(path))
        except ValueError as exc:
            exclusions.append({"case_id": path.stem, "reason": str(exc)})
        elapsed = time.monotonic() - started
        rate = index / max(elapsed, 1.0e-12)
        print(json.dumps({
            "stage": "load",
            "completed": index,
            "total": len(paths),
            "percent": 100.0 * index / len(paths),
            "elapsed_s": elapsed,
            "eta_s": (len(paths) - index) / rate,
        }), flush=True)
    return cases, exclusions


def predict_e(case: dict, params: tuple[float, float, float]) -> np.ndarray:
    c_x, c_z, c_n = params
    e = case["e_inputs"]
    output = np.zeros_like(np.asarray(e["reference_aligned_mps"], dtype=np.float64))
    ground = np.asarray(e["query_ground_elevation_m"], dtype=np.float64)
    components = sorted(e["components"], key=lambda item: item["crest_along_flow_m"])
    for component_index, component in enumerate(components):
        x = np.asarray(component["downwind_from_crest_m"], dtype=np.float64)
        n = np.asarray(component["crosswind_from_component_m"], dtype=np.float64)
        span = float(component["component_crosswind_span_m"])
        slope = float(component["leeward_max_slope_tangent"])
        length = float(component["leeward_valley_length_m"])
        base_elevation = float(component["leeward_base_elevation_m"])
        component_height = float(component["leeward_height_m"])
        if (
            not np.isfinite(slope) or not np.isfinite(length) or length <= 0.0
            or not np.isfinite(base_elevation) or not np.isfinite(component_height)
            or component_height <= 0.0
        ):
            continue
        if not np.isfinite(span) or span <= 0.0:
            span = np.inf
        x_sep = float(np.interp(slope, LIU_SLOPE_ANCHORS, LIU_SEPARATION_X_OVER_L_3D)) * length
        x_re = c_x * float(np.interp(
            slope, LIU_SLOPE_ANCHORS, LIU_REATTACHMENT_X_OVER_L_3D
        )) * length
        active_x = (x >= x_sep) & (x <= x_re)
        separation_elevation = cosine_squared_separation_elevation_m(
            leeward_base_elevation_m=base_elevation,
            component_height_m=component_height,
            leeward_length_m=length,
            separation_distance_m=x_sep,
        )
        lateral_sigma = 0.5 * span + c_n * np.maximum(x, 0.0)
        lateral = np.exp(-0.5 * np.square(n / lateral_sigma))
        local_e = np.zeros_like(output)
        valley_e = np.zeros_like(output)
        activation = float(separation_activation(slope))
        for hi, agl in enumerate(case["heights"]):
            mixing_depth = c_z * np.maximum(x - x_sep, 0.0)
            local_vertical = np.clip(np.divide(
                mixing_depth - agl,
                mixing_depth,
                out=np.zeros_like(mixing_depth),
                where=mixing_depth > 0.0,
            ), 0.0, 1.0)
            local_e[hi] = np.minimum.reduce((
                np.full_like(x, activation), active_x.astype(np.float64),
                local_vertical, lateral
            ))
        # The terrain-following local layer is valid for each lee wake.  The
        # absolute-elevation cavity is a distinct, pair-specific channel: it
        # exists only when this source's nominal separated layer reaches the
        # next crest and only inside that inter-crest valley.  This encodes the
        # double-hill circulation mechanism without declaring every low point
        # behind every source to be separated.
        if component_index + 1 < len(components):
            gap = (
                float(components[component_index + 1]["crest_along_flow_m"])
                - float(component["crest_along_flow_m"])
            )
            if gap > x_sep and gap <= x_re:
                valley_x = (x >= x_sep) & (x <= gap)
                for hi, agl in enumerate(case["heights"]):
                    absolute_vertical = terrain_cavity_vertical_membership(
                        downwind_distance_m=np.maximum(x, 0.0),
                        separation_distance_m=x_sep,
                        reattachment_distance_m=x_re,
                        separation_elevation_m=separation_elevation,
                        leeward_base_elevation_m=base_elevation,
                        query_absolute_elevation_m=ground + agl,
                        c_z=c_z,
                    )
                    valley_e[hi] = np.minimum.reduce((
                        np.full_like(x, activation), valley_x.astype(np.float64),
                        absolute_vertical, lateral,
                    ))
        output = np.maximum(output, np.maximum(local_e, valley_e))
    return output


def case_metrics(case: dict, prediction: np.ndarray) -> dict | None:
    aligned = np.asarray(case["e_inputs"]["reference_aligned_mps"], dtype=np.float64)
    valid = np.isfinite(aligned) & np.isfinite(prediction)
    truth = aligned[valid] < 0.0
    pred = prediction[valid]
    if pred.size == 0:
        return None
    event = pred >= EVENT_THRESHOLD
    tp = int(np.count_nonzero(event & truth))
    fp = int(np.count_nonzero(event & ~truth))
    fn = int(np.count_nonzero(~event & truth))
    return {
        "sample_count": int(pred.size),
        "event_prevalence": float(np.mean(truth)),
        "predicted_exposure_mean": float(np.mean(pred)),
        "brier": float(np.mean(np.square(pred - truth.astype(np.float64)))),
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "false_safe_probability": fn / pred.size,
        "false_hazard_probability": fp / pred.size,
    }


def aggregate(cases: list[dict], params: tuple[float, float, float]) -> dict:
    rows = [case_metrics(case, predict_e(case, params)) for case in cases]
    rows = [row for row in rows if row is not None]
    if not rows:
        raise ValueError("no cases have finite FuXi horizontal-wind E labels")
    result = {"case_count": len(rows), "sample_count": sum(r["sample_count"] for r in rows)}
    for key in (
        "event_prevalence", "predicted_exposure_mean", "brier", "recall",
        "precision", "false_safe_probability", "false_hazard_probability",
    ):
        values = [r[key] for r in rows if r[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def admissible(metrics: dict) -> bool:
    return (
        metrics["recall"] is not None and metrics["recall"] >= 0.85
        and metrics["precision"] is not None and metrics["precision"] >= 0.20
        and metrics["false_hazard_probability"] <= 0.35
    )


def family_recall_ceiling(cases: list[dict]) -> dict:
    """Attribute the E-family event-recall ceiling to each physics factor."""
    rows = []
    for case in cases:
        e = case["e_inputs"]
        aligned = np.asarray(e["reference_aligned_mps"], dtype=np.float64)
        truth = np.isfinite(aligned) & (aligned < 0.0)
        if not np.any(truth):
            continue
        x = np.asarray(e["downwind_from_crest_m"], dtype=np.float64)
        n = np.asarray(e["crosswind_from_component_m"], dtype=np.float64)
        span = float(e["component_crosswind_span_m"])
        slope = float(e["leeward_max_slope_tangent"])
        length = float(e["leeward_valley_length_m"])
        x_sep = float(np.interp(slope, LIU_SLOPE_ANCHORS, LIU_SEPARATION_X_OVER_L_3D)) * length
        x_re = (5.1 / 1.4) * float(np.interp(
            slope, LIU_SLOPE_ANCHORS, LIU_REATTACHMENT_X_OVER_L_3D
        )) * length
        activation = np.full(
            aligned.shape,
            float(separation_activation(slope)) >= EVENT_THRESHOLD,
            dtype=bool,
        )
        stream = np.broadcast_to(((x >= x_sep) & (x <= x_re))[None, :], aligned.shape)
        mixing = 0.30 * np.maximum(x - x_sep, 0.0)
        vertical = np.stack([
            np.divide(agl, mixing, out=np.full_like(mixing, np.inf), where=mixing > 0.0)
            for agl in case["heights"]
        ])
        vertical = (1.0 - vertical) >= EVENT_THRESHOLD
        sigma = 0.5 * span + 0.20 * np.maximum(x, 0.0)
        lateral = np.broadcast_to(
            (np.exp(-0.5 * np.square(n / sigma)) >= EVENT_THRESHOLD)[None, :], aligned.shape
        )
        factors = {
            "separation_activation": activation,
            "maximum_streamwise_extent": stream,
            "maximum_vertical_extent": vertical,
            "maximum_lateral_extent": lateral,
            "combined_single_source_ceiling": activation & stream & vertical & lateral,
        }
        rows.append({key: float(np.count_nonzero(value & truth) / np.count_nonzero(truth)) for key, value in factors.items()})
    return {
        "case_count_with_reverse_flow": len(rows),
        **(
            {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            if rows else {}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation"), default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    wanted = {"train", "development"} if args.phase == "development" else {"confirmation"}
    cases, exclusions = load_cases(wanted, args.max_cases)
    if args.phase == "development":
        train = [c for c in cases if c["split"] == "train"]
        development = [c for c in cases if c["split"] == "development"]
        if not train or not development:
            raise ValueError("non-empty train and development splits are required")
        grid = [
            (float(c_x), float(c_z), float(c_n))
            for c_x in np.linspace(1.0, 5.1 / 1.4, 6)
            for c_z in np.linspace(0.10, 0.30, 6)
            for c_n in np.linspace(0.0, 0.20, 6)
        ]
        scored = []
        grid_started = time.monotonic()
        for index, params in enumerate(grid, 1):
            row = {"parameters": params, "metrics": aggregate(development, params)}
            scored.append(row)
            if index % 25 == 0 or index == len(grid):
                elapsed = time.monotonic() - grid_started
                rate = index / max(elapsed, 1.0e-12)
                print(json.dumps({
                    "stage": "e_physics_grid", "completed": index, "total": len(grid),
                    "percent": 100.0 * index / len(grid), "elapsed_s": elapsed,
                    "eta_s": (len(grid) - index) / rate,
                }), flush=True)
        accepted = [row for row in scored if admissible(row["metrics"])]
        pool = accepted if accepted else scored
        selected = min(pool, key=lambda row: row["metrics"]["brier"])
        params = tuple(selected["parameters"])
        result = {
            "schema": "ursa.fuxi-double-ridge-continuous-E-hybrid-development.v1",
            "status": "complete" if accepted else "no_admissible_E_candidate",
            "selection_rule": "independent E selection: recall>=0.85, precision>=0.20, false-hazard<=0.35, then minimum case-equal Brier",
            "candidate_count": len(grid),
            "admissible_candidate_count": len(accepted),
            "selected_parameters": {"c_x": params[0], "c_z": params[1], "c_n": params[2]},
            "development": selected["metrics"],
            "train_readback": aggregate(train, params),
            "development_family_recall_ceiling": family_recall_ceiling(development),
        }
    else:
        if args.freeze is None:
            raise ValueError("--freeze is required")
        frozen = json.loads(args.freeze.read_text(encoding="utf-8"))
        p = frozen["selected_parameters"]
        params = (p["c_x"], p["c_z"], p["c_n"])
        result = {
            "schema": "ursa.fuxi-double-ridge-continuous-E-confirmation.v1",
            "status": "complete",
            "freeze_path": str(args.freeze),
            "freeze_sha256": sha256(args.freeze),
            "selected_parameters": p,
            "confirmation": aggregate(cases, params),
        }
    result.update({
        "requested_case_count": len(cases) + len(exclusions),
        "scored_case_count": len(cases),
        "extraction_exclusion_count": len(exclusions),
        "extraction_exclusions": exclusions,
        "event_label": "FuXi reference horizontal wind projected on frozen case flow direction is negative",
        "continuous_severity_label": "max(0,-reference_aligned_mps); reported separately, not fabricated as probability",
        "physics": {
            "separation_activation": "linear uncertainty ramp in leeward max slope tangent 0.20--0.32",
            "streamwise": "Liu 3D separation/reattachment slope anchors with selected c_x natural-terrain transfer",
            "vertical": "terrain-following local mixing layer plus pair-gated absolute-elevation inter-crest cavity",
            "lateral": "finite-span Gaussian bypass with sigma=0.5*span+c_n*x",
            "aggregation": "maximum across local wakes and connected adjacent-crest valley channels",
        },
        "machine_learning_model": False,
        "openfoam_runs": 0,
        "runtime_s": time.monotonic() - started,
        "script_sha256": sha256(Path(__file__)),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "runtime_s": result["runtime_s"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
