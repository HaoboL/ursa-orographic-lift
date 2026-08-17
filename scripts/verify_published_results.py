#!/usr/bin/env python3
"""Independently recompute URSA headline results from committed case rows."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("base_only", "base_plus_S", "base_plus_E", "base_plus_S_plus_E")
FORMAL_RESULTS = {
    "three_carrier_45": {
        "path": Path("results/three_carrier_45/result.json"),
        "audit": Path("results/three_carrier_45/audit.json"),
        "sha256": "3d644833c39073e7b93c94d7dc7af594fdab5ab1271f1d2688b1ec07dbcc48f4",
        "schema": "ursa.v01-three-base45-four-arm-same-point.v2",
    },
    "bo04_932": {
        "path": Path("results/bo04_932/result.json"),
        "audit": Path("results/bo04_932/audit.json"),
        "sha256": "4fbbc3dc5afb88c0384f620ac05128347bc2034e2a0bbf44d042df2959089252",
        "schema": "ursa.v01-bo04-full932-four-arm.v1",
    },
    "route_325": {
        "path": Path("results/route_325/result.json"),
        "audit": Path("results/route_325/audit.json"),
        "sha256": "d7cb09add419f9f0539e7e8d1fcd5c5ff152770ff0bdfeb1c3d2b8319eada8fd",
        "schema": "ursa.route-energy-rerun.v2",
    },
}


class VerificationError(RuntimeError):
    """Raised when a committed result differs from its registered value."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise VerificationError(f"expected a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _close(
    actual: float,
    expected: float,
    label: str,
    *,
    atol: float = 1.0e-12,
) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise VerificationError(
            f"{label}: recomputed {actual!r}, stored/registered {expected!r}"
        )


def _numeric_leaves(value: dict[str, Any], prefix: tuple[str, ...] = ()):
    for key, item in value.items():
        path = prefix + (key,)
        if isinstance(item, dict):
            yield from _numeric_leaves(item, path)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            yield path, float(item)


def _nested(value: dict[str, Any], path: Iterable[str]) -> float | None:
    current: Any = value
    for key in path:
        current = current[key]
    return None if current is None else float(current)


def _group_macro(
    rows: list[dict[str, Any]],
    *,
    carrier: str,
    arm: str,
    metric_path: tuple[str, ...],
) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _nested(row["models"][carrier][arm], metric_path)
        if value is not None and math.isfinite(value):
            grouped[str(row["group"])].append(value)
    group_means = [
        float(np.mean(values)) for values in grouped.values() if values
    ]
    _require(bool(group_means), f"no finite values for {carrier}.{arm}.{metric_path}")
    return float(np.mean(group_means))


def _verify_model_aggregate(
    rows: list[dict[str, Any]],
    stored: dict[str, Any],
    *,
    carrier: str,
    label: str,
) -> None:
    group_count = len({str(row["group"]) for row in rows})
    for arm in ARMS:
        aggregate = stored[arm]
        _require(int(aggregate["count"]) == group_count, f"{label}.{arm}.count")
        for metric_path, expected in _numeric_leaves(aggregate):
            if metric_path == ("count",):
                continue
            actual = _group_macro(
                rows,
                carrier=carrier,
                arm=arm,
                metric_path=metric_path,
            )
            _close(actual, expected, f"{label}.{arm}.{'/'.join(metric_path)}")


def _percent_change(new: float, reference: float) -> float:
    return 100.0 * (new - reference) / reference


def _group_wins(
    rows: list[dict[str, Any]],
    *,
    carrier: str,
    challenger: str,
    reference: str,
) -> int:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {challenger: [], reference: []}
    )
    for row in rows:
        group = str(row["group"])
        grouped[group][challenger].append(
            float(row["models"][carrier][challenger]["mae"])
        )
        grouped[group][reference].append(
            float(row["models"][carrier][reference]["mae"])
        )
    return sum(
        float(np.mean(values[challenger])) < float(np.mean(values[reference]))
        for values in grouped.values()
    )


def _verify_45(data: dict[str, Any]) -> dict[str, Any]:
    rows = data["case_rows"]
    _require(data["status"] == "complete", "45-case result status")
    _require(len(rows) == int(data["case_count"]) == 45, "45-case row count")
    groups = {str(row["group"]) for row in rows}
    _require(len(groups) == int(data["terrain_group_count"]) == 16, "45-case group count")

    registered = {
        "evve": (-10.162556668384255, -7.4097383436951105),
        "bo04": (-12.327352881664387, -10.019434309995368),
        "windninja": (-4.1943884506778035, -2.537247602208137),
    }
    output: dict[str, Any] = {}
    for carrier in ("evve", "bo04", "windninja"):
        aggregate = data["aggregate"][carrier]
        _verify_model_aggregate(
            rows,
            aggregate,
            carrier=carrier,
            label=f"45.{carrier}",
        )
        full = _percent_change(
            aggregate["base_plus_S"]["mae"], aggregate["base_only"]["mae"]
        )
        accepted = _percent_change(
            aggregate["base_plus_S_plus_E"]["mae"],
            aggregate["base_plus_E"]["mae"],
        )
        _close(full, registered[carrier][0], f"45.{carrier}.full contrast")
        _close(accepted, registered[carrier][1], f"45.{carrier}.accepted contrast")
        output[carrier] = {
            "full_support_mae_change_percent": full,
            "accepted_support_mae_change_percent": accepted,
        }
    return {"case_count": len(rows), "terrain_group_count": len(groups), "contrasts": output}


def _verify_coverage_932(rows: list[dict[str, Any]], stored: dict[str, Any]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group"])].append(float(row["coverage"]))
    coverage = np.asarray(
        [float(np.mean(values)) for values in grouped.values()], dtype=np.float64
    )
    _close(float(np.mean(coverage)), stored["terrain_group_macro_mean"], "932.coverage.mean")
    for percentile in (10, 25, 50, 75, 90):
        _close(
            float(np.percentile(coverage, percentile)),
            stored["terrain_group_percentiles"][f"p{percentile}"],
            f"932.coverage.p{percentile}",
        )


def _verify_932(data: dict[str, Any]) -> dict[str, Any]:
    rows = data["case_rows"]
    _require(data["status"] == "complete", "932-case result status")
    _require(len(rows) == int(data["case_count"]) == 932, "932-case row count")
    groups = {str(row["group"]) for row in rows}
    _require(len(groups) == int(data["terrain_group_count"]) == 594, "932-case group count")
    aggregate = data["aggregate"]["models"]["bo04"]
    _verify_model_aggregate(rows, aggregate, carrier="bo04", label="932.bo04")
    _verify_coverage_932(rows, data["aggregate"]["coverage"])

    full = _percent_change(
        aggregate["base_plus_S"]["mae"], aggregate["base_only"]["mae"]
    )
    accepted = _percent_change(
        aggregate["base_plus_S_plus_E"]["mae"], aggregate["base_plus_E"]["mae"]
    )
    stored_contrasts = data["legal_mae_contrasts_percent"]
    _close(
        full,
        stored_contrasts["base_to_base_plus_S_full_support"],
        "932.full contrast",
    )
    _close(
        accepted,
        stored_contrasts["base_plus_E_to_base_plus_S_plus_E_same_support"],
        "932.accepted contrast",
    )
    full_wins = _group_wins(
        rows,
        carrier="bo04",
        challenger="base_plus_S",
        reference="base_only",
    )
    accepted_wins = _group_wins(
        rows,
        carrier="bo04",
        challenger="base_plus_S_plus_E",
        reference="base_plus_E",
    )
    _require(full_wins == 593, f"932 full-support group wins: {full_wins}")
    _require(accepted_wins == 591, f"932 accepted-support group wins: {accepted_wins}")
    return {
        "case_count": len(rows),
        "terrain_group_count": len(groups),
        "full_support_mae_change_percent": full,
        "accepted_support_mae_change_percent": accepted,
        "full_support_groups_improved": full_wins,
        "accepted_support_groups_improved": accepted_wins,
    }


def _verify_route_summary(summary: dict[str, Any], label: str) -> dict[str, Any]:
    rows = summary["case_rows"]
    _require(summary["status"] == "complete", f"{label}.status")
    _require(len(rows) == int(summary["case_count"]), f"{label}.case count")

    delta = np.asarray([row["delta_e_select_j"] for row in rows], dtype=np.float64)
    raw_energy = np.asarray(
        [row["raw_reference_energy_j"] for row in rows], dtype=np.float64
    )
    selected_energy = np.asarray(
        [row["selected_reference_energy_j"] for row in rows], dtype=np.float64
    )
    defined = np.asarray(
        [bool(row["reference_denominator_defined"]) for row in rows], dtype=bool
    )
    changed = np.asarray([bool(row["changed_route"]) for row in rows], dtype=bool)
    _require(np.all(defined), f"{label}.undefined reference denominator")
    _require(np.allclose(delta, raw_energy - selected_energy, rtol=0.0, atol=1.0e-9), f"{label}.energy identity")
    _require(
        all(bool(row["changed_route"]) == (row["raw_route"] != row["selected_route"]) for row in rows),
        f"{label}.route-change identity",
    )

    headline = summary["headline"]
    energy_stats = headline["delta_e_select_j"]
    _require(int(energy_stats["count"]) == len(rows), f"{label}.delta count")
    for name, actual in {
        "mean": float(np.mean(delta)),
        "median": float(np.median(delta)),
        "minimum": float(np.min(delta)),
        "maximum": float(np.max(delta)),
    }.items():
        _close(actual, energy_stats[name], f"{label}.delta {name}", atol=1.0e-9)

    changed_saving = np.asarray(
        [float(row["saving_percent"]) for row in rows if row["changed_route"]],
        dtype=np.float64,
    )
    changed_stats = headline["changed_route_saving_percent"]
    _require(int(changed_stats["count"]) == int(np.sum(changed)), f"{label}.changed count")
    for name, actual in {
        "mean": float(np.mean(changed_saving)),
        "median": float(np.median(changed_saving)),
        "minimum": float(np.min(changed_saving)),
        "maximum": float(np.max(changed_saving)),
    }.items():
        _close(actual, changed_stats[name], f"{label}.changed saving {name}", atol=1.0e-9)

    ratio = 100.0 * float(np.sum(delta)) / float(np.sum(raw_energy))
    _close(
        ratio,
        headline["ratio_of_sums_saving_percent"],
        f"{label}.ratio of sums",
        atol=1.0e-10,
    )

    bootstrap = headline["paired_bootstrap_mean_delta_e_j"]
    seed = int(bootstrap["seed"])
    resamples = int(bootstrap["resamples"])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
    bootstrap_means = np.mean(delta[indices], axis=1)
    interval = np.quantile(bootstrap_means, [0.025, 0.975])
    for index, bound in enumerate(interval):
        _close(
            float(bound),
            bootstrap["ci95_j"][index],
            f"{label}.bootstrap bound {index}",
            atol=1.0e-9,
        )

    outcome = Counter(
        "better" if value > 0.0 else "worse" if value < 0.0 else "same"
        for value in delta
    )
    _require(dict(outcome) == summary["outcome_counts"], f"{label}.outcome counts")
    transitions = Counter(
        f"{row['raw_route']}->{row['selected_route']}" for row in rows
    )
    _require(
        dict(transitions) == summary["route_transition_matrix"],
        f"{label}.route transitions",
    )
    _require(int(np.sum(changed)) == int(summary["changed_route_count"]), f"{label}.changed-route count")
    _require(int(np.sum(defined)) == int(summary["finite_case_count"]), f"{label}.finite count")
    _require(int(np.sum(~defined)) == int(summary["undefined_case_count"]), f"{label}.undefined count")
    return {
        "case_count": len(rows),
        "changed_route_count": int(np.sum(changed)),
        "mean_delta_energy_j": float(np.mean(delta)),
        "paired_bootstrap_ci95_j": [float(value) for value in interval],
        "ratio_of_sums_saving_percent": ratio,
        "outcome_counts": dict(outcome),
    }


def _verify_route(data: dict[str, Any]) -> dict[str, Any]:
    _require(data["status"] == "complete", "route result status")
    _require(int(data["requested_case_count"]) == 325, "route requested case count")
    _require(int(data["successful_case_count"]) == 325, "route successful case count")
    summaries = data["summaries"]["hard_exclusion"]["fuxi_w"]
    return {
        "all_325": _verify_route_summary(summaries["all_325"], "route.all_325"),
        "raw_downstream_itt_96": _verify_route_summary(
            summaries["raw_downstream_itt_96"], "route.raw_downstream_itt_96"
        ),
    }


def verify_all(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, specification in FORMAL_RESULTS.items():
        result_path = repository_root / specification["path"]
        audit_path = repository_root / specification["audit"]
        actual_hash = _sha256(result_path)
        _require(
            actual_hash == specification["sha256"],
            f"{name} SHA-256 differs: {actual_hash}",
        )
        data = _load_json(result_path)
        _require(data.get("schema") == specification["schema"], f"{name} schema")
        audit = _load_json(audit_path)
        _require(audit.get("status") == "pass", f"{name} independent audit status")
        loaded[name] = data
        hashes[str(specification["path"])] = actual_hash

    return {
        "status": "pass",
        "formal_result_sha256": hashes,
        "three_carrier_45": _verify_45(loaded["three_carrier_45"]),
        "bo04_932": _verify_932(loaded["bo04_932"]),
        "route_325": _verify_route(loaded["route_325"]),
    }


def main() -> None:
    print(json.dumps(verify_all(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
