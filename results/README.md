# Compact published results

These directories contain the minimum row-level evidence supporting the
reported URSA comparisons. Files are exact copies of the completed formal
outputs; they have not been shortened, rounded, or rewritten for GitHub.

## Directory map

| Directory | Scope | Primary result schema |
|---|---|---|
| `three_carrier_45/` | 45 cases, 16 terrain groups, EVVE/BO04/WindNinja, four arms | `ursa.v01-three-base45-four-arm-same-point.v2` |
| `bo04_932/` | 932 cases, 594 terrain groups, BO04, four arms | `ursa.v01-bo04-full932-four-arm.v1` |
| `route_325/` | 325 continuous-turn route tasks plus registered strata | `ursa.route-energy-rerun.v2` |

Each directory includes the formal `result.json`, an independent `audit.json`,
and machine resource or preparation receipts. `route_325/formula_smoke.json`
records the formula-domain and identity-fallback smoke gate.

## Immutable formal-result hashes

| File | SHA-256 |
|---|---|
| `three_carrier_45/result.json` | `3d644833c39073e7b93c94d7dc7af594fdab5ab1271f1d2688b1ec07dbcc48f4` |
| `bo04_932/result.json` | `4fbbc3dc5afb88c0384f620ac05128347bc2034e2a0bbf44d042df2959089252` |
| `route_325/result.json` | `d7cb09add419f9f0539e7e8d1fcd5c5ff152770ff0bdfeb1c3d2b8319eada8fd` |

Run `python scripts/verify_published_results.py` from the repository root to
validate these hashes and independently recompute all headline aggregates from
the case rows.

## Data boundary

The JSON files contain processed metrics, group identifiers, route decisions,
and execution provenance. They do not redistribute FuXi-CFD, Perdigao, Karim
PIV, ESA WorldCover, or other third-party raw fields. Source datasets remain at
the DOI records linked from the main README.

Some original receipts contain absolute workstation paths. Those values are
preserved because the files are hash-bound; they are provenance strings only
and are not opened by the public verifier. Historical internal schema strings
are retained for the same reason and are not paper-facing model labels.
