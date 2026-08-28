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
| `route_608_tradeoff_v2/` | 608 mechanism-conditioned tasks, 161 terrain groups, four frozen maps, two reference worlds | `orocfd.ursa-bo04-downstream-challenge-analysis.v2` |

The three legacy directories include a formal `result.json`, a same-workflow
`audit.json`, and machine resource or preparation receipts. The V2 route
directory instead publishes exact CSV rows, a JSON analysis receipt, the
registered aggregate table, and the group profile. Historical audit field and
schema names are retained unchanged because the receipts are hash-bound;
`independent` in those historical identifiers means a separate implementation
or aggregation path within the same research workflow, not an independent
team, experiment, or reference dataset. `route_325/formula_smoke.json` records
the formula-domain and identity-fallback smoke gate.

## Immutable formal-result hashes

| File | SHA-256 |
|---|---|
| `three_carrier_45/result.json` | `3d644833c39073e7b93c94d7dc7af594fdab5ab1271f1d2688b1ec07dbcc48f4` |
| `bo04_932/result.json` | `4fbbc3dc5afb88c0384f620ac05128347bc2034e2a0bbf44d042df2959089252` |
| `route_325/result.json` | `d7cb09add419f9f0539e7e8d1fcd5c5ff152770ff0bdfeb1c3d2b8319eada8fd` |
| `route_608_tradeoff_v2/task_method_results.csv` | `271f99b23255b8224e59c11b392098cb22863c8f92eb38f4d03820ae20b6c442` |
| `route_608_tradeoff_v2/analysis_summary.json` | `ee6d0cf3fc3f74657a5feb5c88e47e0ae8222bda3822866deeb2f20b961dc0a9` |

Run `python scripts/verify_published_results.py` from the repository root to
validate these hashes and separately recompute all headline aggregates from the
case rows, including the preserved 325-task result and the 608-task symmetric
false-corridor-correction versus valid-opportunity-abandonment audit.

## Data boundary

The JSON files contain processed metrics, group identifiers, route decisions,
and execution provenance. They do not redistribute FuXi-CFD, Perdigao, Karim
PIV, ESA WorldCover, or other third-party raw fields. Source datasets remain at
the DOI records linked from the main README.

Some original receipts contain absolute workstation paths. Those values are
preserved because the files are hash-bound; they are provenance strings only
and are not opened by the public verifier. Historical internal schema strings
are retained for the same reason and are not paper-facing model labels.

## License

Except where accompanying metadata states otherwise, the original processed
results and audit receipts in this directory are available under
[CC BY 4.0](LICENSE). The cited third-party raw datasets are not redistributed
and are not covered by this license.
