# 608-task correction--opportunity audit

This directory contains the compact, task-level evidence for the post-audit
mechanism-conditioned UAV route panel. It supplements rather than overwrites
the frozen 39-task pilot and the earlier `route_325/` package.

The panel is a scenario-based stress test of a specified failure mechanism,
not an estimate of natural mission frequency. Mission inputs and all four
planning maps were frozen before FuXi reference scoring. The input receipts
recorded zero reference-output reads during mission generation.

## Files

- `task_method_results.csv`: 608 tasks, four frozen planning maps, and two
  reference worlds (4,864 task--method--world rows);
- `analysis_summary.json`: pooled, split, cohort, and 5,000-resample exact-DEM
  cluster-bootstrap summaries;
- `primary_tradeoff_table.csv`: compact registered aggregates;
- `exact_dem_group_tradeoff_profile.csv`: group-level rates used in the
  distribution plot;
- `correction_opportunity_tradeoff.pdf` and `figure_receipt.json`: vector
  figure and hash-bound rendering receipt;
- `MISSION_LIBRARY_PROTOCOL_V2_ZH.md` and `DECISION_LOG_ZH.md`: protocol and
  decision history, including the preserved adverse pilot result.

## Co-primary decision outcomes

| Frozen map | False corridors corrected | Valid routes abandoned | Valid-lift opportunities abandoned | Finite benefit/harm | Hard harm/benefit |
|---|---:|---:|---:|---:|---:|
| Hard warning | 52/52 | 145/406 | 39/129 | 933.18/1270.70 kJ | 43/8 |
| Continuous attenuation | 29/52 | 21/406 | 3/129 | 617.71/62.91 kJ | 11/5 |
| Matched uniform | 26/52 | 55/406 | 5/129 | 565.78/240.72 kJ | 27/2 |
| Retained-lift factor only | 7/52 | 6/406 | 2/129 | 95.97/47.25 kJ | 1/0 |

The hard warning is rejected as a safe operating point. Continuous attenuation
has the best empirical balance in this frozen panel, but its 11 hard harms and
15.12 kJ worst finite harm preclude a deployment guarantee. The public
verifier independently rebuilds these denominators and energy totals from
`task_method_results.csv`.

Third-party FuXi fields are not redistributed. The processed rows and receipts
are available under the repository's results license.
