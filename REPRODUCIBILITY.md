# Reproducibility guide

This release separates three reproducibility levels so that each public claim
matches what the repository actually contains.

## Level 1: install and execute the model core

Use Python 3.11 and the exact package set in `environment/requirements.txt`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m pip install -e .
python examples/synthetic_shelter.py
pytest -q tests/test_core.py
```

The synthetic example creates two analytic ridges, constructs the flow-aligned
ridge inventory, and evaluates the far-wake retained-lift ratio without any
downloaded data.

As an alternative, run `conda env create -f environment/environment.yml` from
the repository root and activate the resulting `ursa-orographic-lift`
environment.

## Level 2: separately recompute the reported results

```bash
python scripts/verify_published_results.py
```

The verifier performs the following operations from the redistributed
case-level rows:

1. validates the three immutable result-file SHA-256 values and the archived
   same-workflow audit status;
2. rebuilds the 16-group macro means for all four arms and three base
   estimators in the 45-case panel;
3. rebuilds the 594-group BO04 macro means, both legal MAE contrasts, and the
   two groupwise win counts in the 932-case extension;
4. rebuilds route-change counts, outcome counts, transition counts, mean energy
   difference, complete-denominator ratio-of-sums saving, and the registered
   5,000-resample paired bootstrap interval for all 325 tasks and the 96-task
   downstream-ridge stratum;
5. separately sums the unrounded row-level energies for the policy-defined 19
   changed routes, reproducing 724.582181 kJ raw, 574.706239 kJ selected,
   149.875942 kJ net saving, and a descriptive changed-route ratio of 20.684464%.
6. filters all 4,864 rows of the 608-task V2 package to the primary FuXi-$w$
   world, verifies 608 tasks and 161 exact-DEM groups for each frozen map, and
   rebuilds false-corridor correction, reference-valid and valid-lift
   abandonment, finite gross benefit/harm, feasibility transitions, route
   changes, and worst finite harm.

No manuscript table is used as an input. A successful run prints a JSON record
with `"status": "pass"`; any mismatch raises an error and returns a nonzero exit
status.

## Level 3: regenerate case rows from third-party raw data

Raw-to-case regeneration additionally requires the public datasets linked in
the main README, the WindNinja executable, OpenFOAM for the controlled
qualitative environment, and study data adapters. The archived driver scripts
preserve command-line structure and provenance, but the current public package
does not redistribute all data adapters or third-party fields. Therefore this
release does **not** label those scripts as a self-contained raw-to-result
workflow.

This boundary is deliberate: the minimum row-level data supporting the central
results are public and separately recomputable, while large or third-party
inputs remain at their authoritative repositories. The recomputation belongs
to the same research workflow and is not an independent experiment, team, or
reference dataset. Absolute paths embedded in the original JSON receipts are
preserved provenance strings and are not required by the portable verifier.

## Recorded resource contract

The original runs avoided nested thread oversubscription:

| Stage | Processes/workers | Threads per process |
|---|---:|---:|
| 45-case heavy preparation | 4 | 1 |
| 45-case cache-only scoring | 16 | 1 |
| 932-case preparation | 4 | 1 |
| 932-case scoring | 16 | 1 |
| 325-route evaluation | 12 | 1 |
| 608-task V2 mission preparation/reference scoring | 16 | 1 |

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
`NUMEXPR_NUM_THREADS` were set to `1` for multi-process runs. These settings and
runtime receipts are embedded in the corresponding result directories.
