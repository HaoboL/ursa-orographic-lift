# Script status

## Portable release script

`verify_published_results.py` is self-contained apart from NumPy and the JSON
files committed under `results/`. It is the supported public command for
checking the published aggregates.

## Archived research drivers

The other scripts are frozen research-driver snapshots. They preserve the
original argument structure, model bindings, worker contracts, and provenance,
but some import study-local data adapters or external packages that are not
part of this compact repository. Their presence documents the experimental
workflow; it does not imply that third-party raw data or every local adapter is
redistributed.

The exact formal row-level outputs and same-workflow audit receipts produced by
the final study workflow are under `results/`. Use those files and the portable
verifier for a row-level recomputation of the paper's numerical claims. This is
a separate implementation-and-aggregation check within the same research
workflow, not an independent team, experiment, or reference dataset.
