# Recorded software and compute environment

## Portable Python package

- Python 3.11.5
- NumPy 1.26.4
- SciPy 1.17.1
- pandas 2.3.3
- Matplotlib 3.11.0
- xarray 2026.7.0
- pytest 8.4.2

Only NumPy and SciPy are required by `src/ursa`. NumPy is used by the public
result verifier, pandas by archived tabular drivers, Matplotlib by the study
figure pipeline, xarray by gridded-data adapters, and pytest by the release
tests.

## External study software

- WindNinja 3.12.2, source commit `35ff789b`, native mass-conserving solver;
- OpenFOAM Foundation 13 for the controlled qualitative CFD environment.

The compact result verification does not invoke WindNinja or OpenFOAM. Their
versions are recorded because they formed part of the original experimental
environment.

## Parallel execution

The formal 45- and 932-case workflows used 4 processes for data-intensive
preparation and 16 processes for cache-only scoring. The 325-task continuous-
turn evaluation used 12 processes. In each multi-process stage, OpenMP, MKL,
OpenBLAS, and NumExpr were restricted to one thread per process. The exact
resource readbacks are preserved beside each result.
