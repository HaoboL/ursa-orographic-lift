# URSA orographic-lift adapter

[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](LICENSE)
[![Results: CC BY 4.0](https://img.shields.io/badge/Results-CC%20BY%204.0-lightgrey.svg)](results/LICENSE)

URSA (Upstream-Ridge Sheltering Attenuation) is a deterministic,
physics-guided adapter that adds upstream terrain history to fast
orographic-lift estimators. It attenuates unsupported positive lift behind
upstream ridges while preserving negative vertical velocity. A separate
exposure state supplies warning and accepted-support decisions; it is not a
second velocity multiplier.

This repository publishes the scientific code, an exact software environment,
compact row-level results, same-workflow implementation-and-aggregation audit
receipts, and a portable result verifier. These audit receipts and the verifier
provide a separate recomputation within the same research workflow; they are
not an independent team, experiment, or reference dataset. The journal
manuscript, its Chinese translation, Supplementary Information, reviewer
copies, and submission files are intentionally excluded.

## Published package

- `src/ursa/`: terrain geometry, sheltering, mixing-layer, cavity, and
  relaxing-wake primitives;
- `examples/synthetic_shelter.py`: data-free smoke example for the public API;
- `environment/`: exact Python and external-solver environment record;
- `results/three_carrier_45/`: 45 cases in 16 exact-terrain groups for EVVE,
  BO04, and WindNinja;
- `results/bo04_932/`: 932 BO04 cases in 594 terrain groups;
- `results/route_325/`: 325 continuous-turn route-selection tasks;
- `scripts/verify_published_results.py`: one-command, row-level recomputation
  of the headline aggregates and checksums;
- `REPRODUCIBILITY.md`: scope, commands, and raw-data boundary.

The compact results reproduce these registered comparisons without refitting:

| Evaluation | Published comparison | Result |
|---|---|---:|
| 45-case transfer | full-support MAE, base to base + sheltering | -10.16% EVVE; -12.33% BO04; -4.19% WindNinja |
| 45-case transfer | identical accepted support, warning-only base to sheltering + warning | -7.41% EVVE; -10.02% BO04; -2.54% WindNinja |
| 932-case BO04 extension | full-support / identical accepted-support MAE | -14.39% / -18.43% |
| 932-case BO04 extension | terrain groups improved | 593/594 / 591/594 |
| 325 route tasks | route outcomes | 13 better / 306 same / 6 worse; 19 routes changed |
| 325 route tasks | changed-route descriptive magnitude | 149.876 kJ net; 20.684% of changed raw-route energy, including all 6 unfavorable changes |
| 325 route tasks | complete-denominator primary result | 1.398%; mean 461.157 J/task (paired-bootstrap 95% CI 92.096--939.752 J) |

The changed-route statistic describes the magnitude of the policy-defined 19
interventions. It does not replace the primary complete-denominator result,
which retains all 325 tasks and assigns zero paired difference to the 306
unchanged route decisions.

## Quick start

Python 3.11 is the recorded environment. The core package supports Python
3.10 and newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
python -m pip install -e .
python examples/synthetic_shelter.py
python scripts/verify_published_results.py
pytest -q
```

The verifier exits nonzero if a stored result hash, audit status, row count,
group count, aggregate, paired contrast, or route statistic differs from its
registered value. Its JSON output is suitable for continuous-integration or
archive validation.

## Frozen study configuration

- far-region support: `x/H >= 7.5`;
- retained-deficit scale: `1.30`;
- source/target height-ratio exponent: `0.25`;
- post-recovery pressure coefficient: `0.55`;
- pressure normalization: `P90 = 0.897903`;
- warning threshold: `E >= 0.05`.

The retained-lift combination is clipped linear superposition. Unsupported or
invalid terrain geometry uses an identity fallback and is marked outside the
formula domain. The exact branch definitions and parameter values are also
embedded in the immutable result receipts.

## Public source data

- [FuXi-CFD numerical fields](https://doi.org/10.5281/zenodo.18770845)
- [Perdigao ISFS data](https://doi.org/10.26023/ZDMJ-D1TY-FG14)
- [Karim et al. successive-hill PIV data](https://doi.org/10.5281/zenodo.4294745)
- [ESA WorldCover 2020 v100](https://doi.org/10.5281/zenodo.5571936)

Third-party raw fields are not duplicated here. Bolund and RUSHIL values are
derived from their cited benchmark publications and official benchmark files.
The exact redistributed row-level evidence is documented in
[`results/README.md`](results/README.md).

## Reproducibility boundary

The core model and the complete case-to-aggregate calculations are portable
in this repository. The archived research-driver scripts record the original
study workflow, but some expect local data adapters and licensed or separately
downloaded inputs that are not redistributed. They are provenance artifacts,
not a claim of a data-free raw-to-result rerun. See
[`scripts/README.md`](scripts/README.md) and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the precise distinction.

Machine receipts retain historical internal schema strings because changing
them would invalidate their hashes. Those strings are provenance identifiers,
not model names used in the paper.

## License

Unless a directory states otherwise, the original software and documentation
in this repository are licensed under the [MIT License](LICENSE). The original
processed tables, manifests, and audit receipts under `results/` are licensed
under [CC BY 4.0](results/LICENSE). Third-party raw datasets are not
redistributed and remain subject to their source licenses.
