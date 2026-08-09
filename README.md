# URSA orographic-lift adapter

URSA (Upstream-Ridge Sheltering Attenuation) is a deterministic, physics-guided adapter for reducing unsupported positive orographic lift behind an upstream ridge. It is not a machine-learning model. The inputs are a digital elevation model (DEM), one representative horizontal wind vector, aerodynamic roughness, query height above ground, and a signed vertical-wind field from a compatible engineering estimator.

The adapter returns:

- `S`: retained-positive-lift factor;
- `E`: continuous separation/recovery exposure metadata;
- a configurable warning (`E >= 0.05` in the paper);
- an applicability flag;
- the corrected field `min(w0, 0) + S * max(w0, 0)`.

## Repository contents

- `src/ursa/`: frozen terrain, ridge-geometry, shear-layer, cavity, and relaxing-wake primitives;
- `scripts/`: versioned development, validation, and route-evaluation scripts used in the study;
- `results/`: compact processed receipts that can be redistributed (added with the archived release).

The journal manuscript and Supplementary Information are intentionally not distributed in this pre-submission code repository. The source code records the frozen implementation, including raster edge handling and branch definitions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core modules require Python 3.10+, NumPy, and SciPy. Full figure and route reproduction also requires the downloaded third-party datasets and the processed manifests described in the paper.

## Public source data

- FuXi-CFD numerical fields: https://doi.org/10.5281/zenodo.18770845
- Perdigão ISFS data: https://doi.org/10.26023/ZDMJ-D1TY-FG14
- Karim et al. successive-hill PIV data: https://doi.org/10.5281/zenodo.4294745
- ESA WorldCover 2020 v100: https://doi.org/10.5281/zenodo.5571936

Third-party raw data are not redistributed in this repository. Bolund and RUSHIL values are derived from the cited benchmark publications and official benchmark files.

## Frozen study configuration

- far-region support: `x / H >= 7.5`;
- retained-deficit scale: `1.30`;
- source/target height-ratio exponent: `0.25`;
- post-recovery pressure coefficient: `0.55`;
- pressure normalization: `P90 = 0.897903`;
- warning threshold used in the engineering evaluation: `E >= 0.05`.

URSA preserves negative input vertical velocity and modifies only the positive branch. Unsupported or invalid ridge geometry returns the identity adapter and a false applicability flag.

## Reproducibility status

This repository is the public code home for the study. Manuscript and Supplementary Information sources are excluded during pre-submission development. Compact result tables, checksums, one-command examples, and an archived release identifier will be added with the publication release; no scientific coefficient will be refitted during packaging.
