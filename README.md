# lightcurve-imputation

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)


> **Simulation-based evaluation of 13 missing-data imputation methods for astronomical light curves**  
> Accompanying code for the master's thesis *"Missing Data Imputation for Irregular Light
Curves in Astronomy: A Statistical Evaluation of Imputation Methods for Preserving
Periodic Signal Properties"*

---

## Overview

Modern astronomical surveys (Kepler, TESS, ZTF, Rubin LSST) produce light curves for millions of objects, but observational gaps from weather, satellite safe-modes, and detector issues are unavoidable. This repository provides a **fully reproducible, simulation-based benchmark** for comparing 13 imputation methods on synthetic periodic light curves with a known ground truth.

### Key features

- **13 imputation methods** from mean-fill to Transformer-based SAITS
- **No external datasets required** — fully self-contained synthetic pipeline
- **Complete reproducibility** via fixed random seeds
- **Publication-quality figures** and **LaTeX tables** generated automatically
- **Single command** to reproduce every result in the thesis: `make reproduce`

---

## Motivation and research questions

The imputation step is often treated as a minor implementation detail, yet the choice of method measurably affects:

1. **Pointwise fidelity** (RMSE, MAE at missing cadences)
2. **Period recovery** (Lomb–Scargle period accuracy after imputation)
3. **Amplitude and phase preservation** (scientifically critical for classification)
4. **Computational cost** (essential for survey-scale pipelines)

**Central research question:**
> *Which missing-data imputation method best preserves the periodic signal structure of an astronomical light curve, and what fraction of missing observations can be tolerated before significant degradation occurs?*

---

## The 13 imputation methods

| # | Method | Paradigm | Key tool |
|---|--------|----------|----------|
| 1 | Mean-Fill | Deterministic | `SimpleImputer` |
| 2 | Forward-Fill (LOCF) | Deterministic | `DataFrame.ffill` |
| 3 | Linear Interpolation | Interpolation | `scipy.interp1d` |
| 4 | Cubic Spline | Interpolation | `CubicSpline` |
| 5 | GP (Matérn-3/2) | Probabilistic | `george` |
| 6 | TS-MICE | Multiple imputation | `IterativeImputer` |
| 7 | KNN-Impute | Machine learning | `KNNImputer` |
| 8 | RF-Impute | Machine learning | `RandomForestRegressor` |
| 9 | RNN-Impute (BiLSTM) | Deep learning | `PyTorch` |
| 10 | GAIN-Impute | Deep learning (GAN) | `PyTorch` |
| 11 | MF-Impute | Matrix factorisation | `NumPy` ALS |
| 12 | GB-MICE | Machine learning | `XGBoost` |
| 13 | SAITS | Deep learning (Transformer) | `PyTorch` |

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Adrita-Khan/lightcurve-imputation.git
cd lightcurve-imputation
```

### 2. Create and activate the conda environment

```bash
conda create -n lc_imputation python=3.10
conda activate lc_imputation
pip install -r requirements.txt
pip install -e .
```

### 3. Reproduce all results

```bash
make reproduce
# or equivalently:
python run_all.py
```

This single command:
1. Generates the synthetic light curve (N=1323, Kepler-cadence)
2. Injects MCAR gaps at 10%, 30%, 50% (30 realisations each)
3. Applies all 13 imputation methods
4. Computes RMSE, MAE, period recovery, amplitude error, phase error, runtime
5. Runs Wilcoxon, Friedman, and Nemenyi statistical tests
6. Saves all figures to `figures/` and tables to `tables/`

### 4. Fast smoke test (CI/reduced settings)

```bash
make fast-test
# or:
python run_all.py --config configs/fast_test.yml
```

---

## Repository structure

```
thesis-lightcurve-imputation/
│
├── README.md               ← This file
├── LICENSE                 ← MIT licence
├── CITATION.cff            ← Machine-readable citation
├── .gitignore
├── pyproject.toml          ← Package metadata (PEP 621)
├── requirements.txt        ← Pinned dependencies
├── environment.yml         ← Conda environment
├── setup.py                ← Backward-compat shim
├── Makefile                ← Convenience commands
├── Dockerfile              ← Container for full reproducibility
├── docker-compose.yml
├── run_all.py              ← Single entry point
│
├── configs/
│   ├── experiment.yml      ← Full experiment (30 seeds × 3 fractions)
│   └── fast_test.yml       ← Reduced settings for CI
│
├── src/
│   ├── simulation/         ← Synthetic light-curve generator
│   ├── missingness/        ← MCAR gap injection
│   ├── imputation/         ← All 13 imputation methods + registry
│   ├── evaluation/         ← Metrics + result aggregation
│   ├── statistics/         ← Wilcoxon, Friedman, Nemenyi, bootstrap CI
│   ├── visualization/      ← Publication-quality figure generation
│   ├── utils/              ← Config, I/O, logging, seeds
│   └── pipeline/           ← Experiment orchestration
│
├── scripts/                ← Standalone utility scripts
├── notebooks/              ← Jupyter demonstration notebooks
├── tests/                  ← pytest unit tests (≥80% coverage target)
├── docs/                   ← Extended documentation
├── figures/                ← Generated figures (gitignored)
├── tables/                 ← Generated tables (gitignored)
└── data/
    ├── raw/                ← (empty; no external data needed)
    ├── simulated/          ← Generated signals
    ├── processed/          ← Intermediate processed data
    └── results/            ← raw_results.csv, summary_results.csv
```

---

## Configuration

All parameters are in `configs/experiment.yml` — **no hard-coded values** exist in the code.

Key parameters:

```yaml
signal:
  N: 1323        # Cadences (Kepler Q1 long-cadence)
  dt: 0.0204     # Days per cadence (~29.4 min)
  A: 0.1         # Amplitude (normalised flux)
  P0: 1.0        # True period (days)
  sigma_eps: 0.02  # Gaussian noise level

missingness:
  fractions: [0.10, 0.30, 0.50]
  n_seeds: 30

methods:
  gp_matern: true
  saits: true
  # ... all 13 methods toggleable
```

---

## Python API

```python
from src.simulation.generator import generate_synthetic_lightcurve
from src.missingness.injector import inject_gaps
from src.imputation import GPMaternImputer, LinearInterpImputer
from src.evaluation.metrics import evaluate_imputation

# Generate signal
t, flux, params = generate_synthetic_lightcurve(N=1323, P0=1.0, seed=0)

# Inject 30% MCAR gaps
gapped, missing_idx, true_vals = inject_gaps(flux, p=0.30, seed=42)

# Impute
imp = LinearInterpImputer()
imputed = imp.fit_impute(t, gapped, missing_idx, period_est=1.0)

# Evaluate
metrics = evaluate_imputation(
    t=t, flux_imputed=imputed, missing_idx=missing_idx,
    true_vals=true_vals, true_period=1.0, true_amplitude=0.1, true_phase=0.0
)
print(f"RMSE: {metrics['rmse']:.4f}, PRR: {metrics['period_recovered']}")
```

---

## Docker (fully isolated)

```bash
make docker-build
make docker-run
```

Results are written to the mounted host directories `data/results/`, `figures/`, `tables/`.

---

## Running tests

```bash
make test
# or:
pytest tests/ -v --cov=src --cov-report=html
```

Open `htmlcov/index.html` to view coverage report.

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{khan2026repo,
  author  = {Khan, Adrita},
  title   = {Computational Aspects of Cosmology: Imputation Methods for
             Astronomical Light Curves with Missing Data},
  school  = {Your University},
  year    = {2026},
  url     = {https://github.com/Adrita-Khan/lightcurve-imputation}
}
```

---

## References

- Borucki et al. (2010) — Kepler mission
- Ricker et al. (2015) — TESS mission
- Bellm et al. (2019) — ZTF
- Ivezić et al. (2019) — LSST/Rubin
- VanderPlas (2018) — Lomb–Scargle periodogram
- Che et al. (2018) — GRU-D
- Yoon et al. (2018) — GAIN
- Du et al. (2023) — SAITS
- Chen & Guestrin (2016) — XGBoost

---

## Licence

MIT © 2026 Adrita Khan
