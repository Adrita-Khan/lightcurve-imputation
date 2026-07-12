# Experimental Workflow

This document walks through the complete five-stage pipeline from data acquisition to figure generation.

---

## Overview

```
Raw Kepler PDCSAP flux (MAST)
        │
        ▼
  1. Preprocessing (Algorithm 1)
        │  quarter concatenation, outlier rejection,
        │  quality masking, median normalisation
        ▼
  Clean labelled dataset D
        │
        ├──────────────────────────────────────────────┐
        │ D_train (70%)                                 │ D_test (30%)
        ▼                                               ▼
  Feature extraction (gap-free)           2. Inject gaps (Algorithm 2)
        │                                      │  p ∈ {10%, 30%, 50%}
        ▼                                      │  MCAR + block mixture
  Train RF classifier (Algorithm 6)            │  30 seeds per (lc, p)
        │ 500 trees, Gini, balanced             ▼
        │ Frozen after training        3. Impute (Algorithms 3–12)
        ▼                                      │  13 methods
  Gap-free baseline accuracy                   ▼
                                       4. Feature extraction (Algorithm 5)
                                              │  35-D vector φ_j
                                              ▼
                                       5. Frozen classifier evaluation
                                              │
                                              ▼
                                   RMSE, MAE, ΔAcc, PRR, Δφ_k
```

---

## Stage 1: Data Download and Preprocessing

```bash
# Download all labelled Kepler light curves (~3,006 targets)
python scripts/download_data.py --config configs/experiment.yaml

# For a quick test with 50 targets per class:
python scripts/download_data.py --max_per_class 50
```

This produces one `.parquet` file per target in `data/processed/`, each containing:
- `time` — BKJD timestamps
- `flux` — median-normalised PDCSAP flux
- `class_label` — integer class label (0–5)

---

## Stage 2: Run the Experiment

```bash
# Full experiment (all 13 methods × 3 fractions × 30 seeds)
python scripts/run_experiment.py --config configs/experiment.yaml

# Quick test with a subset:
python scripts/run_experiment.py \
    --methods GP_Matern32 Linear Mean_Fill \
    --fractions 0.3 \
    --n_seeds 5
```

Outputs saved to `results/tables/`:
- `experiment_results.csv` — main results table (one row per method × fraction)
- `baseline_meta.json` — gap-free baseline accuracy and metadata
- `seed_level_accuracy.csv` — per-seed accuracies (if `--save_seeds` flag used)

---

## Stage 3: Statistical Tests

```bash
python scripts/run_statistical_tests.py --results_dir results
```

Outputs:
- `results/tables/table_friedman.csv`
- `results/tables/table_nemenyi_30pct.csv`
- `results/tables/table_wilcoxon_gp_vs_spline.csv`

---

## Stage 4: Generate Figures and Tables

```bash
python scripts/generate_figures.py --results_dir results
```

Outputs saved to `results/figures/` and `results/tables/`:
- `figure_accuracy_vs_fraction.pdf` — accuracy loss vs missingness fraction
- `figure_rmse_heatmap.pdf` — RMSE heatmap (methods × fractions)
- `figure_lightcurve_pipeline_demo.pdf` — Figure 1.1 reproduction
- `figure_feature_distortion.pdf` — top-15 feature distortion bar chart
- `table_reconstruction.csv` — Table 4.1 (RMSE/MAE)
- `table_period_recovery.csv` — Table 4.2 (PRR)
- `table_classification.csv` — Table 4.3 (accuracy + 95% CIs)
- `table_safe_thresholds.csv` — Table 4.9 (p*)

---

## Notebooks

For interactive exploration:

```bash
cd notebooks
jupyter lab
```

| Notebook | Covers |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Chapter 3 EDA |
| `02_imputation_methods_demo.ipynb` | Visual comparison of all 13 methods on one light curve |
| `03_results_analysis.ipynb` | Chapter 4 results — tables and figures |

---

## Key Design Decisions

### No train/test leakage
All 13 imputers are verified leakage-free:
- Each ML imputer trains only on the **observed cadences of the current light curve**
- The downstream RF classifier is trained once on **gap-free features** and frozen
- Withheld ground-truth values `{f_i*}` are used **only for evaluation** (RMSE, MAE)

### Reproducibility
- All random seeds are fixed and recorded in `configs/experiment.yaml`
- Gap realisations are generated deterministically: seed `s = base_seed + realisation_index`
- Results are fully reproducible given the same software versions (see `requirements.txt`)

### Computational cost
Approximate wall-clock times per light curve (single CPU core, 4,000 cadences):

| Method | Approx. time |
|---|---|
| Mean/Forward/Linear/Spline | < 0.01 s |
| KNN-Impute | ~ 0.1 s |
| RF-Impute | ~ 0.5 s |
| TS-MICE | ~ 2–5 s |
| MF-Impute | ~ 1–3 s |
| GB-MICE | ~ 5–15 s |
| GP (Matérn-3/2) | ~ 10–30 s |
| RNN-Impute | ~ 5–20 s |
| GAIN-Impute | ~ 10–30 s |
| SAITS | ~ 10–30 s |

For the full dataset (3,006 light curves × 3 fractions × 30 seeds), the GP and deep-learning methods are the computational bottleneck. Parallelism across light curves is supported via `joblib`.
