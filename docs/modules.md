# Module Documentation

This document describes every Python module in the repository and its role in the experimental pipeline.

---

## `src/data/`

### `download.py`
Handles data acquisition from MAST via `lightkurve`.

| Function | Description |
|---|---|
| `build_label_catalogue(cache_dir)` | Merges the three Kepler catalogues (Kirk 2016, Nemec 2013, Debosscher 2011) into a single `[KIC, class_name, class_label]` DataFrame. Results are cached as `kepler_labels.csv`. |
| `download_light_curve(kic, raw_dir, ...)` | Downloads all available Kepler quarters for a given KIC target via `lightkurve`, applies a completeness filter, and saves as `.parquet`. |
| `load_light_curve(path)` | Loads a saved `.parquet` light curve file. |

### `preprocessing.py`
Implements **Algorithm 1** (Light-Curve Preprocessing and Normalisation).

| Function | Description |
|---|---|
| `preprocess_light_curve(df, ...)` | Full five-step pipeline: concatenation, inter-quarter offset correction, 5σ outlier rejection, quality-flag masking, median normalisation. |
| `compute_completeness(df)` | Returns the fraction of finite-flux cadences. |
| `truncate_to_window(df, window_size, strategy)` | Selects a contiguous window of `window_size` cadences. |

### `gap_injection.py`
Implements **Algorithm 2** (MCAR Gap Injection with Block/Scattered Mixture).

| Function | Description |
|---|---|
| `inject_gaps(flux, p, seed, ...)` | Injects synthetic gaps into a complete flux vector. Returns `(gapped_flux, mask, ground_truth)`. |
| `generate_all_seeds(flux, p, n_seeds, ...)` | Generator yielding `n_seeds` independent gap realisations. |

---

## `src/imputation/`

### `base.py`
Abstract base class `BaseImputer`. All imputers share the interface:
```python
imputer.impute(flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray
```

### `classical.py`
Implements **Algorithm 3** — four deterministic/interpolation methods:
- `MeanFillImputer` — global mean substitution (Eq. 3.1)
- `ForwardFillImputer` — LOCF with back-fill fallback (Eq. 3.2)
- `LinearImputer` — piecewise linear interpolation (Eq. 3.3)
- `SplineImputer` — natural cubic spline via `scipy.interpolate.CubicSpline` (Eq. 3.4)

### `gp_imputer.py`
Implements **Algorithm 4** — Gaussian Process regression with Matérn-3/2 kernel.
- `GPMatern32Imputer` — hyperparameters optimised per light curve via L-BFGS-B with 20 random restarts (Eqs. 3.5–3.13)

### `ts_mice.py`
Implements **Algorithm 5** — TS-MICE.
- `TSMICEImputer` — 5 independent Bayesian Ridge MICE chains, 10 iterations, lag-order 5 (Eqs. 3.14–3.15)
- `build_lag_phase_matrix(flux, time, lag_order, period)` — shared design matrix builder used by TS-MICE, RF-Impute, and GB-MICE

### `ml_imputers.py`
Implements **Algorithms 6–12** — the seven ML-based imputers:

| Class | Algorithm | Key equations |
|---|---|---|
| `KNNImputer` | KNN-Impute (Alg. 6) | Eqs. 3.16–3.18 |
| `RFImputer` | RF-Impute (Alg. 7) | Eqs. 3.19–3.20 |
| `RNNImputer` | RNN-Impute / BiLSTM (Alg. 8) | Eqs. 3.21–3.25 |
| `GAINImputer` | GAIN-Impute (Alg. 9) | Eqs. 3.26–3.31 |
| `MFImputer` | MF-Impute / Hankel ALS (Alg. 10) | Eqs. 3.33–3.38 |
| `GBMICEImputer` | GB-MICE (Alg. 11) | Eqs. 3.39–3.41 |
| `SAITSImputer` | SAITS / DMSA (Alg. 12) | Eqs. 3.42–3.49 |

### `registry.py`
Factory module providing:
- `get_imputer(name, seed, **kwargs)` — instantiate by string name
- `get_all_imputers(seed, config)` — instantiate all 13 imputers at once
- `ALL_METHOD_NAMES` — canonical list of 13 method names

---

## `src/features/`

### `extraction.py`
Implements **Algorithm 5** — 35-dimensional feature extraction.

| Function | Description |
|---|---|
| `extract_features(time, flux, flux_err, acf_lags)` | Extracts the 35-D feature vector φ_j from a light curve. Returns `np.ndarray` of shape `(35,)`. |
| `FEATURE_NAMES` | List of 35 canonical feature name strings. |

Feature groups:
1. Period features (4): LS period, LS power, phase amplitude, phase skewness
2. Statistical moments (5): weighted mean, std, skewness, kurtosis, MAD
3. Variability indices (5): Stetson J, K, L; η; r_cs
4. Flux percentile ratios (3): F₅/₉₅, F₁₀/₉₀, F₂₅/₇₅
5. Autocorrelation at lags {1,2,3,5,10,20} (6)
6. Additional period features (6)
7. Additional variability features (6)

---

## `src/classification/`

### `classifier.py`
Implements **Algorithm 6** — fixed-classifier train/test protocol.

| Function | Description |
|---|---|
| `train_classifier(X_train, y_train, ...)` | Trains the RF classifier (500 trees, Gini, balanced) on gap-free features. Saves `.pkl` model. |
| `evaluate_classifier(clf, X_test, y_test, ...)` | Evaluates a frozen classifier. Returns accuracy, macro-F1, MCC, confusion matrix. |
| `train_svm_classifier(X_train, y_train, ...)` | Trains the secondary SVM classifier for sensitivity analysis. |
| `compute_feature_importance(clf, feature_names)` | Returns sorted RF feature importance `pd.Series`. |

---

## `src/evaluation/`

### `metrics.py`
All evaluation metrics from Section 3.6 of the thesis.

| Function | Equation |
|---|---|
| `rmse(y_true, y_pred)` | Eq. 3.50 |
| `mae(y_true, y_pred)` | Eq. 3.51 |
| `accuracy_loss(acc_baseline, acc_imputed)` | Eq. 3.53 |
| `relative_period_error(p_true, p_imp)` | Eq. 3.54 |
| `period_recovery_rate(p_true_array, p_imp_array, threshold)` | Eq. 3.55 |
| `feature_distortion(phi_imputed, phi_true, eps)` | Eq. 3.56 |
| `bootstrap_ci(values, n_bootstrap, alpha, seed)` | 1000-sample bootstrap 95% CI |
| `friedman_test(acc_matrix)` | Friedman non-parametric test |
| `nemenyi_test(acc_matrix, method_names, alpha)` | Nemenyi post-hoc pairwise test |
| `wilcoxon_test(a, b)` | Paired Wilcoxon signed-rank test |
| `safe_missingness_threshold(acc_loss_at_fractions, threshold)` | Linear interpolation for p* |

---

## `src/utils/`

### `period.py`
Lomb-Scargle period estimation utilities shared across modules.

| Function | Description |
|---|---|
| `lomb_scargle_period(time, flux, ...)` | Returns dominant period using `astropy.timeseries.LombScargle`. |
| `lomb_scargle_peak_power(time, flux, ...)` | Returns `(period, peak_power)` tuple. |

---

## `scripts/`

| Script | Description |
|---|---|
| `download_data.py` | Download and preprocess all Kepler light curves (run first). |
| `run_experiment.py` | Main experiment runner — reproduces all Chapter 4 results. |
| `generate_figures.py` | Generate all thesis figures and tables from saved results. |
| `run_statistical_tests.py` | Run Friedman, Nemenyi, and Wilcoxon significance tests. |

---

## `notebooks/`

| Notebook | Description |
|---|---|
| `01_exploratory_data_analysis.ipynb` | Chapter 3 EDA: class distributions, example light curves, period histograms, feature correlations. |
| `02_imputation_methods_demo.ipynb` | Interactive comparison of all 13 imputers on a single light curve. |
| `03_results_analysis.ipynb` | Chapter 4 results: load `experiment_results.csv`, reproduce all tables and figures interactively. |

---

## `configs/`

| File | Description |
|---|---|
| `experiment.yaml` | Master configuration: data settings, gap injection parameters, imputer hyperparameters, classifier settings, evaluation parameters, paths. |
