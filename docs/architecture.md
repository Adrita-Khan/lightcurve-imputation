# Repository Architecture

## Package structure

```
src/
├── simulation/         Synthetic light-curve generation
│   └── generator.py   generate_synthetic_lightcurve()
│
├── missingness/        MCAR gap injection
│   └── injector.py    inject_gaps()
│
├── imputation/         All 13 imputation methods
│   ├── base.py        ImputerBase (abstract)
│   ├── deterministic.py  Mean, ForwardFill, Linear, Spline
│   ├── gp_imputer.py  GP Matérn-3/2 (george)
│   ├── ts_mice.py     TS-MICE (sklearn IterativeImputer)
│   ├── knn_imputer.py KNN in lag-feature space
│   ├── rf_imputer.py  Random Forest
│   ├── rnn_imputer.py BiLSTM (PyTorch)
│   ├── gain_imputer.py GAIN GAN (PyTorch)
│   ├── mf_imputer.py  Hankel ALS matrix factorisation
│   ├── gb_mice.py     XGBoost MICE
│   ├── saits_imputer.py SAITS Transformer (PyTorch)
│   └── registry.py    get_imputer() / list_imputers()
│
├── evaluation/         Metrics and aggregation
│   ├── metrics.py     RMSE, MAE, PRR, ε_A, ε_φ, runtime
│   └── aggregator.py  aggregate_results(), summarise_results()
│
├── statistics/         Hypothesis testing
│   ├── tests.py       Wilcoxon, Friedman, Nemenyi, bootstrap CI
│   └── tables.py      generate_stats_table()
│
├── visualization/      Figure generation
│   └── plots.py       All publication figures
│
├── utils/              Shared utilities
│   ├── config.py      load_config(), merge_configs()
│   ├── io.py          save_results(), save_table()
│   ├── logging_utils.py get_logger()
│   └── seeds.py       make_seed_sequence()
│
└── pipeline/           Orchestration
    └── runner.py      ExperimentPipeline, run_experiment()
```

## Data flow

```
configs/experiment.yml
        │
        ▼
src/simulation/generator.py  →  t, flux (ground truth)
        │
        ▼
src/missingness/injector.py  →  gapped, missing_idx, true_vals
        │
        ▼
src/imputation/<method>.py   →  flux_imputed
        │
        ▼
src/evaluation/metrics.py    →  {rmse, mae, prr, amp_err, phase_err, runtime}
        │
        ▼
src/evaluation/aggregator.py →  raw_df, summary_df
        │
        ├──► src/statistics/  →  Wilcoxon, Friedman, tables/
        └──► src/visualization/ →  figures/
```

## Imputer API

All imputers inherit from `ImputerBase` and implement:

```python
imputer.impute(t, flux, missing_idx, period_est) -> np.ndarray
imputer.fit_impute(t, flux, missing_idx, period_est) -> np.ndarray  # records timing
imputer.runtime_s   # seconds
imputer.memory_mb   # MiB
```

Registry dispatch:

```python
from src.imputation.registry import get_imputer
imp = get_imputer("gp_matern", params={"n_restarts": 20}, seed=42)
imputed = imp.fit_impute(t, gapped, missing_idx, period_est=1.0)
```
