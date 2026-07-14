# Methodology

## Signal model

The ground-truth light curve is generated as:

```
f_i = μ₀ + A sin(2π t_i / P₀ + φ₀) + ε_i
```

with parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 1323 | Cadence count (Kepler Q1) |
| Δt | 0.0204 d | Cadence (~29.4 min) |
| A | 0.1 | Amplitude |
| P₀ | 1.0 d | True period |
| φ₀ | 0.0 | Initial phase |
| μ₀ | 1.0 | Baseline flux |
| σ_ε | 0.02 | Gaussian noise std |

## Gap injection (MCAR)

Gaps are injected as a 50/50 mixture of:

- **Block gaps**: contiguous runs of 5–50 cadences (satellite outages)
- **Scattered gaps**: individual random cadences (cosmic rays, telemetry dropouts)

At three target fractions: **10%, 30%, 50%**, with **S = 30** independent realisations each.

## Evaluation metrics

| Metric | Equation |
|--------|----------|
| RMSE | `sqrt(mean((f̂_i - f*_i)²))` at missing positions |
| MAE | `mean(|f̂_i - f*_i|)` at missing positions |
| PRR | fraction of seeds with `|P̂ - P₀| / P₀ < 1%` |
| ε_A | `|Â - A| / (A + ε_stab)` |
| ε_φ | phase offset normalised to period fraction |

## Statistical comparison

1. **Wilcoxon signed-rank test**: pairwise comparison against a baseline (across 30 seeds)
2. **Friedman test**: multi-method non-parametric ANOVA
3. **Nemenyi post-hoc**: critical difference analysis
4. **Bootstrap 95% CI**: non-parametric confidence intervals on all metrics
