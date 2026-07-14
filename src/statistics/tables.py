"""Generate publication-ready statistical summary tables."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .tests import wilcoxon_signed_rank, friedman_test, bootstrap_ci, effect_size_cohens_d


def generate_stats_table(
    raw_df: pd.DataFrame,
    metric: str = "rmse",
    fraction: float = 0.30,
    baseline_method: str = "Linear-Interp",
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a descriptive statistics table with CIs and Wilcoxon p-values.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw results from :func:`aggregate_results`.
    metric : str
        Metric column to analyse.
    fraction : float
        Missingness fraction to filter.
    baseline_method : str
        Reference method for pairwise Wilcoxon tests.
    ci : float
        Confidence level for bootstrap intervals.
    seed : int
        Random seed for bootstrap.

    Returns
    -------
    pd.DataFrame
        Table with columns: method, mean, std, ci_lower, ci_upper,
        wilcoxon_stat, wilcoxon_p, cohens_d.
    """
    sub = raw_df[raw_df["fraction"] == fraction]
    methods = sub["method"].unique()
    baseline_vals = sub[sub["method"] == baseline_method][metric].values

    rows = []
    for m in methods:
        vals = sub[sub["method"] == m][metric].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1))
        ci_lo, ci_hi = bootstrap_ci(vals, ci=ci, seed=seed)

        if m == baseline_method or len(vals) < 2:
            w_stat, w_p, d = np.nan, np.nan, 0.0
        else:
            try:
                w_stat, w_p = wilcoxon_signed_rank(baseline_vals[:len(vals)], vals)
                d = effect_size_cohens_d(baseline_vals[:len(vals)], vals)
            except Exception:
                w_stat, w_p, d = np.nan, np.nan, np.nan

        rows.append({
            "method": m,
            "mean": mean_val,
            "std": std_val,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "cohens_d": d,
        })

    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)
