"""
Result aggregation: collect per-seed metric dicts into summary DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_results(results: list[dict]) -> pd.DataFrame:
    """Convert a list of per-seed result dicts into a flat DataFrame.

    Parameters
    ----------
    results : list of dict
        Each dict must contain keys: ``method``, ``fraction``, ``seed``,
        and the metric keys from :func:`evaluate_imputation`.

    Returns
    -------
    pd.DataFrame
        One row per (method, fraction, seed) combination.
    """
    return pd.DataFrame(results)


def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± std of each metric grouped by (method, fraction).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`aggregate_results`.

    Returns
    -------
    pd.DataFrame
        MultiIndex on (method, fraction) with mean and std columns.
    """
    metric_cols = [
        "rmse", "mae", "mse", "relative_error",
        "period_recovered", "period_rel_err",
        "amplitude_error", "phase_error",
        "runtime_s", "memory_mb",
    ]
    available = [c for c in metric_cols if c in df.columns]

    grouped = df.groupby(["method", "fraction"])[available]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std().add_suffix("_std")

    summary = pd.concat([mean_df, std_df], axis=1)

    # PRR: fraction of seeds that recovered the period (period_recovered == 1)
    if "period_recovered" in available:
        prr_series = df.groupby(["method", "fraction"])["period_recovered"].mean().rename("prr")
        summary = pd.concat([summary, prr_series], axis=1)

    return summary.reset_index()
