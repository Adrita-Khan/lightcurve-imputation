"""
Evaluation metrics for reconstruction fidelity and classification performance.

Implements:
  - RMSE  (Eq. 3.50)
  - MAE   (Eq. 3.51)
  - Accuracy loss ΔAcc (Eq. 3.53)
  - Period recovery rate PRR (Eq. 3.55)
  - Feature distortion Δφ_k (Eq. 3.56)
  - Bootstrap confidence intervals
  - Friedman and Nemenyi statistical tests
  - Wilcoxon signed-rank test
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EPS_STAB = 1e-8  # numerical stabiliser (Section 3.6.3)


# ---------------------------------------------------------------------------
# Reconstruction metrics
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error at missing cadences (Eq. 3.50)."""
    if len(y_true) == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error at missing cadences (Eq. 3.51)."""
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs(y_pred - y_true)))


# ---------------------------------------------------------------------------
# Classification metric
# ---------------------------------------------------------------------------

def accuracy_loss(acc_baseline: float, acc_imputed: float) -> float:
    """ΔAcc = Acc_baseline − Acc_imputed (Eq. 3.53). Positive = worse."""
    return float(acc_baseline - acc_imputed)


# ---------------------------------------------------------------------------
# Period recovery
# ---------------------------------------------------------------------------

def relative_period_error(p_true: float, p_imp: float) -> float:
    """ε_P = |P̂_true − P̂_imp| / P̂_true  (Eq. 3.54)."""
    if p_true <= 0:
        return np.nan
    return float(abs(p_true - p_imp) / p_true)


def period_recovery_rate(
    p_true_array: np.ndarray,
    p_imp_array: np.ndarray,
    threshold: float = 0.01,
) -> float:
    """
    Fraction of light curves with ε_P < threshold (Eq. 3.55).

    Parameters
    ----------
    p_true_array : np.ndarray
        Ground-truth periods.
    p_imp_array : np.ndarray
        Imputed-curve estimated periods.
    threshold : float
        Recovery threshold (default 1%).

    Returns
    -------
    float in [0, 1]
    """
    eps = np.abs(p_true_array - p_imp_array) / (np.abs(p_true_array) + EPS_STAB)
    return float(np.mean(eps < threshold))


# ---------------------------------------------------------------------------
# Feature distortion
# ---------------------------------------------------------------------------

def feature_distortion(
    phi_imputed: np.ndarray,   # (n_test, n_features)
    phi_true: np.ndarray,      # (n_test, n_features)
    eps: float = EPS_STAB,
) -> np.ndarray:
    """
    Normalised per-feature distortion Δφ_k (Eq. 3.56).

    Parameters
    ----------
    phi_imputed : (n_test, n_features)
    phi_true    : (n_test, n_features)
    eps         : numerical stabiliser

    Returns
    -------
    np.ndarray, shape (n_features,)
        Mean normalised distortion per feature.
    """
    denom = np.abs(phi_true) + eps
    relative_error = np.abs(phi_imputed - phi_true) / denom
    return relative_error.mean(axis=0)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean of `values`.

    Parameters
    ----------
    values : np.ndarray, shape (S,)
        Seed-level metric values.
    n_bootstrap : int
        Number of bootstrap samples (default 1000).
    alpha : float
        Significance level (default 0.05 → 95% CI).
    seed : int
        RNG seed.

    Returns
    -------
    (lower, upper) : tuple of float
    """
    rng = np.random.default_rng(seed)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def friedman_test(
    acc_matrix: np.ndarray,  # (n_seeds, n_methods)
) -> dict:
    """
    Friedman test for equal classification accuracy across methods.

    Parameters
    ----------
    acc_matrix : np.ndarray, shape (S, M)
        S = number of seeds, M = number of methods.

    Returns
    -------
    dict with keys: statistic, p_value
    """
    from scipy.stats import friedmanchisquare

    # scipy.stats.friedmanchisquare takes each method as a separate argument
    cols = [acc_matrix[:, j] for j in range(acc_matrix.shape[1])]
    stat, pval = friedmanchisquare(*cols)
    return {"statistic": float(stat), "p_value": float(pval)}


def nemenyi_test(
    acc_matrix: np.ndarray,  # (n_seeds, n_methods)
    method_names: List[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Nemenyi post-hoc pairwise test.

    Parameters
    ----------
    acc_matrix : (S, M) accuracy matrix
    method_names : list of M method names
    alpha : significance level

    Returns
    -------
    pd.DataFrame of p-values (M × M), indexed by method names.
    """
    try:
        import scikit_posthocs as sp
    except ImportError:
        raise ImportError("scikit-posthocs required: pip install scikit-posthocs")

    df = pd.DataFrame(acc_matrix, columns=method_names)
    p_matrix = sp.posthoc_nemenyi_friedman(df)
    return p_matrix


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
) -> dict:
    """
    Paired Wilcoxon signed-rank test between two methods' seed-level accuracies.

    Parameters
    ----------
    a, b : np.ndarray, shape (S,)
        Seed-level accuracy for method A and method B.

    Returns
    -------
    dict with keys: statistic (W), p_value, effect_size (r_W)
    """
    from scipy.stats import wilcoxon

    try:
        stat, pval = wilcoxon(a, b, alternative="two-sided")
        # Effect size r_W = Z / sqrt(S)
        from scipy.stats import norm
        n = len(a)
        z = norm.isf(pval / 2)  # convert p-value to z approximation
        r_w = z / np.sqrt(n)
        return {
            "statistic": float(stat),
            "p_value":   float(pval),
            "effect_size": float(r_w),
            "n": n,
        }
    except Exception as exc:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size": np.nan, "n": len(a)}


# ---------------------------------------------------------------------------
# Safe missingness threshold p*
# ---------------------------------------------------------------------------

def safe_missingness_threshold(
    acc_loss_at_fractions: Dict[float, float],
    threshold: float = 0.05,
) -> float:
    """
    Linearly interpolate to find the missingness fraction p* at which
    ΔAcc first exceeds `threshold` (default 5 pp).

    Parameters
    ----------
    acc_loss_at_fractions : dict mapping p → ΔAcc
    threshold : float

    Returns
    -------
    float or np.nan if the threshold is never exceeded in [0, max_p].
    """
    fractions = sorted(acc_loss_at_fractions.keys())
    losses    = [acc_loss_at_fractions[f] for f in fractions]

    # Add p=0, ΔAcc=0 as baseline
    fractions = [0.0] + fractions
    losses    = [0.0] + losses

    for i in range(1, len(fractions)):
        if losses[i] >= threshold:
            # Linear interpolation
            p0, p1 = fractions[i - 1], fractions[i]
            l0, l1 = losses[i - 1], losses[i]
            if l1 == l0:
                return float(p0)
            p_star = p0 + (threshold - l0) / (l1 - l0) * (p1 - p0)
            return float(p_star)

    return float("nan")
