"""
Statistical comparison methods for imputation benchmark results.

Implements:
- Wilcoxon signed-rank test (pairwise)
- Friedman test (multi-method)
- Nemenyi post-hoc test (critical differences)
- Bootstrap confidence intervals
- Cohen's d effect size
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    import scikit_posthocs as sp

    _SP_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SP_AVAILABLE = False


def wilcoxon_signed_rank(
    x: np.ndarray, y: np.ndarray, alternative: str = "two-sided"
) -> tuple[float, float]:
    """Wilcoxon signed-rank test between two paired metric vectors.

    Parameters
    ----------
    x, y : np.ndarray
        Paired samples (e.g. RMSE of method A and B across 30 seeds).
    alternative : str
        ``'two-sided'``, ``'less'``, or ``'greater'``.

    Returns
    -------
    statistic : float
    p_value : float
    """
    result = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def friedman_test(data: np.ndarray) -> tuple[float, float]:
    """Friedman test for differences across K ≥ 3 methods.

    Parameters
    ----------
    data : np.ndarray, shape (n_seeds, n_methods)
        Each column is one method's metric values across seeds.

    Returns
    -------
    statistic : float
    p_value : float
    """
    result = stats.friedmanchisquare(*[data[:, j] for j in range(data.shape[1])])
    return float(result.statistic), float(result.pvalue)


def nemenyi_posthoc(
    data: np.ndarray, method_names: Optional[list[str]] = None
) -> pd.DataFrame:
    """Nemenyi post-hoc test following a significant Friedman test.

    Parameters
    ----------
    data : np.ndarray, shape (n_seeds, n_methods)
    method_names : list of str or None

    Returns
    -------
    pd.DataFrame
        p-value matrix indexed by method names.
    """
    if not _SP_AVAILABLE:
        raise ImportError(
            "scikit-posthocs is required for the Nemenyi test. "
            "pip install scikit-posthocs"
        )
    df = pd.DataFrame(data, columns=method_names)
    return sp.posthoc_nemenyi_friedman(df)


def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: Optional[int] = None,
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval.

    Parameters
    ----------
    values : np.ndarray
        Sample to bootstrap.
    stat_fn : callable
        Statistic to bootstrap (default: mean).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level.
    seed : int or None
        Random seed.

    Returns
    -------
    lower : float
    upper : float
    """
    rng = np.random.default_rng(seed)
    boot_stats = np.array(
        [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1.0 - alpha)))
    return lower, upper


def effect_size_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size between two independent samples.

    Uses pooled standard deviation.
    """
    n1, n2 = len(x), len(y)
    m1, m2 = np.mean(x), np.mean(y)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return float((m1 - m2) / (s_pooled + 1e-10))
