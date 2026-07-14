"""
TS-MICE: Time-Series Multiple Imputation by Chained Equations.

Uses lagged flux values and phase features (sin/cos of the estimated period)
as predictors for a Bayesian Ridge regression in sklearn's IterativeImputer.
Multiple chains are averaged for a stable point estimate.

Implements Algorithm in Section 3.4 (TS-MICE) of the thesis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from .base import ImputerBase


class TSMICEImputer(ImputerBase):
    """TS-MICE: Time-Series MICE with lag and phase features.

    Parameters
    ----------
    L : int
        Number of lag/lead features on each side.
    n_chains : int
        Number of independent MICE chains to average.
    max_iter : int
        Maximum MICE iterations per chain.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        L: int = 5,
        n_chains: int = 5,
        max_iter: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="TS-MICE", seed=seed)
        self.L = L
        self.n_chains = n_chains
        self.max_iter = max_iter

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        N = len(flux)
        if period_est is None or period_est <= 0:
            period_est = float(t[-1] - t[0]) / 2.0  # rough fallback

        # Build design matrix: lagged columns + phase features
        X = _build_feature_matrix(flux, t, period_est, L=self.L)

        chain_results = []
        rng = np.random.default_rng(self.seed)
        for c in range(self.n_chains):
            chain_seed = int(rng.integers(0, 2**31))
            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.max_iter,
                random_state=chain_seed,
                min_value=-np.inf,
                max_value=np.inf,
            )
            X_imp = imputer.fit_transform(X)
            chain_results.append(X_imp[:, 0])  # lag-0 column is the flux

        imputed = flux.copy()
        mean_chain = np.nanmean(np.stack(chain_results, axis=1), axis=1)
        imputed[missing_idx] = mean_chain[missing_idx]
        return imputed


def _build_feature_matrix(
    flux: np.ndarray, t: np.ndarray, period_est: float, L: int
) -> np.ndarray:
    """Build lag-and-phase design matrix (Equation 3.5 in thesis)."""
    N = len(flux)
    cols = [flux.copy()]  # lag 0

    for lag in range(1, L + 1):
        lagged = np.full(N, np.nan)
        lagged[lag:] = flux[:-lag]
        cols.append(lagged)

        led = np.full(N, np.nan)
        led[:-lag] = flux[lag:]
        cols.append(led)

    # Phase features
    phase_sin = np.sin(2.0 * np.pi * t / period_est)
    phase_cos = np.cos(2.0 * np.pi * t / period_est)
    cols.append(phase_sin)
    cols.append(phase_cos)

    return np.column_stack(cols)
