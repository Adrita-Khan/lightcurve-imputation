"""
TS-MICE: Time-Series-Adapted Multiple Imputation by Chained Equations.

Implements Algorithm 5 (TS-MICE) from the thesis.

The design matrix augments the flux vector with:
  - L=5 backward and forward lag columns
  - sin(2π t / P̂) and cos(2π t / P̂) phase columns

where P̂ is the dominant Lomb-Scargle period estimated from the observed cadences.

C_chains=5 independent IterativeImputer chains (Bayesian Ridge, 10 iterations)
are run and averaged.

Requires: scikit-learn, astropy (for LombScargle)
"""

from __future__ import annotations

import logging

import numpy as np

from .base import BaseImputer
from ..utils.period import lomb_scargle_period

logger = logging.getLogger(__name__)


def build_lag_phase_matrix(
    flux: np.ndarray,
    time: np.ndarray,
    lag_order: int = 5,
    period: float | None = None,
) -> np.ndarray:
    """
    Build the (N × (2L+3)) lag-and-phase design matrix used by TS-MICE,
    RF-Impute, and GB-MICE.

    Column order: [f_{i-L}, …, f_{i-1}, f_i, f_{i+1}, …, f_{i+L},
                   sin(2πt/P̂), cos(2πt/P̂)]

    The lag-zero column (f_i) is the imputation target.
    Boundary lag values are filled by LOCF/NOCB.

    Parameters
    ----------
    flux : np.ndarray, shape (N,)
        Flux vector (NaN at missing positions).
    time : np.ndarray, shape (N,)
        Cadence times.
    lag_order : int
        Number of lag/lead steps L (default 5).
    period : float | None
        Pre-computed period; if None, Lomb-Scargle is computed on observed flux.

    Returns
    -------
    X : np.ndarray, shape (N, 2L+3)
        Design matrix. The lag-0 column (index L) is the flux target.
    period_est : float
        Estimated dominant period.
    """
    N = len(flux)
    # Fill-in for boundary lags using LOCF/NOCB
    obs_idx = np.where(np.isfinite(flux))[0]
    flux_filled = flux.copy()
    if len(obs_idx) > 0:
        for i in range(N):
            if not np.isfinite(flux_filled[i]):
                prev = obs_idx[obs_idx < i]
                nxt  = obs_idx[obs_idx > i]
                if len(prev):
                    flux_filled[i] = flux[prev[-1]]
                elif len(nxt):
                    flux_filled[i] = flux[nxt[0]]

    # Estimate period if not provided
    if period is None:
        mask = np.isfinite(flux)
        if mask.sum() > 10:
            period = lomb_scargle_period(time[mask], flux[mask])
        else:
            period = float(np.ptp(time) / 2.0)  # fallback

    # Lag columns
    cols = []
    for lag in range(-lag_order, lag_order + 1):
        shifted = np.roll(flux_filled, -lag)
        if lag > 0:
            shifted[-lag:] = flux_filled[-1]
        elif lag < 0:
            shifted[:-lag] = flux_filled[0]
        cols.append(shifted)

    # Phase columns
    phase = 2.0 * np.pi * time / period
    cols.append(np.sin(phase))
    cols.append(np.cos(phase))

    X = np.column_stack(cols)  # (N, 2L+3)
    return X, period


class TSMICEImputer(BaseImputer):
    """
    Time-Series-Adapted Multiple Imputation by Chained Equations (TS-MICE).

    Parameters
    ----------
    lag_order : int
        Number of lagged predictors L (default 5).
    n_chains : int
        Number of independent MICE chains (default 5).
    n_iter : int
        Iterations per chain (default 10).
    seed : int
        Random seed for first chain; subsequent chains use seed + c.
    """

    def __init__(
        self,
        lag_order: int = 5,
        n_chains: int = 5,
        n_iter: int = 10,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.lag_order = lag_order
        self.n_chains = n_chains
        self.n_iter = n_iter

    def impute(
        self,
        flux: np.ndarray,
        mask: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge

        X, _ = build_lag_phase_matrix(flux, time, lag_order=self.lag_order)
        target_col = self.lag_order  # lag-0 column index

        chain_preds = []
        for c in range(self.n_chains):
            imp = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=self.n_iter,
                sample_posterior=True,
                random_state=self.seed + c,
                initial_strategy="mean",
            )
            X_imp = imp.fit_transform(X)
            chain_preds.append(X_imp[:, target_col])

        # Average across chains
        f_imputed = np.mean(np.stack(chain_preds, axis=0), axis=0)

        out = flux.copy().astype(float)
        out[~mask] = f_imputed[~mask]
        return out

    @property
    def name(self) -> str:
        return "TS_MICE"
