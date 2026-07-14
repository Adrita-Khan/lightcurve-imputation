"""
GB-MICE: Gradient Boosting MICE chained equations.

Replaces TS-MICE's Bayesian Ridge with an XGBoost estimator inside
sklearn's IterativeImputer.  Multiple chains are averaged.

Implements the GB-MICE method described in Section 3.4 of the thesis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from .base import ImputerBase
from .ts_mice import _build_feature_matrix

try:
    from xgboost import XGBRegressor

    _XGB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _XGB_AVAILABLE = False


class GBMICEImputer(ImputerBase):
    """Gradient Boosting MICE imputation.

    Parameters
    ----------
    n_estimators : int
        XGBoost trees per estimator.
    max_depth : int
        Maximum tree depth.
    L : int
        Lag order for design matrix.
    n_chains : int
        Number of independent MICE chains.
    max_iter : int
        MICE iterations per chain.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        L: int = 5,
        n_chains: int = 5,
        max_iter: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required for GBMICEImputer. pip install xgboost==1.7.6")
        super().__init__(name="GB-MICE", seed=seed)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
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
        if period_est is None or period_est <= 0:
            period_est = float(t[-1] - t[0]) / 2.0

        X = _build_feature_matrix(flux, t, period_est, L=self.L)
        rng = np.random.default_rng(self.seed)
        chain_results = []

        for c in range(self.n_chains):
            chain_seed = int(rng.integers(0, 2**31))
            est = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=chain_seed,
                verbosity=0,
                n_jobs=1,
            )
            imputer = IterativeImputer(
                estimator=est,
                max_iter=self.max_iter,
                random_state=chain_seed,
            )
            X_imp = imputer.fit_transform(X)
            chain_results.append(X_imp[:, 0])

        imputed = flux.copy()
        mean_chain = np.nanmean(np.stack(chain_results, axis=1), axis=1)
        imputed[missing_idx] = mean_chain[missing_idx]
        return imputed
