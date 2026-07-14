"""
RF-Impute: Random Forest flux imputation.

A Random Forest is fitted on lag-and-phase design matrix rows where the
flux is observed; missing cadences are filled by prediction.

Implements Algorithm 4 (RF-Impute) from the thesis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import ImputerBase
from .ts_mice import _build_feature_matrix


class RFImputer(ImputerBase):
    """Random Forest imputation using lag and phase features.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the Random Forest.
    L : int
        Lag order for the design matrix.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        L: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="RF-Impute", seed=seed)
        self.n_estimators = n_estimators
        self.L = L

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

        obs_mask = ~np.isnan(flux)
        # Rows where both the target (column 0) and all features are finite
        row_finite = np.all(np.isfinite(X), axis=1) & obs_mask
        X_train = X[row_finite, 1:]   # features (exclude lag-0 target column)
        y_train = flux[row_finite]

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        imputed = flux.copy()
        # For missing rows, replace NaN features with column medians before predicting
        X_miss = X[missing_idx, 1:].copy()
        col_medians = np.nanmedian(X[:, 1:], axis=0)
        nan_locs = np.isnan(X_miss)
        for col_j in range(X_miss.shape[1]):
            X_miss[nan_locs[:, col_j], col_j] = col_medians[col_j]

        imputed[missing_idx] = rf.predict(X_miss)
        return imputed
