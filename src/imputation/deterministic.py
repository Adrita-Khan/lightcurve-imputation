"""
Deterministic / interpolation-based imputation methods.

Implements:
    - Mean-Fill          (Equation 3.2)
    - Forward-Fill LOCF  (Equation 3.3)
    - Linear Interpolation
    - Cubic Spline Interpolation (natural boundary conditions)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

from .base import ImputerBase


class MeanFillImputer(ImputerBase):
    """Replace every missing value with the global mean of observed cadences.

    This is the lower-bound baseline (Equation 3.2 in the thesis).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="Mean-Fill", seed=seed)

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        imputed = flux.copy()
        observed_mask = ~np.isnan(flux)
        global_mean = float(np.mean(flux[observed_mask]))
        imputed[missing_idx] = global_mean
        return imputed


class ForwardFillImputer(ImputerBase):
    """Last Observation Carried Forward (LOCF).

    Equation 3.3 in the thesis.  Remaining leading NaN values (if any)
    are back-filled with the first observed value.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="Forward-Fill", seed=seed)

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        series = pd.Series(flux.copy())
        series = series.ffill().bfill()  # bfill handles leading NaN
        return series.to_numpy(dtype=float)


class LinearInterpImputer(ImputerBase):
    """Piecewise linear interpolation between nearest observed neighbours."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="Linear-Interp", seed=seed)

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        obs_mask = ~np.isnan(flux)
        t_obs = t[obs_mask]
        f_obs = flux[obs_mask]

        # Linear interpolation; fill_value='extrapolate' handles boundary gaps
        interp_fn = interp1d(t_obs, f_obs, kind="linear", fill_value="extrapolate")
        imputed = flux.copy()
        imputed[missing_idx] = interp_fn(t[missing_idx])
        return imputed


class SplineInterpImputer(ImputerBase):
    """Cubic spline interpolation with natural boundary conditions.

    Uses ``scipy.interpolate.CubicSpline`` with ``bc_type='natural'``.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__(name="Spline-Interp", seed=seed)

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        obs_mask = ~np.isnan(flux)
        t_obs = t[obs_mask]
        f_obs = flux[obs_mask]

        cs = CubicSpline(t_obs, f_obs, bc_type="natural", extrapolate=True)
        imputed = flux.copy()
        imputed[missing_idx] = cs(t[missing_idx])
        return imputed
