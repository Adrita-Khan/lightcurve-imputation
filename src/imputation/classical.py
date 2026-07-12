"""
Classical (deterministic / interpolation-based) imputation methods.

Implements Algorithm 3 (Deterministic and Interpolation-Based Imputation)
from the thesis. Four methods are provided:

  MeanFillImputer       — global mean of observed cadences
  ForwardFillImputer    — LOCF (last observation carried forward)
  LinearImputer         — piecewise linear interpolation
  SplineImputer         — natural cubic spline (scipy.interpolate.CubicSpline)

All methods share the same interface via BaseImputer.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from .base import BaseImputer


class MeanFillImputer(BaseImputer):
    """
    Mean-fill: replace every missing cadence with the global observed mean.

    Equation (3.1): f̂_i = f̄_O for all i ∈ M.

    This is the lower-bound baseline — unbiased under MCAR but suppresses
    all amplitude-related variability features.
    """

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        out = flux.copy().astype(float)
        obs_mean = float(np.nanmean(flux[mask]))
        out[~mask] = obs_mean
        return out


class ForwardFillImputer(BaseImputer):
    """
    Forward-fill (LOCF): propagate the most recent observed value forward.

    Equation (3.2): f̂_i = f_{max{j ∈ O : j < i}}

    Gaps at the very start of the light curve are back-filled from the
    first observed value after the gap.
    """

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        out = flux.copy().astype(float)
        N = len(out)

        # Forward pass (LOCF)
        last_obs = np.nan
        for i in range(N):
            if mask[i]:
                last_obs = out[i]
            elif not np.isnan(last_obs):
                out[i] = last_obs

        # Backward pass for leading NaN values
        first_obs = np.nan
        for i in range(N - 1, -1, -1):
            if mask[i]:
                first_obs = out[i]
            elif np.isnan(out[i]) and not np.isnan(first_obs):
                out[i] = first_obs

        return out


class LinearImputer(BaseImputer):
    """
    Linear interpolation between the nearest observed neighbours.

    Equation (3.3): f̂_i = f_a + (t_i - t_a)/(t_b - t_a) * (f_b - f_a)

    For leading/trailing gaps (no observed point on one side), the nearest
    observed point is used as a constant extension (boundary flat-fill).
    """

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        out = flux.copy().astype(float)
        obs_times  = time[mask]
        obs_fluxes = out[mask]

        if len(obs_times) < 2:
            # Degenerate case: fill with the single observed value
            out[~mask] = obs_fluxes[0] if len(obs_fluxes) == 1 else np.nan
            return out

        f_interp = interp1d(
            obs_times, obs_fluxes,
            kind="linear",
            bounds_error=False,
            fill_value=(obs_fluxes[0], obs_fluxes[-1]),
        )
        out[~mask] = f_interp(time[~mask])
        return out


class SplineImputer(BaseImputer):
    """
    Natural cubic spline interpolation.

    Minimises the integrated squared second derivative subject to
    interpolation conditions and natural boundary conditions
    S''(t_1) = S''(t_N) = 0.

    Implemented via scipy.interpolate.CubicSpline with bc_type='natural'.
    Can exhibit Runge-type oscillations near large gaps.
    """

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        out = flux.copy().astype(float)
        obs_times  = time[mask]
        obs_fluxes = out[mask]

        if len(obs_times) < 2:
            out[~mask] = np.nanmean(obs_fluxes) if len(obs_fluxes) > 0 else np.nan
            return out

        try:
            cs = CubicSpline(obs_times, obs_fluxes, bc_type="natural", extrapolate=True)
            out[~mask] = cs(time[~mask])
        except Exception:
            # Fall back to linear if spline fails (e.g., duplicate time stamps)
            f_interp = interp1d(
                obs_times, obs_fluxes,
                kind="linear",
                bounds_error=False,
                fill_value=(obs_fluxes[0], obs_fluxes[-1]),
            )
            out[~mask] = f_interp(time[~mask])

        return out
