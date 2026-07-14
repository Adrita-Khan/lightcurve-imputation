"""
Reconstruction quality metrics for imputed light curves.

All metrics are evaluated only at the artificially missing cadences (those in
``missing_idx``), comparing the imputed flux against the withheld ground truth.

Metrics implemented
-------------------
- RMSE   : Root Mean Squared Error (Equation 3.13)
- MAE    : Mean Absolute Error (Equation 3.14)
- MSE    : Mean Squared Error
- RelErr : Mean relative error at missing cadences
- PRR    : Period Recovery Rate (fraction of seeds with ε_P < tol)
- ε_A   : Mean relative amplitude error (Equation 3.15)
- ε_φ   : Mean phase offset (Equation 3.16)
- Runtime, Memory
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from astropy.timeseries import LombScargle

EPS_STAB = 1e-8


# ---------------------------------------------------------------------------
# Pointwise error metrics
# ---------------------------------------------------------------------------


def compute_rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Root Mean Squared Error at missing positions."""
    diff = np.asarray(true, float) - np.asarray(pred, float)
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_mae(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Error at missing positions."""
    diff = np.asarray(true, float) - np.asarray(pred, float)
    return float(np.mean(np.abs(diff)))


def compute_mse(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean Squared Error at missing positions."""
    diff = np.asarray(true, float) - np.asarray(pred, float)
    return float(np.mean(diff ** 2))


def compute_relative_error(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean relative error |true - pred| / (|true| + ε) at missing positions."""
    true = np.asarray(true, float)
    pred = np.asarray(pred, float)
    return float(np.mean(np.abs(true - pred) / (np.abs(true) + EPS_STAB)))


# ---------------------------------------------------------------------------
# Signal-property metrics
# ---------------------------------------------------------------------------


def compute_period_recovery(
    t: np.ndarray,
    flux_imputed: np.ndarray,
    true_period: float,
    tol: float = 0.01,
) -> tuple[bool, float]:
    """Assess period recovery via Lomb–Scargle periodogram.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    flux_imputed : np.ndarray
        Imputed flux (no NaN).
    true_period : float
        Ground-truth period.
    tol : float
        Relative period error threshold for a successful recovery.

    Returns
    -------
    recovered : bool
        True if ``|P_est - P_true| / P_true < tol``.
    rel_err : float
        Relative period error ε_P.
    """
    ls = LombScargle(t, flux_imputed)
    freq, power = ls.autopower(minimum_frequency=0.5 / t[-1], maximum_frequency=10.0)
    best_freq = freq[np.argmax(power)]
    P_est = 1.0 / best_freq if best_freq > 0 else np.inf
    rel_err = abs(P_est - true_period) / (true_period + EPS_STAB)
    return bool(rel_err < tol), float(rel_err)


def compute_amplitude_error(
    flux_imputed: np.ndarray,
    true_amplitude: float,
) -> float:
    """Relative amplitude error ε_A (Equation 3.15).

    Estimated amplitude = (max - min) / 2 of the imputed signal.
    """
    amp_est = (np.max(flux_imputed) - np.min(flux_imputed)) / 2.0
    return float(abs(amp_est - true_amplitude) / (true_amplitude + EPS_STAB))


def compute_phase_error(
    t: np.ndarray,
    flux_imputed: np.ndarray,
    true_period: float,
    true_phase: float,
) -> float:
    """Mean phase offset ε_φ in units of the signal period (Equation 3.16).

    Estimated phase from the dominant Fourier coefficient.
    """
    ls = LombScargle(t, flux_imputed)
    freq, power = ls.autopower(minimum_frequency=0.5 / t[-1], maximum_frequency=10.0)
    best_freq = freq[np.argmax(power)]

    # Fit sinusoidal model at the estimated frequency
    omega = 2.0 * np.pi * best_freq
    A_matrix = np.column_stack([np.sin(omega * t), np.cos(omega * t)])
    try:
        coeffs, *_ = np.linalg.lstsq(A_matrix, flux_imputed, rcond=None)
        phase_est = float(np.arctan2(coeffs[0], coeffs[1]))
    except np.linalg.LinAlgError:
        return 1.0

    diff = abs(phase_est - true_phase)
    # Wrap to [0, π]
    diff = diff % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    # Normalise to period fractions
    return float(diff / (2.0 * np.pi))


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------


def evaluate_imputation(
    t: np.ndarray,
    flux_imputed: np.ndarray,
    missing_idx: np.ndarray,
    true_vals: np.ndarray,
    true_period: float,
    true_amplitude: float,
    true_phase: float,
    period_tol: float = 0.01,
    runtime_s: float = 0.0,
    memory_mb: float = 0.0,
) -> dict:
    """Compute the full suite of evaluation metrics for one imputation result.

    Parameters
    ----------
    t : np.ndarray
        Time vector.
    flux_imputed : np.ndarray
        Fully imputed flux (no NaN).
    missing_idx : np.ndarray
        Indices of missing cadences.
    true_vals : np.ndarray
        Withheld ground-truth flux at ``missing_idx``.
    true_period, true_amplitude, true_phase : float
        Known ground-truth signal parameters.
    period_tol : float
        Relative period error threshold for PRR.
    runtime_s : float
        Wall-clock time (seconds) from the imputer.
    memory_mb : float
        Peak memory increase (MiB) from the imputer.

    Returns
    -------
    dict
        Mapping of metric names to float values.
    """
    pred_vals = flux_imputed[missing_idx]

    rmse = compute_rmse(true_vals, pred_vals)
    mae = compute_mae(true_vals, pred_vals)
    mse = compute_mse(true_vals, pred_vals)
    rel_err = compute_relative_error(true_vals, pred_vals)
    recovered, period_rel_err = compute_period_recovery(t, flux_imputed, true_period, period_tol)
    amp_err = compute_amplitude_error(flux_imputed, true_amplitude)
    phase_err = compute_phase_error(t, flux_imputed, true_period, true_phase)

    return {
        "rmse": rmse,
        "mae": mae,
        "mse": mse,
        "relative_error": rel_err,
        "period_recovered": int(recovered),
        "period_rel_err": period_rel_err,
        "amplitude_error": amp_err,
        "phase_error": phase_err,
        "runtime_s": runtime_s,
        "memory_mb": memory_mb,
    }
