"""
Lomb-Scargle period estimation utilities.

Used by TS-MICE, RF-Impute, GB-MICE (for phase features), and the
feature extraction pipeline.
"""

from __future__ import annotations

import numpy as np


def lomb_scargle_period(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float | None = None,
    max_period: float | None = None,
    n_terms: int = 1,
    samples_per_peak: int = 5,
) -> float:
    """
    Estimate the dominant period using the Lomb-Scargle periodogram.

    Parameters
    ----------
    time : np.ndarray
        Observation times.
    flux : np.ndarray
        Flux values (no NaN).
    min_period : float | None
        Minimum search period (default: 2 × median cadence interval).
    max_period : float | None
        Maximum search period (default: half the time baseline).
    n_terms : int
        Number of Fourier terms (1 = standard LS).
    samples_per_peak : int
        Frequency grid density.

    Returns
    -------
    float
        Dominant period in the same units as `time`.
    """
    from astropy.timeseries import LombScargle

    if len(time) < 5:
        return float(np.ptp(time) / 2.0) if len(time) > 1 else 1.0

    baseline = np.ptp(time)
    dt_med   = float(np.median(np.diff(np.sort(time))))

    if min_period is None:
        min_period = max(2.0 * dt_med, 0.01)
    if max_period is None:
        max_period = baseline / 2.0

    if min_period >= max_period:
        return float(max_period)

    ls = LombScargle(time, flux, nterms=n_terms)
    freqs, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=samples_per_peak,
    )

    if len(power) == 0:
        return float(max_period)

    best_freq = float(freqs[np.argmax(power)])
    if best_freq <= 0:
        return float(max_period)

    return 1.0 / best_freq


def lomb_scargle_peak_power(
    time: np.ndarray,
    flux: np.ndarray,
    **kwargs,
) -> tuple[float, float]:
    """
    Return (peak_period, peak_power) from the Lomb-Scargle periodogram.

    Parameters
    ----------
    time : np.ndarray
    flux : np.ndarray

    Returns
    -------
    (period, power) : tuple of float
    """
    from astropy.timeseries import LombScargle

    if len(time) < 5:
        return float(np.ptp(time) / 2.0), 0.0

    baseline = np.ptp(time)
    dt_med   = float(np.median(np.diff(np.sort(time))))
    min_p = max(2.0 * dt_med, 0.01)
    max_p = baseline / 2.0

    if min_p >= max_p:
        return float(max_p), 0.0

    ls = LombScargle(time, flux)
    freqs, power = ls.autopower(
        minimum_frequency=1.0 / max_p,
        maximum_frequency=1.0 / min_p,
        samples_per_peak=5,
    )

    if len(power) == 0:
        return float(max_p), 0.0

    best_idx  = int(np.argmax(power))
    best_freq = float(freqs[best_idx])
    if best_freq <= 0:
        return float(max_p), 0.0

    return 1.0 / best_freq, float(power[best_idx])
