"""
35-dimensional feature vector extraction for variable-star classification.

Implements Algorithm 5 (Feature Extraction for Variable-Star Classification).

Feature groups:
  1. Period features     (4):  LS peak period, LS peak power, phase-folded amplitude, skewness
  2. Statistical moments (5):  weighted mean, std, skewness, kurtosis, MAD
  3. Variability indices (5):  Stetson J, K, L; η index; r_cs
  4. Flux percentile ratios (3): F_{5/95}, F_{10/90}, F_{25/75}
  5. Autocorrelation     (6):  ACF at lags {1, 2, 3, 5, 10, 20}
  ... plus 12 additional features to reach 35 total (detailed below)

Total = 4 + 5 + 5 + 3 + 6 + 12 = 35 features
"""

from __future__ import annotations

import logging

import numpy as np

from ..utils.period import lomb_scargle_peak_power

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    # Group 1 — Period (4)
    "ls_period",
    "ls_power",
    "phase_amplitude",
    "phase_skewness",
    # Group 2 — Statistical moments (5)
    "flux_mean",
    "flux_std",
    "flux_skewness",
    "flux_kurtosis",
    "flux_mad",
    # Group 3 — Variability indices (5)
    "stetson_j",
    "stetson_k",
    "stetson_l",
    "eta_index",
    "r_cs",
    # Group 4 — Flux percentile ratios (3)
    "f_5_95",
    "f_10_90",
    "f_25_75",
    # Group 5 — Autocorrelation (6)
    "acf_lag1",
    "acf_lag2",
    "acf_lag3",
    "acf_lag5",
    "acf_lag10",
    "acf_lag20",
    # Group 6 — Additional period features (6)
    "phase_kurtosis",
    "phase_std",
    "period_ratio_half",
    "period_ratio_double",
    "ls_power_2nd",
    "ls_period_2nd",
    # Group 7 — Additional variability (6)
    "flux_range",
    "above_2sigma_frac",
    "below_2sigma_frac",
    "flux_median",
    "flux_trimmed_mean",
    "slope_ratio",
]

N_FEATURES = len(FEATURE_NAMES)  # 35


def extract_features(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray | None = None,
    acf_lags: list[int] | None = None,
) -> np.ndarray:
    """
    Extract the 35-dimensional feature vector from a light curve.

    Parameters
    ----------
    time : np.ndarray, shape (N,)
        Cadence times.
    flux : np.ndarray, shape (N,)
        Normalised flux (should be complete or imputed — no NaN).
    flux_err : np.ndarray | None
        Photometric uncertainties. If None, uniform weights are assumed.
    acf_lags : list[int] | None
        Autocorrelation lags to compute (default: [1, 2, 3, 5, 10, 20]).

    Returns
    -------
    phi : np.ndarray, shape (35,)
        Feature vector φ_j.
    """
    if acf_lags is None:
        acf_lags = [1, 2, 3, 5, 10, 20]

    # Handle degenerate inputs
    flux = np.array(flux, dtype=float)
    time = np.array(time, dtype=float)
    finite = np.isfinite(flux)
    if finite.sum() < 5:
        return np.zeros(N_FEATURES)

    flux = flux[finite]
    time = time[finite]
    if flux_err is not None:
        flux_err = np.array(flux_err, dtype=float)[finite]
        flux_err = np.where(flux_err > 0, flux_err, np.nanmedian(flux_err[flux_err > 0]))
    else:
        flux_err = np.ones(len(flux))

    weights = 1.0 / (flux_err ** 2 + 1e-20)
    weights /= weights.sum()

    phi = np.zeros(N_FEATURES)

    # ---- Group 1: Period features ----
    period, ls_pow = lomb_scargle_peak_power(time, flux)
    phi[0] = np.log10(period + 1e-10) if period > 0 else 0.0
    phi[1] = ls_pow

    # Phase-fold
    phase = (time % period) / period if period > 0 else time / (time[-1] - time[0] + 1e-10)
    order = np.argsort(phase)
    f_phase = flux[order]
    phi[2] = float(np.ptp(f_phase))               # phase-folded amplitude
    phi[3] = float(_safe_skewness(f_phase))        # phase-folded skewness

    # ---- Group 2: Statistical moments ----
    phi[4] = float(np.average(flux, weights=weights))
    phi[5] = float(np.sqrt(np.average((flux - phi[4]) ** 2, weights=weights)))
    phi[6] = float(_safe_skewness(flux))
    phi[7] = float(_safe_kurtosis(flux))
    phi[8] = float(np.median(np.abs(flux - np.median(flux))))

    # ---- Group 3: Variability indices ----
    phi[9]  = float(_stetson_j(flux, flux_err))
    phi[10] = float(_stetson_k(flux))
    phi[11] = float(_stetson_l(flux, flux_err))
    phi[12] = float(_eta_index(flux))
    phi[13] = float(_r_cs(flux))

    # ---- Group 4: Flux percentile ratios ----
    p5, p95 = np.percentile(flux, [5, 95])
    p10, p90 = np.percentile(flux, [10, 90])
    p25, p75 = np.percentile(flux, [25, 75])
    denom = np.abs(np.median(flux)) + 1e-10
    phi[14] = (p95 - p5)  / denom
    phi[15] = (p90 - p10) / denom
    phi[16] = (p75 - p25) / denom

    # ---- Group 5: Autocorrelation ----
    for li, lag in enumerate(acf_lags[:6]):
        phi[17 + li] = float(_acf(flux, lag))

    # ---- Group 6: Additional period features ----
    phi[23] = float(_safe_kurtosis(f_phase))
    phi[24] = float(np.std(f_phase))
    phi[25] = phi[0] - np.log10(2 + 1e-10)        # log(period/2)
    phi[26] = phi[0] + np.log10(2 + 1e-10)        # log(2*period)

    # Second LS peak
    period2, ls_pow2 = _lomb_scargle_second_peak(time, flux, period)
    phi[27] = ls_pow2
    phi[28] = np.log10(period2 + 1e-10) if period2 > 0 else 0.0

    # ---- Group 7: Additional variability ----
    phi[29] = float(np.ptp(flux))
    sigma = phi[5] if phi[5] > 0 else 1.0
    mu = phi[4]
    phi[30] = float(np.mean(flux > mu + 2 * sigma))
    phi[31] = float(np.mean(flux < mu - 2 * sigma))
    phi[32] = float(np.median(flux))
    phi[33] = float(_trimmed_mean(flux, 0.05))
    phi[34] = float(_slope_sign_ratio(flux))

    # Replace any NaN/Inf with 0
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    return phi


# ---------------------------------------------------------------------------
# Variability index helpers
# ---------------------------------------------------------------------------

def _stetson_j(flux: np.ndarray, flux_err: np.ndarray) -> float:
    """Stetson J index (Stetson 1996)."""
    n = len(flux)
    if n < 2:
        return 0.0
    mu = np.average(flux, weights=1.0 / flux_err ** 2)
    delta = np.sqrt(n / (n - 1.0)) * (flux - mu) / flux_err
    pairs = n // 2
    if pairs == 0:
        return 0.0
    d1 = delta[:2 * pairs:2]
    d2 = delta[1:2 * pairs:2]
    sgn = np.sign(d1 * d2)
    return float(np.mean(sgn * np.sqrt(np.abs(d1 * d2))))


def _stetson_k(flux: np.ndarray) -> float:
    """Stetson K index."""
    n = len(flux)
    if n < 2:
        return 0.0
    mu = np.mean(flux)
    resid = flux - mu
    sigma = np.std(resid)
    if sigma == 0:
        return 0.0
    z = resid / sigma
    return float(np.mean(np.abs(z)) / np.sqrt(np.mean(z ** 2) + 1e-10))


def _stetson_l(flux: np.ndarray, flux_err: np.ndarray) -> float:
    """Stetson L index (combination of J and K)."""
    j = _stetson_j(flux, flux_err)
    k = _stetson_k(flux)
    return j * k / 0.798  # 0.798 is the expected K under Gaussian noise


def _eta_index(flux: np.ndarray) -> float:
    """Eta variability index: ratio of successive-difference variance to variance."""
    n = len(flux)
    if n < 2:
        return 0.0
    var_total = np.var(flux)
    if var_total == 0:
        return 0.0
    var_diff = np.mean(np.diff(flux) ** 2)
    return float(var_diff / (2.0 * var_total + 1e-10))


def _r_cs(flux: np.ndarray) -> float:
    """
    Consecutive-slope-sign ratio — proxy for smoothness.
    Fraction of consecutive pairs with the same sign of slope.
    """
    if len(flux) < 3:
        return 0.0
    slopes = np.diff(flux)
    signs  = np.sign(slopes)
    same_sign = (signs[:-1] == signs[1:])
    return float(np.mean(same_sign))


def _acf(flux: np.ndarray, lag: int) -> float:
    """Normalised autocorrelation at integer lag."""
    n = len(flux)
    if lag >= n:
        return 0.0
    mu = np.mean(flux)
    var = np.var(flux)
    if var == 0:
        return 0.0
    cov = np.mean((flux[:n - lag] - mu) * (flux[lag:] - mu))
    return float(cov / (var + 1e-20))


def _safe_skewness(x: np.ndarray) -> float:
    from scipy.stats import skew
    if len(x) < 3 or np.std(x) == 0:
        return 0.0
    return float(skew(x))


def _safe_kurtosis(x: np.ndarray) -> float:
    from scipy.stats import kurtosis
    if len(x) < 4 or np.std(x) == 0:
        return 0.0
    return float(kurtosis(x))


def _trimmed_mean(x: np.ndarray, frac: float = 0.05) -> float:
    n = len(x)
    lo = int(np.floor(frac * n))
    hi = n - lo
    x_sorted = np.sort(x)
    return float(np.mean(x_sorted[lo:hi])) if hi > lo else float(np.mean(x))


def _slope_sign_ratio(flux: np.ndarray) -> float:
    """Fraction of positive slopes."""
    if len(flux) < 2:
        return 0.5
    slopes = np.diff(flux)
    return float(np.mean(slopes > 0))


def _lomb_scargle_second_peak(
    time: np.ndarray,
    flux: np.ndarray,
    first_period: float,
    window: float = 0.05,
) -> tuple[float, float]:
    """Find the second-highest LS peak (excluding ±window around first_period)."""
    from astropy.timeseries import LombScargle

    if len(time) < 5:
        return first_period, 0.0

    baseline = np.ptp(time)
    dt_med   = float(np.median(np.diff(np.sort(time))))
    min_p = max(2.0 * dt_med, 0.01)
    max_p = baseline / 2.0

    if min_p >= max_p:
        return first_period, 0.0

    ls = LombScargle(time, flux)
    freqs, power = ls.autopower(
        minimum_frequency=1.0 / max_p,
        maximum_frequency=1.0 / min_p,
        samples_per_peak=5,
    )

    # Exclude frequencies near first_period
    f1 = 1.0 / first_period if first_period > 0 else 0.0
    excl = np.abs(freqs - f1) / (f1 + 1e-10) < window
    power2 = power.copy()
    power2[excl] = 0.0

    if power2.max() == 0:
        return first_period, 0.0

    best_idx = int(np.argmax(power2))
    best_f2  = float(freqs[best_idx])
    return (1.0 / best_f2 if best_f2 > 0 else first_period), float(power2[best_idx])
