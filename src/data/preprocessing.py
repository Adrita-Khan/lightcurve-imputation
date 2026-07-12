"""
Light-curve preprocessing pipeline.

Implements Algorithm 1 (Light-Curve Preprocessing and Normalisation) from the thesis:
  1. Quarter concatenation with inter-quarter median offset correction.
  2. 5-sigma outlier rejection via 30-point rolling median + MAD-based scale.
  3. Quality-flag masking (Kepler non-zero flags dropped).
  4. Normalisation by the per-curve median flux.

All steps operate on (time, flux, flux_err) arrays and return a clean,
normalised light curve ready for gap injection.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_light_curve(
    df: pd.DataFrame,
    outlier_sigma: float = 5.0,
    rolling_window: int = 30,
    quality_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline (Algorithm 1).

    Parameters
    ----------
    df : pd.DataFrame
        Raw light curve with columns: time, flux, [flux_err], [quality].
    outlier_sigma : float
        Sigma-clipping threshold (default 5.0).
    rolling_window : int
        Window size for rolling-median reference (default 30).
    quality_col : str | None
        Column name for Kepler quality flags; None skips Step 3.

    Returns
    -------
    pd.DataFrame
        Cleaned, normalised light curve with columns: time, flux, flux_err.
        flux values are normalised fractional deviations (centred near 1.0).
    """
    df = df.copy().sort_values("time").reset_index(drop=True)

    # --- Step 1: Inter-quarter offset correction ---
    # (Quarter labels are inferred from time discontinuities if not present)
    df = _correct_quarter_offsets(df)

    # --- Step 2: 5-sigma outlier rejection ---
    df = _sigma_clip(df, sigma=outlier_sigma, window=rolling_window)

    # --- Step 3: Quality-flag masking ---
    if quality_col and quality_col in df.columns:
        before = len(df)
        df = df[df[quality_col] == 0].reset_index(drop=True)
        logger.debug("Quality masking removed %d cadences", before - len(df))

    # --- Step 4: Normalise by median flux ---
    median_flux = np.nanmedian(df["flux"].values)
    if median_flux == 0 or not np.isfinite(median_flux):
        raise ValueError("Median flux is zero or non-finite; cannot normalise.")
    df["flux"] = df["flux"] / median_flux
    if "flux_err" in df.columns:
        df["flux_err"] = df["flux_err"] / median_flux

    # Drop any remaining NaN flux rows
    df = df.dropna(subset=["flux"]).reset_index(drop=True)
    logger.debug("Preprocessing complete: %d cadences remain", len(df))
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _correct_quarter_offsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Correct inter-quarter flux level offsets by aligning each quarter's
    median to the global median.

    Quarters are detected via time-gap discontinuities (>5 cadences apart,
    where the typical Kepler long-cadence interval is ~0.0204 days).
    """
    times = df["time"].values
    fluxes = df["flux"].values

    # Detect quarter boundaries: gaps > 10× median cadence interval
    dt = np.diff(times)
    median_dt = np.median(dt)
    boundaries = np.where(dt > 10 * median_dt)[0] + 1
    quarter_starts = np.concatenate([[0], boundaries])
    quarter_ends   = np.concatenate([boundaries, [len(times)]])

    global_median = np.nanmedian(fluxes)
    corrected = fluxes.copy()

    for s, e in zip(quarter_starts, quarter_ends):
        q_flux = fluxes[s:e]
        q_median = np.nanmedian(q_flux)
        corrected[s:e] = q_flux - q_median + global_median

    df = df.copy()
    df["flux"] = corrected
    return df


def _sigma_clip(
    df: pd.DataFrame,
    sigma: float = 5.0,
    window: int = 30,
) -> pd.DataFrame:
    """
    Remove outliers deviating more than `sigma` MAD-based standard deviations
    from a rolling median.

    The scale estimate is MAD / 0.6745 (consistent standard deviation proxy),
    preferred over the sample std because extreme outliers would inflate it.
    """
    flux = df["flux"].values.copy().astype(float)
    series = pd.Series(flux)

    # Rolling median as local baseline
    rolling_med = series.rolling(window=window, center=True, min_periods=1).median().values
    residuals = flux - rolling_med

    # MAD-based scale estimate (robust to outliers)
    mad = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    sigma_clip = mad / 0.6745

    if sigma_clip == 0:
        # Constant flux segment — no clipping needed
        return df

    mask = np.abs(residuals) <= sigma * sigma_clip
    clipped = df[mask].reset_index(drop=True)
    n_removed = len(df) - len(clipped)
    if n_removed > 0:
        logger.debug("Sigma-clipping removed %d cadences", n_removed)
    return clipped


def compute_completeness(df: pd.DataFrame) -> float:
    """
    Return the fraction of finite-flux cadences.

    Parameters
    ----------
    df : pd.DataFrame
        Light curve DataFrame with a 'flux' column.
    """
    if len(df) == 0:
        return 0.0
    return float(np.sum(np.isfinite(df["flux"].values))) / len(df)


def truncate_to_window(
    df: pd.DataFrame,
    window_size: int = 4000,
    strategy: str = "first",
) -> pd.DataFrame:
    """
    Truncate or select a contiguous window of cadences.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-processed light curve.
    window_size : int
        Target number of cadences (default 4000).
    strategy : str
        'first' — take the first window_size cadences.
        'middle' — take the central window_size cadences.
        'densest' — take the window_size consecutive cadences with fewest NaNs.

    Returns
    -------
    pd.DataFrame
        Truncated light curve with exactly min(len(df), window_size) rows.
    """
    n = len(df)
    if n <= window_size:
        return df.reset_index(drop=True)

    if strategy == "first":
        return df.iloc[:window_size].reset_index(drop=True)
    if strategy == "middle":
        start = (n - window_size) // 2
        return df.iloc[start:start + window_size].reset_index(drop=True)
    if strategy == "densest":
        flux = df["flux"].values
        obs = np.isfinite(flux).astype(int)
        cum = np.cumsum(obs)
        counts = cum[window_size:] - cum[:-window_size]
        best_start = int(np.argmax(counts))
        return df.iloc[best_start:best_start + window_size].reset_index(drop=True)

    raise ValueError(f"Unknown strategy '{strategy}'. Choose 'first', 'middle', or 'densest'.")
