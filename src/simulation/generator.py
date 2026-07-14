"""
Synthetic periodic light-curve generator.

Implements Algorithm 1 (Signal Generation) from the thesis.  All parameters
are exposed as keyword arguments with sensible defaults matching
Table 3.1 (signal parameters) in the thesis.

Supported signal models
-----------------------
``sinusoidal``      : Single-frequency sinusoid  f = mu + A sin(2πt/P + φ) + ε
``multi_harmonic``  : Sum of K sinusoids with user-supplied harmonics
``eclipsing_binary``: Approximate eclipsing-binary light curve (trapezoidal dip)
``sawtooth``        : Sawtooth wave built from Fourier partial sums
``custom``          : User-supplied callable func(t) -> array
"""

from __future__ import annotations

import warnings
from typing import Callable, Literal, Optional, Sequence

import numpy as np

SignalModel = Literal["sinusoidal", "multi_harmonic", "eclipsing_binary", "sawtooth", "custom"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_synthetic_lightcurve(
    N: int = 1323,
    dt: float = 0.0204,
    A: float = 0.1,
    P0: float = 1.0,
    phi0: float = 0.0,
    mu0: float = 1.0,
    sigma_eps: float = 0.02,
    seed: int = 0,
    model: SignalModel = "sinusoidal",
    # Multi-harmonic options
    harmonics: Optional[Sequence[int]] = None,
    harmonic_amps: Optional[Sequence[float]] = None,
    harmonic_phases: Optional[Sequence[float]] = None,
    # Eclipsing-binary options
    eclipse_depth: float = 0.08,
    eclipse_duration: float = 0.15,
    secondary_depth: float = 0.03,
    # Sawtooth options
    n_harmonics_sawtooth: int = 10,
    # Custom model
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a synthetic periodic light curve with Gaussian noise.

    Parameters
    ----------
    N : int
        Number of uniformly spaced cadences.
    dt : float
        Cadence spacing in days.
    A : float
        Signal amplitude (half-range) in normalised flux units.
    P0 : float
        True period in days.
    phi0 : float
        Initial phase in radians.
    mu0 : float
        Baseline (mean) flux.
    sigma_eps : float
        Standard deviation of i.i.d. Gaussian noise.
    seed : int
        Random seed for noise generation.  Fixed so that the same
        ground-truth signal is reproduced on every call.
    model : str
        Signal model: ``'sinusoidal'`` (default), ``'multi_harmonic'``,
        ``'eclipsing_binary'``, ``'sawtooth'``, or ``'custom'``.
    harmonics : sequence of int, optional
        For ``'multi_harmonic'`` model: harmonic numbers (e.g. [1, 2, 3]).
    harmonic_amps : sequence of float, optional
        Amplitudes for each harmonic (same length as ``harmonics``).
    harmonic_phases : sequence of float, optional
        Phases for each harmonic (same length as ``harmonics``).
    eclipse_depth : float
        Primary eclipse depth for ``'eclipsing_binary'`` model.
    eclipse_duration : float
        Eclipse duration as a fraction of the period.
    secondary_depth : float
        Secondary eclipse depth for ``'eclipsing_binary'`` model.
    n_harmonics_sawtooth : int
        Number of Fourier partial-sum terms for ``'sawtooth'`` model.
    custom_func : callable, optional
        A function ``func(t) -> np.ndarray`` defining the noiseless signal
        when ``model='custom'``.

    Returns
    -------
    t : np.ndarray, shape (N,)
        Time vector in days.
    flux : np.ndarray, shape (N,)
        Noisy flux vector (ground truth).
    params : dict
        Dictionary of all input parameters (useful for logging).

    Raises
    ------
    ValueError
        If an unknown ``model`` string is provided.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N, dtype=float) * dt

    signal = _build_noiseless_signal(
        t=t,
        A=A,
        P0=P0,
        phi0=phi0,
        mu0=mu0,
        model=model,
        harmonics=harmonics,
        harmonic_amps=harmonic_amps,
        harmonic_phases=harmonic_phases,
        eclipse_depth=eclipse_depth,
        eclipse_duration=eclipse_duration,
        secondary_depth=secondary_depth,
        n_harmonics_sawtooth=n_harmonics_sawtooth,
        custom_func=custom_func,
    )

    noise = rng.normal(0.0, sigma_eps, size=N)
    flux = signal + noise

    params = {
        "N": N,
        "dt": dt,
        "A": A,
        "P0": P0,
        "phi0": phi0,
        "mu0": mu0,
        "sigma_eps": sigma_eps,
        "seed": seed,
        "model": model,
    }

    return t, flux, params


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_noiseless_signal(
    t: np.ndarray,
    A: float,
    P0: float,
    phi0: float,
    mu0: float,
    model: SignalModel,
    harmonics: Optional[Sequence[int]],
    harmonic_amps: Optional[Sequence[float]],
    harmonic_phases: Optional[Sequence[float]],
    eclipse_depth: float,
    eclipse_duration: float,
    secondary_depth: float,
    n_harmonics_sawtooth: int,
    custom_func: Optional[Callable[[np.ndarray], np.ndarray]],
) -> np.ndarray:
    """Return the noiseless signal array for the chosen model."""
    if model == "sinusoidal":
        return _sinusoidal(t, A, P0, phi0, mu0)

    elif model == "multi_harmonic":
        return _multi_harmonic(t, A, P0, phi0, mu0, harmonics, harmonic_amps, harmonic_phases)

    elif model == "eclipsing_binary":
        return _eclipsing_binary(t, A, P0, phi0, mu0, eclipse_depth, eclipse_duration, secondary_depth)

    elif model == "sawtooth":
        return _sawtooth(t, A, P0, phi0, mu0, n_harmonics_sawtooth)

    elif model == "custom":
        if custom_func is None:
            raise ValueError("custom_func must be provided when model='custom'.")
        return np.asarray(custom_func(t), dtype=float)

    else:
        raise ValueError(
            f"Unknown signal model '{model}'. "
            "Choose from: sinusoidal, multi_harmonic, eclipsing_binary, sawtooth, custom."
        )


def _sinusoidal(t: np.ndarray, A: float, P0: float, phi0: float, mu0: float) -> np.ndarray:
    """Single-frequency sinusoid (Equation 3.1 in thesis)."""
    return mu0 + A * np.sin(2.0 * np.pi * t / P0 + phi0)


def _multi_harmonic(
    t: np.ndarray,
    A: float,
    P0: float,
    phi0: float,
    mu0: float,
    harmonics: Optional[Sequence[int]],
    harmonic_amps: Optional[Sequence[float]],
    harmonic_phases: Optional[Sequence[float]],
) -> np.ndarray:
    """Sum of sinusoids at integer harmonics of the fundamental period."""
    if harmonics is None:
        harmonics = [1, 2, 3]
    harmonics = list(harmonics)
    K = len(harmonics)

    if harmonic_amps is None:
        # Exponentially decaying amplitudes: A, A/2, A/4, ...
        harmonic_amps = [A / (2 ** k) for k in range(K)]
    if harmonic_phases is None:
        harmonic_phases = [phi0] * K

    if len(harmonic_amps) != K or len(harmonic_phases) != K:
        raise ValueError("harmonics, harmonic_amps, and harmonic_phases must have the same length.")

    signal = np.full_like(t, mu0)
    for h, amp, phase in zip(harmonics, harmonic_amps, harmonic_phases):
        signal += amp * np.sin(2.0 * np.pi * h * t / P0 + phase)
    return signal


def _eclipsing_binary(
    t: np.ndarray,
    A: float,
    P0: float,
    phi0: float,
    mu0: float,
    eclipse_depth: float,
    eclipse_duration: float,
    secondary_depth: float,
) -> np.ndarray:
    """Approximate eclipsing-binary light curve with primary and secondary eclipse."""
    phase = ((t / P0 + phi0 / (2 * np.pi)) % 1.0)
    signal = np.full_like(t, mu0)

    # Primary eclipse centred at phase 0 (or 1)
    _apply_eclipse(signal, phase, centre=0.0, depth=eclipse_depth, duration=eclipse_duration)

    # Secondary eclipse centred at phase 0.5
    _apply_eclipse(signal, phase, centre=0.5, depth=secondary_depth, duration=eclipse_duration * 0.7)

    # Ellipsoidal variation (low-amplitude sinusoid at half the period)
    signal += 0.5 * A * np.cos(2.0 * 2.0 * np.pi * t / P0)

    return signal


def _apply_eclipse(
    signal: np.ndarray, phase: np.ndarray, centre: float, depth: float, duration: float
) -> None:
    """Subtract a trapezoidal dip from ``signal`` in place."""
    half = duration / 2.0
    d = np.minimum(np.abs(phase - centre), 1.0 - np.abs(phase - centre))
    ingress = half * 0.2  # 20% of half-duration for ingress/egress ramp
    mask_full = d <= (half - ingress)
    mask_ingress = (d > (half - ingress)) & (d <= half)
    signal[mask_full] -= depth
    ramp = (half - d[mask_ingress]) / ingress
    signal[mask_ingress] -= depth * ramp


def _sawtooth(
    t: np.ndarray,
    A: float,
    P0: float,
    phi0: float,
    mu0: float,
    n_harmonics: int,
) -> np.ndarray:
    """Sawtooth wave via Fourier partial sums."""
    signal = np.full_like(t, mu0)
    for k in range(1, n_harmonics + 1):
        sign = (-1) ** (k + 1)
        signal += sign * (2.0 * A / (np.pi * k)) * np.sin(2.0 * np.pi * k * t / P0 + phi0)
    return signal
