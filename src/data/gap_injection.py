"""
MCAR gap injection for controlled corrupt-and-recover experiments.

Implements Algorithm 2 (MCAR Gap Injection with Block/Scattered Mixture) from the thesis.

Two complementary gap patterns are combined:
  - **Block gaps**: contiguous runs of 5–50 cadences, simulating satellite downtime.
  - **Scattered gaps**: individually dropped cadences (MCAR).

The default mixture is 50% block, 50% scattered.

Usage
-----
    from src.data.gap_injection import inject_gaps

    gapped_flux, mask, ground_truth = inject_gaps(
        flux=flux_array,
        p=0.30,
        seed=7,
    )
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def inject_gaps(
    flux: np.ndarray,
    p: float,
    seed: int,
    block_ratio: float = 0.50,
    block_min_len: int = 5,
    block_max_len: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject synthetic gaps into a complete flux vector (Algorithm 2).

    Parameters
    ----------
    flux : np.ndarray, shape (N,)
        Complete, normalised flux vector. Must not contain NaN.
    p : float
        Target missingness fraction ∈ (0, 1).
    seed : int
        Random seed for reproducibility.
    block_ratio : float
        Fraction of total gaps allocated to contiguous blocks (default 0.5).
    block_min_len : int
        Minimum block gap length in cadences (default 5).
    block_max_len : int
        Maximum block gap length in cadences (default 50).

    Returns
    -------
    gapped_flux : np.ndarray, shape (N,)
        Flux vector with NaN at missing positions.
    mask : np.ndarray of bool, shape (N,)
        True at *observed* positions, False at missing positions.
    ground_truth : np.ndarray
        Withheld ground-truth flux values at missing positions, indexed by
        the positions in np.where(~mask)[0].
    """
    rng = np.random.default_rng(seed)
    N = len(flux)
    n_gap = int(np.floor(p * N))
    n_block = int(np.floor(block_ratio * n_gap))

    missing = set()

    # --- Stage 1: Contiguous block gaps ---
    attempts = 0
    while len(missing) < n_block:
        length = int(rng.integers(block_min_len, block_max_len + 1))
        start  = int(rng.integers(0, N - length + 1))
        missing.update(range(start, start + length))
        attempts += 1
        if attempts > 10 * N:  # guard against infinite loop
            break

    # --- Stage 2: Scattered MCAR gaps ---
    n_scatter = n_gap - len(missing)
    if n_scatter > 0:
        candidates = np.array([i for i in range(N) if i not in missing])
        if len(candidates) >= n_scatter:
            chosen = rng.choice(candidates, size=n_scatter, replace=False)
            missing.update(chosen.tolist())

    missing_idx = np.array(sorted(missing), dtype=int)

    # --- Stage 3: Apply mask and store withheld ground truth ---
    gapped_flux = flux.copy().astype(float)
    ground_truth = flux[missing_idx].copy()
    gapped_flux[missing_idx] = np.nan

    mask = np.ones(N, dtype=bool)
    mask[missing_idx] = False

    return gapped_flux, mask, ground_truth


def generate_all_seeds(
    flux: np.ndarray,
    p: float,
    n_seeds: int = 30,
    base_seed: int = 0,
    **kwargs,
):
    """
    Generate `n_seeds` independent gap realisations for a single light curve.

    Parameters
    ----------
    flux : np.ndarray
        Complete flux vector.
    p : float
        Missingness fraction.
    n_seeds : int
        Number of independent realisations (default 30).
    base_seed : int
        First seed; subsequent seeds are base_seed + 1, base_seed + 2, …

    Yields
    ------
    tuple (seed, gapped_flux, mask, ground_truth)
    """
    for s in range(n_seeds):
        seed = base_seed + s
        gapped_flux, mask, ground_truth = inject_gaps(flux, p, seed, **kwargs)
        yield seed, gapped_flux, mask, ground_truth
