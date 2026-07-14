"""
MCAR gap injection with a 50/50 mixture of scattered and block gaps.

Implements Algorithm 2 (Gap Injection) from the thesis.  The mechanism is
Missing Completely At Random (MCAR): whether a cadence is removed is
independent of the flux value.

Gap patterns
------------
``scattered`` : Individual cadences removed independently at random.
``block``     : Contiguous runs of 5–50 cadences removed as a unit.
``mixed``     : 50/50 mixture of block and scattered (thesis default).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

GapPattern = Literal["scattered", "block", "mixed"]


def inject_gaps(
    flux: np.ndarray,
    p: float,
    seed: int | None = None,
    block_ratio: float = 0.50,
    block_min_len: int = 5,
    block_max_len: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject MCAR gaps into a complete flux vector.

    Parameters
    ----------
    flux : np.ndarray, shape (N,)
        Complete (gap-free) flux vector.
    p : float
        Target missingness fraction in (0, 1).
    seed : int or None
        Random seed for reproducibility.
    block_ratio : float
        Fraction of ``p * N`` missing points placed in contiguous blocks.
    block_min_len : int
        Minimum block gap length in cadences.
    block_max_len : int
        Maximum block gap length in cadences.

    Returns
    -------
    gapped : np.ndarray, shape (N,)
        Flux vector with NaN at missing positions.
    missing_idx : np.ndarray of int
        Sorted array of missing cadence indices.
    true_vals : np.ndarray
        Ground-truth flux values at missing positions (withheld for evaluation).

    Raises
    ------
    ValueError
        If ``p`` is not in (0, 1).
    """
    flux = np.asarray(flux, dtype=float)
    N = len(flux)

    if not (0.0 < p < 1.0):
        raise ValueError(f"Missingness fraction p must be in (0, 1); got {p}.")

    rng = np.random.default_rng(seed)
    n_missing = int(p * N)
    masked: set[int] = set()

    # --- Stage 1: contiguous block gaps ---
    n_block = int(block_ratio * n_missing)
    _fill_block_gaps(masked, rng, N, n_block, block_min_len, block_max_len)

    # --- Stage 2: scattered point gaps ---
    n_scatter = n_missing - len(masked)
    candidates = np.array([i for i in range(N) if i not in masked], dtype=int)
    if n_scatter > 0 and len(candidates) >= n_scatter:
        chosen = rng.choice(candidates, size=n_scatter, replace=False)
        masked.update(chosen.tolist())

    missing_idx = np.array(sorted(masked), dtype=int)
    true_vals = flux[missing_idx].copy()

    gapped = flux.copy()
    gapped[missing_idx] = np.nan

    return gapped, missing_idx, true_vals


def _fill_block_gaps(
    masked: set[int],
    rng: np.random.Generator,
    N: int,
    n_block: int,
    block_min_len: int,
    block_max_len: int,
) -> None:
    """Randomly place contiguous block gaps until ``n_block`` cadences are masked."""
    max_attempts = 10 * N  # safety limit to prevent infinite loop
    attempts = 0
    while len(masked) < n_block and attempts < max_attempts:
        blen = int(rng.integers(block_min_len, block_max_len + 1))
        start = int(rng.integers(0, max(1, N - blen)))
        masked.update(range(start, min(start + blen, N)))
        attempts += 1
