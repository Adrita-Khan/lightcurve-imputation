"""
MF-Impute: Low-rank Hankel matrix factorisation imputation.

The flux vector is embedded into a Hankel matrix H ∈ R^{D×n_c}, where
D = floor(N/2).  A rank-r factorisation H ≈ UV^T is solved by Alternating
Least Squares (ALS) with Tikhonov regularisation on the observed entries.
The imputed flux is recovered by anti-diagonal averaging.

Implements Algorithm 7 and Equation 3.12 from the thesis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ImputerBase


class MFImputer(ImputerBase):
    """Low-rank Hankel Matrix Factorisation imputation.

    Parameters
    ----------
    rank : int
        Target rank of the factorisation.
    alpha : float
        Tikhonov (L2) regularisation coefficient.
    tol : float
        ALS convergence tolerance (relative Frobenius norm change).
    max_iter : int
        Maximum ALS iterations.
    seed : int or None
        Random seed for ALS initialisation.
    """

    def __init__(
        self,
        rank: int = 10,
        alpha: float = 1e-3,
        tol: float = 1e-4,
        max_iter: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(name="MF-Impute", seed=seed)
        self.rank = rank
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        N = len(flux)
        D = N // 2
        n_c = N - D + 1

        # Build Hankel matrix; NaN at missing positions
        H = _build_hankel(flux, D, n_c)

        # Observed-entry mask
        Omega = ~np.isnan(H)

        # Mean-fill for SVD initialisation
        col_means = np.nanmean(H, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        H_init = np.where(np.isnan(H), col_means[None, :], H)

        # Truncated SVD initialisation
        rng = np.random.default_rng(self.seed)
        r = min(self.rank, D, n_c)
        try:
            U0, s0, Vt0 = np.linalg.svd(H_init, full_matrices=False)
            U = U0[:, :r] * s0[:r]
            V = Vt0[:r, :].T
        except np.linalg.LinAlgError:
            U = rng.standard_normal((D, r)).astype(float)
            V = rng.standard_normal((n_c, r)).astype(float)

        # ALS optimisation
        U, V = _als(H, Omega, U, V, self.alpha, self.tol, self.max_iter)

        H_completed = U @ V.T

        # Anti-diagonal averaging to recover flux (Equation 3.12)
        flux_rec = _hankel_deembed(H_completed, N)

        imputed = flux.copy()
        imputed[missing_idx] = flux_rec[missing_idx]
        return imputed


def _build_hankel(flux: np.ndarray, D: int, n_c: int) -> np.ndarray:
    """Build a Hankel matrix from flux; H[j, k] = flux[j + k]."""
    H = np.full((D, n_c), np.nan)
    for j in range(D):
        H[j, :] = flux[j : j + n_c]
    return H


def _als(
    H: np.ndarray,
    Omega: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    alpha: float,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating Least Squares on observed entries."""
    D, n_c = H.shape
    r = U.shape[1]
    prev_loss = np.inf

    for _ in range(max_iter):
        # Update U: for each row i, solve (V^T V + alpha I) u_i = V^T h_i_obs
        VtV = V.T @ V + alpha * np.eye(r)
        for i in range(D):
            obs_cols = Omega[i, :]
            if not obs_cols.any():
                continue
            V_obs = V[obs_cols, :]
            h_obs = H[i, obs_cols]
            A = V_obs.T @ V_obs + alpha * np.eye(r)
            b = V_obs.T @ h_obs
            try:
                U[i, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                pass

        # Update V: for each col k, solve (U^T U + alpha I) v_k = U^T h_k_obs
        for k in range(n_c):
            obs_rows = Omega[:, k]
            if not obs_rows.any():
                continue
            U_obs = U[obs_rows, :]
            h_obs = H[obs_rows, k]
            A = U_obs.T @ U_obs + alpha * np.eye(r)
            b = U_obs.T @ h_obs
            try:
                V[k, :] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                pass

        H_approx = U @ V.T
        residual = H_approx[Omega] - H[Omega]
        loss = float(np.sqrt(np.mean(residual ** 2)))
        if prev_loss > 0 and abs(prev_loss - loss) / (prev_loss + 1e-10) < tol:
            break
        prev_loss = loss

    return U, V


def _hankel_deembed(H: np.ndarray, N: int) -> np.ndarray:
    """Recover flux from completed Hankel matrix by anti-diagonal averaging."""
    D, n_c = H.shape
    flux_rec = np.zeros(N)
    counts = np.zeros(N)
    for j in range(D):
        for k in range(n_c):
            idx = j + k
            if idx < N:
                flux_rec[idx] += H[j, k]
                counts[idx] += 1
    counts = np.where(counts == 0, 1, counts)
    return flux_rec / counts
