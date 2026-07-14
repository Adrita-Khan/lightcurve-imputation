"""
KNN-Impute: K-Nearest-Neighbour flux imputation.

Context descriptors of width 2W are formed around each missing cadence.
The k most similar observed cadences (by Euclidean distance) provide
a distance-inverse-weighted average as the imputed value.

Implements Algorithm 3 (KNN-Impute) from the thesis.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ImputerBase

EPS_STAB = 1e-8


class KNNImputer(ImputerBase):
    """K-Nearest-Neighbour imputation in lag-feature space.

    Parameters
    ----------
    k : int
        Number of nearest neighbours.
    W : int
        Context window half-width.
    seed : int or None
        Random seed (unused; included for API consistency).
    """

    def __init__(self, k: int = 5, W: int = 10, seed: Optional[int] = None) -> None:
        super().__init__(name="KNN-Impute", seed=seed)
        self.k = k
        self.W = W

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        N = len(flux)
        W = self.W
        k = self.k

        # Pad the flux with NaN so edge windows are handled cleanly
        pad = np.full(W, np.nan)
        flux_pad = np.concatenate([pad, flux, pad])

        # Build lag matrix: each row i => flux[i-W:i+W] (2W features)
        def context(i: int) -> np.ndarray:
            return flux_pad[i : i + 2 * W]  # shape (2W,)

        obs_mask = ~np.isnan(flux)
        obs_idx = np.where(obs_mask)[0]

        # Pre-compute descriptors for all observed rows that are fully finite
        obs_descs = []
        obs_valid = []
        for j in obs_idx:
            d = context(j)
            if not np.any(np.isnan(d)):
                obs_descs.append(d)
                obs_valid.append(j)
        obs_descs = np.array(obs_descs)  # shape (M, 2W)
        obs_valid = np.array(obs_valid, dtype=int)

        imputed = flux.copy()

        for i in missing_idx:
            d_i = context(i)
            nan_mask = np.isnan(d_i)
            if nan_mask.all():
                # No context at all: use global mean
                imputed[i] = float(np.nanmean(flux))
                continue

            # Partial context: compute distance only over finite entries
            if nan_mask.any():
                finite = ~nan_mask
                if obs_descs.shape[0] == 0 or not finite.any():
                    imputed[i] = float(np.nanmean(flux))
                    continue
                dists = np.linalg.norm(obs_descs[:, finite] - d_i[finite], axis=1)
            else:
                if obs_descs.shape[0] == 0:
                    imputed[i] = float(np.nanmean(flux))
                    continue
                dists = np.linalg.norm(obs_descs - d_i, axis=1)

            # Pick k nearest
            nn_k = min(k, len(dists))
            nn_idx = np.argpartition(dists, nn_k - 1)[:nn_k]
            nn_dists = dists[nn_idx]
            nn_flux = flux[obs_valid[nn_idx]]

            weights = 1.0 / (nn_dists + EPS_STAB)
            imputed[i] = float(np.sum(weights * nn_flux) / np.sum(weights))

        return imputed
