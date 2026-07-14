"""
GAIN-Impute: Generative Adversarial Imputation Network.

Trains a generator G and discriminator D jointly per realisation.
The hint mechanism with rate ρ=0.1 is used to guide the discriminator.

Implements Algorithm 6 and Equations 3.8–3.11 from the thesis.
Reference: Yoon et al. (2018) "GAIN: Missing Data Imputation using
Generative Adversarial Nets."
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import ImputerBase

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class GAINImputer(ImputerBase):
    """GAIN-based imputation.

    Parameters
    ----------
    hint_rate : float
        Fraction of observed entries revealed as hints to discriminator.
    lambda_recon : float
        Weight of the reconstruction loss in the generator objective.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        hint_rate: float = 0.1,
        lambda_recon: float = 10.0,
        epochs: int = 200,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GAINImputer. pip install torch==2.0.1")
        super().__init__(name="GAIN-Impute", seed=seed)
        self.hint_rate = hint_rate
        self.lambda_recon = lambda_recon
        self.epochs = epochs
        self.lr = lr

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        N = len(flux)
        m = (~np.isnan(flux)).astype(np.float32)  # observed mask
        x = np.where(m.astype(bool), flux, 0.0).astype(np.float32)

        # Normalise to [0, 1] for GAIN training stability
        x_min = float(np.min(x[m.astype(bool)]))
        x_max = float(np.max(x[m.astype(bool)]))
        x_range = max(x_max - x_min, 1e-8)
        x_norm = (x - x_min) / x_range

        X_t = torch.tensor(x_norm).unsqueeze(0)   # (1, N)
        M_t = torch.tensor(m).unsqueeze(0)         # (1, N)

        G = _FCNet(N, N)
        D = _FCNet(N * 2, N)
        opt_G = torch.optim.Adam(G.parameters(), lr=self.lr)
        opt_D = torch.optim.Adam(D.parameters(), lr=self.lr)

        rng_t = np.random.default_rng(self.seed)

        for _ in range(self.epochs):
            # Build inputs
            z = torch.tensor(rng_t.standard_normal(N).astype(np.float32)).unsqueeze(0)
            x_tilde = M_t * X_t + (1 - M_t) * z  # Eq 3.8

            # Hint
            b_vec = (rng_t.uniform(size=N) < self.hint_rate).astype(np.float32)
            h = M_t * (1 - torch.tensor(b_vec).unsqueeze(0)) + 0.5 * torch.tensor(b_vec).unsqueeze(0)

            # Generator
            g_out = torch.sigmoid(G(x_tilde))
            x_hat = M_t * X_t + (1 - M_t) * g_out  # Eq 3.11

            # Discriminator update
            d_in = torch.cat([x_hat.detach(), h], dim=1)
            d_out = torch.sigmoid(D(d_in))
            loss_D = -torch.mean(
                M_t * torch.log(d_out + 1e-8)
                + (1 - M_t) * torch.log(1 - d_out + 1e-8)
            )
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Generator update
            d_in2 = torch.cat([x_hat, h], dim=1)
            d_out2 = torch.sigmoid(D(d_in2))
            loss_G_adv = -torch.mean((1 - M_t) * torch.log(d_out2 + 1e-8))
            loss_G_recon = torch.mean(M_t * (g_out - X_t) ** 2)
            loss_G = loss_G_adv + self.lambda_recon * loss_G_recon
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        # Final imputation
        with torch.no_grad():
            z_final = torch.zeros(1, N)
            x_tilde_final = M_t * X_t + (1 - M_t) * z_final
            g_final = torch.sigmoid(G(x_tilde_final)).squeeze(0).numpy()

        # Denormalise
        g_denorm = g_final * x_range + x_min

        imputed = flux.copy()
        imputed[missing_idx] = g_denorm[missing_idx]
        return imputed


if _TORCH_AVAILABLE:

    class _FCNet(nn.Module):
        def __init__(self, in_dim: int, out_dim: int) -> None:
            super().__init__()
            hidden = max(32, min(256, in_dim))
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)
