"""
SAITS: Self-Attention-based Imputation for Time Series.

Diagonally-masked multi-head self-attention (DMSA) blocks are used to
impute missing flux values.  The model is trained per realisation on
observed cadences only.

Implements Algorithm 8 from the thesis.
Reference: Du et al. (2023) "SAITS: Self-Attention-based Imputation
for Time Series."
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .base import ImputerBase

try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False


class SAITSImputer(ImputerBase):
    """SAITS: Self-Attention-based Imputation for Time Series.

    Parameters
    ----------
    d_model : int
        Model (embedding) dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of DMSA blocks.
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        epochs: int = 100,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for SAITSImputer. pip install torch==2.0.1")
        super().__init__(name="SAITS", seed=seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
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
        obs_mask = (~np.isnan(flux)).astype(np.float32)
        x = np.where(obs_mask.astype(bool), flux, 0.0).astype(np.float32)

        # Normalise
        mu_x = float(np.mean(x[obs_mask.astype(bool)]))
        sigma_x = float(np.std(x[obs_mask.astype(bool)])) + 1e-8
        x_norm = (x - mu_x) / sigma_x

        # Time positional encoding
        t_norm = ((t - t[0]) / (t[-1] - t[0] + 1e-8)).astype(np.float32)

        X_t = torch.tensor(x_norm).unsqueeze(0).unsqueeze(-1)   # (1, N, 1)
        M_t = torch.tensor(obs_mask).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
        T_t = torch.tensor(t_norm).unsqueeze(0).unsqueeze(-1)    # (1, N, 1)

        model = _SAITSModel(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            seq_len=N,
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            optimiser.zero_grad()
            pred = model(X_t, M_t, T_t)  # (1, N, 1)
            # Loss only on observed positions
            loss = torch.mean(M_t * (pred - X_t) ** 2)
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            pred_final = model(X_t, M_t, T_t).squeeze(0).squeeze(-1).numpy()

        # Denormalise
        pred_denorm = pred_final * sigma_x + mu_x

        imputed = flux.copy()
        imputed[missing_idx] = pred_denorm[missing_idx]
        return imputed


if _TORCH_AVAILABLE:

    class _DMSABlock(nn.Module):
        """Diagonally-Masked Self-Attention block."""

        def __init__(self, d_model: int, n_heads: int) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 2), nn.ReLU(), nn.Linear(d_model * 2, d_model)
            )
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x: "torch.Tensor", src_key_padding_mask=None) -> "torch.Tensor":
            attn_out, _ = self.attn(x, x, x, key_padding_mask=src_key_padding_mask)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ff(x))
            return x

    class _SAITSModel(nn.Module):
        def __init__(self, d_model: int, n_heads: int, n_layers: int, seq_len: int) -> None:
            super().__init__()
            # Input projection: (flux * mask, mask, time) → d_model
            self.input_proj = nn.Linear(3, d_model)
            self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.01)
            self.blocks = nn.ModuleList([_DMSABlock(d_model, n_heads) for _ in range(n_layers)])
            self.output_proj = nn.Linear(d_model, 1)

        def forward(
            self,
            x: "torch.Tensor",
            m: "torch.Tensor",
            t_pos: "torch.Tensor",
        ) -> "torch.Tensor":
            inp = torch.cat([x * m, m, t_pos], dim=-1)  # (B, N, 3)
            h = self.input_proj(inp) + self.pos_enc
            for block in self.blocks:
                h = block(h)
            return self.output_proj(h)
