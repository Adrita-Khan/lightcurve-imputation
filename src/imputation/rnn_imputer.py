"""
RNN-Impute: Bidirectional LSTM flux imputation.

A Bidirectional LSTM with hidden size H=64 processes masked inputs
(f_i * m_i, m_i, Δt_i) following the GRU-D convention.  The model is
trained for E=50 epochs on observed cadences per realisation and then
used to predict at missing positions.

Implements Algorithm 5 (RNN-Impute) from the thesis.
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


class RNNImputer(ImputerBase):
    """Bidirectional LSTM imputation.

    Parameters
    ----------
    hidden_size : int
        Hidden units per direction (total 2*H).
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate.
    seed : int or None
        Random seed.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for RNNImputer. pip install torch==2.0.1")
        super().__init__(name="RNN-Impute", seed=seed)
        self.hidden_size = hidden_size
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
        mask = (~np.isnan(flux)).astype(float)
        dt = np.concatenate([[0.0], np.diff(t)])

        # Masked input following GRU-D convention
        flux_obs = np.where(mask.astype(bool), flux, 0.0)
        U = np.stack([flux_obs, mask, dt], axis=1).astype(np.float32)
        U_tensor = torch.tensor(U).unsqueeze(0)  # (1, N, 3)

        model = _BiLSTMImputer(input_size=3, hidden_size=self.hidden_size)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.lr)

        obs_mask_tensor = torch.tensor(mask, dtype=torch.float32)

        for _ in range(self.epochs):
            optimiser.zero_grad()
            pred = model(U_tensor).squeeze(0).squeeze(-1)  # (N,)
            loss = torch.mean(obs_mask_tensor * (pred - torch.tensor(flux_obs)) ** 2)
            loss.backward()
            optimiser.step()

        with torch.no_grad():
            pred = model(U_tensor).squeeze(0).squeeze(-1).numpy()

        imputed = flux.copy()
        imputed[missing_idx] = pred[missing_idx]
        return imputed


if _TORCH_AVAILABLE:

    class _BiLSTMImputer(nn.Module):
        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.bilstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
            )
            self.readout = nn.Linear(hidden_size * 2, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            out, _ = self.bilstm(x)
            return self.readout(out)
