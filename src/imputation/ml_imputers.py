"""
Machine-Learning-Based Imputation Methods (Algorithms 6–12).

All seven methods are leakage-free under the frozen-classifier protocol:
each imputer is trained on the *observed cadences of the current light
curve only* and never accesses withheld ground-truth values or the
classifier's training partition.

Methods implemented:
  KNNImputer       — Algorithm 6 (KNN-Impute)
  RFImputer        — Algorithm 7 (RF-Impute)
  RNNImputer       — Algorithm 8 (RNN-Impute; Bidirectional LSTM)
  GAINImputer      — Algorithm 9 (GAIN-Impute; Generative Adversarial)
  MFImputer        — Algorithm 10 (MF-Impute; Hankel Low-Rank Factorisation)
  GBMICEImputer    — Algorithm 11 (GB-MICE; XGBoost inside MICE)
  SAITSImputer     — Algorithm 12 (SAITS; Diagonally-Masked Self-Attention)

Requires: scikit-learn, xgboost, torch
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from .base import BaseImputer
from .ts_mice import build_lag_phase_matrix

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# KNN-Impute (Algorithm 6)
# ---------------------------------------------------------------------------

class KNNImputer(BaseImputer):
    """
    K-Nearest-Neighbour imputation via distance-weighted average of the k=5
    most contextually similar observed cadences (context half-width W=10).

    Equations (3.16)–(3.18) in the thesis.
    """

    def __init__(self, n_neighbors: int = 5, context_half_width: int = 10, seed: int = 42):
        super().__init__(seed=seed)
        self.n_neighbors = n_neighbors
        self.context_half_width = context_half_width

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        from sklearn.impute import KNNImputer as SklearnKNN

        W = self.context_half_width
        X, _ = build_lag_phase_matrix(flux, time, lag_order=W)
        # Target column is the W-th column (lag-0)
        X_target_col = W

        imp = SklearnKNN(n_neighbors=self.n_neighbors, weights="distance")
        X_imp = imp.fit_transform(X)

        out = flux.copy().astype(float)
        out[~mask] = X_imp[~mask, X_target_col]
        return out

    @property
    def name(self) -> str:
        return "KNN_Impute"


# ---------------------------------------------------------------------------
# RF-Impute (Algorithm 7)
# ---------------------------------------------------------------------------

class RFImputer(BaseImputer):
    """
    Random Forest imputation — single-pass; trains on observed cadences,
    predicts at missing cadences.

    Equations (3.19)–(3.20) in the thesis.
    """

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        super().__init__(seed=seed)
        self.n_estimators = n_estimators

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        from sklearn.ensemble import RandomForestRegressor

        X, _ = build_lag_phase_matrix(flux, time, lag_order=5)
        target_col = 5  # lag-0 index

        # Build training set: rows where all predictors and target are observed
        predictor_cols = [c for c in range(X.shape[1]) if c != target_col]
        X_pred = X[:, predictor_cols]
        y = flux.copy()

        # Drop rows with any NaN in predictors or target
        train_mask = mask & np.all(np.isfinite(X_pred), axis=1)
        if train_mask.sum() < 2:
            out = flux.copy().astype(float)
            out[~mask] = np.nanmean(flux[mask])
            return out

        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        rf.fit(X_pred[train_mask], y[train_mask])

        # Replace NaN predictors in missing rows with mean for prediction
        X_miss = X_pred[~mask].copy()
        col_means = np.nanmean(X_miss, axis=0)
        for col in range(X_miss.shape[1]):
            nan_rows = np.isnan(X_miss[:, col])
            X_miss[nan_rows, col] = col_means[col] if np.isfinite(col_means[col]) else 0.0

        out = flux.copy().astype(float)
        out[~mask] = rf.predict(X_miss)
        return out

    @property
    def name(self) -> str:
        return "RF_Impute"


# ---------------------------------------------------------------------------
# RNN-Impute (Algorithm 8) — Bidirectional LSTM
# ---------------------------------------------------------------------------

class RNNImputer(BaseImputer):
    """
    Bidirectional LSTM imputation.

    Input triple at each cadence: (flux * mask, mask, Δt).
    Trained on observed cadences only (observed-MSE loss).

    Equations (3.21)–(3.25) in the thesis.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        n_epochs: int = 50,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        device = torch.device("cpu")

        N = len(flux)
        flux_filled = flux.copy().astype(float)
        flux_filled[~mask] = 0.0  # NaN → 0 via mask

        dt = np.zeros(N)
        dt[1:] = time[1:] - time[:-1]

        # Input: (flux*mask, mask, Δt) — shape (1, N, 3)
        X = np.stack([flux_filled * mask.astype(float),
                       mask.astype(float),
                       dt], axis=1)
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)  # (1,N,3)
        m_tensor  = torch.tensor(mask, dtype=torch.float32, device=device)
        f_tensor  = torch.tensor(flux_filled * mask.astype(float), dtype=torch.float32, device=device)

        # Build model
        bilstm = nn.LSTM(input_size=3, hidden_size=self.hidden_size,
                         batch_first=True, bidirectional=True)
        linear = nn.Linear(self.hidden_size * 2, 1)
        params = list(bilstm.parameters()) + list(linear.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Training loop
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            h, _ = bilstm(X_tensor)          # (1, N, 2H)
            pred = linear(h).squeeze(-1).squeeze(0)  # (N,)
            obs_pred = pred * m_tensor
            obs_true = f_tensor
            loss = ((obs_pred - obs_true) ** 2 * m_tensor).sum() / (m_tensor.sum() + 1e-8)
            loss.backward()
            optimizer.step()

        # Predict at all positions
        with torch.no_grad():
            h, _ = bilstm(X_tensor)
            pred = linear(h).squeeze(-1).squeeze(0).numpy()

        out = flux.copy().astype(float)
        out[~mask] = pred[~mask]
        return out

    @property
    def name(self) -> str:
        return "RNN_Impute"


# ---------------------------------------------------------------------------
# GAIN-Impute (Algorithm 9) — Generative Adversarial Imputation Network
# ---------------------------------------------------------------------------

class GAINImputer(BaseImputer):
    """
    GAIN: Generative Adversarial Imputation Network (Yoon et al. 2018),
    adapted for per-object light-curve imputation.

    Equations (3.26)–(3.32) in the thesis.
    """

    def __init__(
        self,
        hidden_dims: List[int] | None = None,
        hint_rate: float = 0.1,
        lambda_recon: float = 10.0,
        n_epochs: int = 200,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.hidden_dims = hidden_dims or [32, 64, 32]
        self.hint_rate = hint_rate
        self.lambda_recon = lambda_recon
        self.n_epochs = n_epochs
        self.lr = lr

    def _build_network(self, in_dim: int, hidden_dims: List[int], out_dim: int):
        import torch.nn as nn
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim), nn.Sigmoid()]
        return nn.Sequential(*layers)

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        N = len(flux)
        f0 = flux.copy().astype(float)
        f0[~mask] = 0.0

        m_np = mask.astype(float)
        f0_t = torch.tensor(f0, dtype=torch.float32)
        m_t  = torch.tensor(m_np, dtype=torch.float32)

        # Networks: input is (flux, mask) concatenated → size 2N
        # But for sequential data we treat each cadence independently:
        # We pass the full vector at once (single sample).
        in_dim = 2 * N
        G = self._build_network(in_dim, self.hidden_dims, N)
        D = self._build_network(in_dim, self.hidden_dims, N)
        opt_G = torch.optim.Adam(G.parameters(), lr=self.lr)
        opt_D = torch.optim.Adam(D.parameters(), lr=self.lr)

        for epoch in range(self.n_epochs):
            # Sample noise for missing positions
            z = torch.randn(N)
            z_input = m_t * f0_t + (1 - m_t) * z  # Eq. (3.26)

            # Hint vector: reveal mask with prob (1 - hint_rate)
            h_np = m_np.copy()
            hint_mask = rng.random(N) < self.hint_rate
            h_np[hint_mask] = 0.5
            h_t = torch.tensor(h_np, dtype=torch.float32)

            # Generator forward pass
            gin = torch.cat([z_input, m_t])           # (2N,)
            g_out = G(gin)                             # (N,) in [0,1]
            f_completed = m_t * f0_t + (1 - m_t) * g_out  # Eq. (3.28)

            # Discriminator input
            din = torch.cat([f_completed.detach(), h_t])
            d_out = D(din)                             # (N,)

            # Discriminator loss (Eq. 3.29)
            loss_D = -(m_t * torch.log(d_out + 1e-8) +
                       (1 - m_t) * torch.log(1 - d_out + 1e-8)).mean()
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # Generator loss (Eq. 3.30)
            din_g = torch.cat([f_completed, h_t])
            d_out_g = D(din_g)
            adv_loss  = -(1 - m_t) * torch.log(d_out_g + 1e-8)
            recon_loss = (m_t * (g_out - f0_t) ** 2)
            loss_G = adv_loss.mean() + self.lambda_recon * recon_loss.mean()
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        # Final deterministic imputation (zero noise) — Eq. (3.31)
        with torch.no_grad():
            gin_final = torch.cat([f0_t, m_t])
            g_final = G(gin_final).numpy()

        out = flux.copy().astype(float)
        out[~mask] = g_final[~mask]
        return out

    @property
    def name(self) -> str:
        return "GAIN_Impute"


# ---------------------------------------------------------------------------
# MF-Impute (Algorithm 10) — Hankel Low-Rank Matrix Factorisation
# ---------------------------------------------------------------------------

class MFImputer(BaseImputer):
    """
    Low-Rank Hankel Matrix Factorisation imputation.

    Embeds the flux vector into a Hankel matrix, solves a rank-r matrix
    completion problem via Alternating Least Squares (ALS), and recovers
    the flux by anti-diagonal averaging.

    Equations (3.33)–(3.38) in the thesis.
    """

    def __init__(
        self,
        rank: int = 10,
        alpha: float = 1e-3,
        tol: float = 1e-4,
        max_iter: int = 200,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.rank = rank
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        N = len(flux)
        D = N // 2
        nc = N - D + 1

        # Build Hankel matrix
        H = np.full((D, nc), np.nan)
        for j in range(D):
            for k in range(nc):
                idx = j + k
                if idx < N:
                    H[j, k] = flux[idx]

        # Observed Hankel entries
        obs_jk = [(j, k) for j in range(D) for k in range(nc)
                   if not np.isnan(H[j, k])]
        obs_j = np.array([p[0] for p in obs_jk])
        obs_k = np.array([p[1] for p in obs_jk])
        h_obs = H[obs_j, obs_k]

        # Warm-start: truncated SVD of mean-filled Hankel
        rng = np.random.default_rng(self.seed)
        H0 = H.copy()
        col_means = np.nanmean(H0, axis=0)
        nan_mask = np.isnan(H0)
        H0[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        H0 = np.nan_to_num(H0, nan=0.0)

        U_init, s_init, Vt_init = np.linalg.svd(H0, full_matrices=False)
        r = min(self.rank, len(s_init))
        U = U_init[:, :r] * np.sqrt(s_init[:r])
        V = Vt_init[:r, :].T * np.sqrt(s_init[:r])

        # ALS optimisation
        prev_loss = np.inf
        for _ in range(self.max_iter):
            # Update U (fix V)
            for j in range(D):
                idx_k = np.where(obs_j == j)[0]
                if len(idx_k) == 0:
                    continue
                Vj = V[obs_k[idx_k], :]        # (nobs, r)
                hj = h_obs[idx_k]               # (nobs,)
                A  = Vj.T @ Vj + self.alpha * np.eye(r)
                b  = Vj.T @ hj
                U[j, :] = np.linalg.solve(A, b)

            # Update V (fix U)
            for k in range(nc):
                idx_j = np.where(obs_k == k)[0]
                if len(idx_j) == 0:
                    continue
                Uk = U[obs_j[idx_j], :]
                hk = h_obs[idx_j]
                A  = Uk.T @ Uk + self.alpha * np.eye(r)
                b  = Uk.T @ hk
                V[k, :] = np.linalg.solve(A, b)

            # Convergence check
            H_hat = U @ V.T
            resid = H_hat[obs_j, obs_k] - h_obs
            loss = float(np.dot(resid, resid) +
                         self.alpha * (np.sum(U ** 2) + np.sum(V ** 2)))
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

        H_completed = U @ V.T

        # Anti-diagonal averaging — Eq. (3.37)
        f_rec = np.zeros(N)
        counts = np.zeros(N)
        for j in range(D):
            for k in range(nc):
                i = j + k
                if i < N:
                    f_rec[i]   += H_completed[j, k]
                    counts[i]  += 1
        counts = np.maximum(counts, 1)
        f_rec /= counts

        out = flux.copy().astype(float)
        out[~mask] = f_rec[~mask]
        return out

    @property
    def name(self) -> str:
        return "MF_Impute"


# ---------------------------------------------------------------------------
# GB-MICE (Algorithm 11) — Gradient-Boosting MICE
# ---------------------------------------------------------------------------

class GBMICEImputer(BaseImputer):
    """
    Gradient-Boosting MICE: replaces Bayesian Ridge with XGBoost inside
    the IterativeImputer chaining framework.

    Equations (3.39)–(3.41) in the thesis.
    """

    def __init__(
        self,
        lag_order: int = 5,
        n_chains: int = 5,
        n_iter: int = 10,
        n_estimators: int = 100,
        max_depth: int = 4,
        xgb_lr: float = 0.1,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.lag_order = lag_order
        self.n_chains = n_chains
        self.n_iter = n_iter
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.xgb_lr = xgb_lr

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from xgboost import XGBRegressor

        X, _ = build_lag_phase_matrix(flux, time, lag_order=self.lag_order)
        target_col = self.lag_order

        chain_preds = []
        for c in range(self.n_chains):
            estimator = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.xgb_lr,
                random_state=self.seed + c,
                verbosity=0,
                n_jobs=-1,
            )
            imp = IterativeImputer(
                estimator=estimator,
                max_iter=self.n_iter,
                random_state=self.seed + c,
                initial_strategy="mean",
            )
            X_imp = imp.fit_transform(X)
            chain_preds.append(X_imp[:, target_col])

        f_imputed = np.mean(np.stack(chain_preds, axis=0), axis=0)

        out = flux.copy().astype(float)
        out[~mask] = f_imputed[~mask]
        return out

    @property
    def name(self) -> str:
        return "GB_MICE"


# ---------------------------------------------------------------------------
# SAITS (Algorithm 12) — Self-Attention Imputation for Time Series
# ---------------------------------------------------------------------------

class SAITSImputer(BaseImputer):
    """
    Self-Attention-based Imputation for Time Series (Du et al. 2023).

    Two stacked Diagonally-Masked Multi-Head Self-Attention (DMSA) blocks.
    Trained per light curve on observed cadences only.

    Equations (3.42)–(3.49) in the thesis.
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        alpha_s: float = 0.3,
        n_epochs: int = 100,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        super().__init__(seed=seed)
        self.d_model = d_model
        self.n_heads = n_heads
        self.alpha_s = alpha_s
        self.n_epochs = n_epochs
        self.lr = lr

    def impute(self, flux: np.ndarray, mask: np.ndarray, time: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        N = len(flux)

        f0 = flux.copy().astype(float)
        f0[~mask] = 0.0
        m_np = mask.astype(float)

        # Input: (flux*mask, mask) — (N, 2)
        V_np = np.stack([f0 * m_np, m_np], axis=1)   # (N, 2)
        V_t = torch.tensor(V_np, dtype=torch.float32)
        m_t = torch.tensor(m_np, dtype=torch.float32)
        f_t = torch.tensor(f0 * m_np, dtype=torch.float32)

        # Learnable positional encoding
        P = nn.Parameter(torch.randn(N, self.d_model) * 0.01)

        # Input projection
        W_in1 = nn.Linear(2, self.d_model, bias=False)
        W_in2 = nn.Linear(2, self.d_model, bias=False)

        # Two DMSA blocks (simplified as nn.MultiheadAttention)
        attn1 = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)
        attn2 = nn.MultiheadAttention(self.d_model, self.n_heads, batch_first=True)

        # Output heads
        W_out1 = nn.Linear(self.d_model, 1, bias=True)
        W_out2 = nn.Linear(self.d_model, 1, bias=True)

        params = (list(W_in1.parameters()) + list(W_in2.parameters()) +
                  list(attn1.parameters()) + list(attn2.parameters()) +
                  list(W_out1.parameters()) + list(W_out2.parameters()) + [P])
        optimizer = torch.optim.Adam(params, lr=self.lr)

        # Diagonal attention mask (−∞ on diagonal → forces cross-position attention)
        diag_mask = torch.zeros(N, N)
        diag_mask.fill_diagonal_(float("-inf"))

        for _ in range(self.n_epochs):
            optimizer.zero_grad()

            # Block 1
            Z0 = W_in1(V_t) + P                     # (N, d_model)
            Z0_b = Z0.unsqueeze(0)                   # (1, N, d_model)
            Z1, _ = attn1(Z0_b, Z0_b, Z0_b, attn_mask=diag_mask)
            X1 = W_out1(Z1.squeeze(0)).squeeze(-1)   # (N,)

            # Combine block-1 with observations
            X1_comb = m_t * f_t + (1 - m_t) * X1
            V2_np = torch.stack([X1_comb, m_t], dim=1)   # (N, 2)
            Z1b = W_in2(V2_np) + P
            Z1b_b = Z1b.unsqueeze(0)
            Z2, _ = attn2(Z1b_b, Z1b_b, Z1b_b, attn_mask=diag_mask)
            X2 = W_out2(Z2.squeeze(0)).squeeze(-1)   # (N,)

            # Observed-only loss (Eq. 3.48)
            l1 = ((X1 - f_t) ** 2 * m_t).sum() / (m_t.sum() + 1e-8)
            l2 = ((X2 - f_t) ** 2 * m_t).sum() / (m_t.sum() + 1e-8)
            loss = self.alpha_s * l1 + (1 - self.alpha_s) * l2
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            Z0 = W_in1(V_t) + P
            Z1, _ = attn1(Z0.unsqueeze(0), Z0.unsqueeze(0), Z0.unsqueeze(0),
                          attn_mask=diag_mask)
            X1 = W_out1(Z1.squeeze(0)).squeeze(-1)
            X1c = m_t * f_t + (1 - m_t) * X1
            V2 = torch.stack([X1c, m_t], dim=1)
            Z1b = W_in2(V2) + P
            Z2, _ = attn2(Z1b.unsqueeze(0), Z1b.unsqueeze(0), Z1b.unsqueeze(0),
                          attn_mask=diag_mask)
            X2 = W_out2(Z2.squeeze(0)).squeeze(-1).numpy()

        out = flux.copy().astype(float)
        out[~mask] = X2[~mask]
        return out

    @property
    def name(self) -> str:
        return "SAITS"
