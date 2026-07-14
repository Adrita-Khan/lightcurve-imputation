"""
Gaussian Process imputation with the Matérn-3/2 kernel.

Implements Appendix C (GP code) from the thesis.  Hyperparameters
(log-amplitude, log-length-scale) are optimised per realisation via
marginal likelihood maximisation using L-BFGS-B with multiple random
restarts to escape local optima.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from .base import ImputerBase

# george is an optional dependency; import lazily to allow unit tests
# to mock it without requiring a full george installation.
try:
    import george
    from george import kernels
    from scipy.optimize import minimize as scipy_minimize

    _GEORGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GEORGE_AVAILABLE = False


class GPMaternImputer(ImputerBase):
    """Gaussian Process regression with a Matérn-3/2 kernel.

    Hyperparameters are optimised by maximising the log marginal likelihood
    with ``n_restarts`` L-BFGS-B runs.  The posterior predictive mean is
    used as the imputed value.

    Parameters
    ----------
    n_restarts : int
        Number of hyperparameter optimisation restarts.
    seed : int or None
        Random seed for restart initialisation.
    """

    def __init__(self, n_restarts: int = 20, seed: Optional[int] = None) -> None:
        if not _GEORGE_AVAILABLE:
            raise ImportError(
                "The 'george' package is required for GPMaternImputer. "
                "Install it with: pip install george==0.4.0"
            )
        super().__init__(name="GP-Matern32", seed=seed)
        self.n_restarts = n_restarts

    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        obs_mask = ~np.isnan(flux)
        t_obs = t[obs_mask]
        f_obs = flux[obs_mask]

        # Noise level: use known sigma_eps if available; otherwise estimate from data
        sigma_obs = float(np.std(np.diff(f_obs))) / np.sqrt(2)
        sigma_obs = max(sigma_obs, 1e-6)

        f_pred, _ = _gp_predict(
            t_obs=t_obs,
            f_obs=f_obs,
            sigma_obs=sigma_obs,
            t_pred=t[missing_idx],
            n_restarts=self.n_restarts,
            seed=self.seed,
        )

        imputed = flux.copy()
        imputed[missing_idx] = f_pred
        return imputed

    def impute_with_uncertainty(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return imputed values together with per-cadence posterior variances.

        Used for GP uncertainty calibration (Table 4.2 in thesis).

        Returns
        -------
        imputed : np.ndarray, shape (N,)
        variance : np.ndarray, shape (len(missing_idx),)
        """
        obs_mask = ~np.isnan(flux)
        t_obs = t[obs_mask]
        f_obs = flux[obs_mask]
        sigma_obs = float(np.std(np.diff(f_obs))) / np.sqrt(2)
        sigma_obs = max(sigma_obs, 1e-6)

        f_pred, f_var = _gp_predict(
            t_obs=t_obs,
            f_obs=f_obs,
            sigma_obs=sigma_obs,
            t_pred=t[missing_idx],
            n_restarts=self.n_restarts,
            seed=self.seed,
        )
        imputed = flux.copy()
        imputed[missing_idx] = f_pred
        return imputed, f_var


def _gp_predict(
    t_obs: np.ndarray,
    f_obs: np.ndarray,
    sigma_obs: float,
    t_pred: np.ndarray,
    n_restarts: int,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Optimise GP hyperparameters and return posterior predictive mean + variance."""
    rng = np.random.default_rng(seed)

    log_amp0 = np.log(np.var(f_obs) + 1e-10)
    log_len0 = np.log(max(np.median(np.diff(t_obs)) * 20, 1e-4))

    kernel = np.exp(log_amp0) * kernels.Matern32Kernel(np.exp(log_len0))
    gp = george.GP(kernel, mean=float(np.mean(f_obs)), fit_mean=False)
    gp.compute(t_obs, sigma_obs)

    best_nll = np.inf
    best_params = gp.get_parameter_vector()

    def nll(params):
        gp.set_parameter_vector(params)
        try:
            ll = gp.log_likelihood(f_obs, quiet=True)
        except Exception:
            return 1e10, np.zeros_like(params)
        return -ll

    def grad_nll(params):
        gp.set_parameter_vector(params)
        try:
            g = -gp.grad_log_likelihood(f_obs, quiet=True)
        except Exception:
            return np.zeros_like(params)
        return g

    for _ in range(n_restarts):
        p0 = best_params + 0.5 * rng.standard_normal(len(best_params))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = scipy_minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
        if res.fun < best_nll:
            best_nll = res.fun
            best_params = res.x

    gp.set_parameter_vector(best_params)
    f_pred, cov = gp.predict(f_obs, t_pred, return_cov=True)
    f_var = np.diag(cov)

    return f_pred, f_var
