"""
Gaussian Process imputation with a Matérn-3/2 kernel.

Implements Algorithm 4 (Gaussian Process Imputation with Matérn-3/2 Kernel)
from the thesis.

The GP model:
    f(t) ~ GP(μ, k_{3/2}(t, t'))

where:
    μ  = mean of observed fluxes
    k_{3/2}(t, t') = σ_f² (1 + √3 |t-t'|/ℓ) exp(-√3 |t-t'|/ℓ)  [Matérn-3/2]

Hyperparameters (log σ_f, log ℓ, log σ_n) are optimised per light curve by
maximising the log marginal likelihood (Equation 3.8) using L-BFGS-B with
R=20 random restarts.

Requires: george (pip install george)
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.optimize import minimize

from .base import BaseImputer

logger = logging.getLogger(__name__)


class GPMatern32Imputer(BaseImputer):
    """
    GP regression imputer with a Matérn-3/2 kernel + white noise jitter.

    Parameters
    ----------
    n_restarts : int
        Number of L-BFGS-B optimisation restarts (default 20).
    seed : int
        Random seed for restart initialisation.
    """

    def __init__(self, n_restarts: int = 20, seed: int = 42):
        super().__init__(seed=seed)
        self.n_restarts = n_restarts

    def impute(
        self,
        flux: np.ndarray,
        mask: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        try:
            import george
            from george import kernels
        except ImportError:
            raise ImportError("george is required: pip install george")

        out = flux.copy().astype(float)
        t_obs = time[mask]
        f_obs = out[mask]

        if len(t_obs) < 3:
            out[~mask] = np.nanmean(f_obs) if len(f_obs) > 0 else 0.0
            return out

        # Subtract mean for numerical stability
        mu = float(np.mean(f_obs))
        f_obs_c = f_obs - mu

        # Initial hyperparameter guesses (log-space)
        log_sf2_init = np.log(np.var(f_obs_c) + 1e-10)
        dt = np.median(np.diff(np.sort(t_obs)))
        log_ell_init = np.log(20.0 * dt)
        log_sn2_init = np.log(1e-4 * np.var(f_obs_c) + 1e-12)

        best_nll = np.inf
        best_params = np.array([log_sf2_init, log_ell_init, log_sn2_init])
        rng = np.random.default_rng(self.seed)

        def build_gp(params):
            log_sf2, log_ell, log_sn2 = params
            amp2 = float(np.exp(log_sf2))
            ell  = float(np.exp(log_ell))
            sn2  = float(np.exp(log_sn2))
            kernel = amp2 * kernels.Matern32Kernel(metric=ell ** 2)
            gp = george.GP(kernel, mean=0.0, fit_mean=False,
                           white_noise=np.log(sn2 + 1e-12), fit_white_noise=True)
            return gp

        def neg_log_ml(params):
            try:
                gp = build_gp(params)
                gp.compute(t_obs)
                return -gp.log_likelihood(f_obs_c)
            except Exception:
                return 1e20

        # Optimise with random restarts
        for r in range(self.n_restarts):
            if r == 0:
                p0 = best_params.copy()
            else:
                p0 = best_params + 0.5 * rng.standard_normal(3)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = minimize(
                    neg_log_ml, p0,
                    method="L-BFGS-B",
                    options={"maxiter": 200, "ftol": 1e-6},
                )
            if res.success and res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x

        # Predictive mean at missing epochs
        t_miss = time[~mask]
        try:
            gp = build_gp(best_params)
            gp.compute(t_obs)
            pred_mean, pred_var = gp.predict(f_obs_c, t_miss, return_var=True)
            out[~mask] = pred_mean + mu
        except Exception as exc:
            logger.warning("GP prediction failed (%s); falling back to mean-fill.", exc)
            out[~mask] = mu

        return out

    @property
    def name(self) -> str:
        return "GP_Matern32"
