"""
Imputer registry: map config keys to imputer classes with hyperparameters.

Usage::

    imputer = get_imputer("gp_matern", params={"n_restarts": 20}, seed=42)
"""

from __future__ import annotations

from typing import Any, Optional

from .deterministic import (
    ForwardFillImputer,
    LinearInterpImputer,
    MeanFillImputer,
    SplineInterpImputer,
)
from .gain_imputer import GAINImputer
from .gb_mice import GBMICEImputer
from .gp_imputer import GPMaternImputer
from .knn_imputer import KNNImputer
from .mf_imputer import MFImputer
from .rf_imputer import RFImputer
from .rnn_imputer import RNNImputer
from .saits_imputer import SAITSImputer
from .ts_mice import TSMICEImputer

_REGISTRY: dict[str, type] = {
    "mean_fill": MeanFillImputer,
    "forward_fill": ForwardFillImputer,
    "linear_interp": LinearInterpImputer,
    "spline_interp": SplineInterpImputer,
    "gp_matern": GPMaternImputer,
    "ts_mice": TSMICEImputer,
    "knn_impute": KNNImputer,
    "rf_impute": RFImputer,
    "rnn_impute": RNNImputer,
    "gain_impute": GAINImputer,
    "mf_impute": MFImputer,
    "gb_mice": GBMICEImputer,
    "saits": SAITSImputer,
}

# Param key mapping: config-level param names → constructor argument names
_PARAM_MAP: dict[str, dict[str, str]] = {
    "gp_matern": {"n_restarts": "n_restarts"},
    "ts_mice": {"L": "L", "n_chains": "n_chains", "max_iter": "max_iter"},
    "knn_impute": {"k": "k", "W": "W"},
    "rf_impute": {"n_estimators": "n_estimators", "L": "L"},
    "rnn_impute": {"hidden_size": "hidden_size", "epochs": "epochs", "lr": "lr"},
    "gain_impute": {
        "hint_rate": "hint_rate",
        "lambda_recon": "lambda_recon",
        "epochs": "epochs",
        "lr": "lr",
    },
    "mf_impute": {"rank": "rank", "alpha": "alpha", "tol": "tol", "max_iter": "max_iter"},
    "gb_mice": {
        "n_estimators": "n_estimators",
        "max_depth": "max_depth",
        "L": "L",
        "n_chains": "n_chains",
        "max_iter": "max_iter",
    },
    "saits": {"d_model": "d_model", "n_heads": "n_heads", "n_layers": "n_layers", "epochs": "epochs", "lr": "lr"},
}


def get_imputer(
    key: str,
    params: Optional[dict[str, Any]] = None,
    seed: Optional[int] = None,
):
    """Instantiate an imputer by its registry key.

    Parameters
    ----------
    key : str
        Registry key (e.g. ``'gp_matern'``, ``'ts_mice'``).
    params : dict or None
        Hyperparameter dict from the YAML config.
    seed : int or None
        Random seed.

    Returns
    -------
    ImputerBase
    """
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown imputer '{key}'. Available: {list(_REGISTRY.keys())}"
        )
    cls = _REGISTRY[key]
    kwargs: dict[str, Any] = {"seed": seed}

    if params:
        pmap = _PARAM_MAP.get(key, {})
        for cfg_name, ctor_name in pmap.items():
            if cfg_name in params:
                kwargs[ctor_name] = params[cfg_name]

    return cls(**kwargs)


def list_imputers() -> list[str]:
    """Return the list of registered imputer keys."""
    return list(_REGISTRY.keys())
