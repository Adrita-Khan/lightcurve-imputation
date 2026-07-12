"""
Imputer registry — maps method name strings to instantiated imputers.

Usage:
    from src.imputation.registry import get_imputer, ALL_METHOD_NAMES

    imputer = get_imputer("GP_Matern32", seed=7)
    flux_imputed = imputer.impute(flux, mask, time)
"""

from __future__ import annotations

from .classical import MeanFillImputer, ForwardFillImputer, LinearImputer, SplineImputer
from .gp_imputer import GPMatern32Imputer
from .ts_mice    import TSMICEImputer
from .ml_imputers import (
    KNNImputer, RFImputer, RNNImputer,
    GAINImputer, MFImputer, GBMICEImputer, SAITSImputer,
)

# Canonical method names (used in tables and filenames)
ALL_METHOD_NAMES = [
    "Mean_Fill",
    "Forward_Fill",
    "Linear",
    "Spline",
    "GP_Matern32",
    "TS_MICE",
    "KNN_Impute",
    "RF_Impute",
    "RNN_Impute",
    "GAIN_Impute",
    "MF_Impute",
    "GB_MICE",
    "SAITS",
]

_REGISTRY = {
    "Mean_Fill":    MeanFillImputer,
    "Forward_Fill": ForwardFillImputer,
    "Linear":       LinearImputer,
    "Spline":       SplineImputer,
    "GP_Matern32":  GPMatern32Imputer,
    "TS_MICE":      TSMICEImputer,
    "KNN_Impute":   KNNImputer,
    "RF_Impute":    RFImputer,
    "RNN_Impute":   RNNImputer,
    "GAIN_Impute":  GAINImputer,
    "MF_Impute":    MFImputer,
    "GB_MICE":      GBMICEImputer,
    "SAITS":        SAITSImputer,
}


def get_imputer(name: str, seed: int = 42, **kwargs):
    """
    Instantiate an imputer by name.

    Parameters
    ----------
    name : str
        One of ALL_METHOD_NAMES.
    seed : int
        Random seed forwarded to the imputer constructor.
    **kwargs
        Additional hyperparameters forwarded to the constructor.

    Returns
    -------
    BaseImputer
        Instantiated imputer object.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown imputer '{name}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](seed=seed, **kwargs)


def get_all_imputers(seed: int = 42, config: dict | None = None) -> dict:
    """
    Instantiate all thirteen imputers.

    Parameters
    ----------
    seed : int
        Base random seed.
    config : dict | None
        Optional config dict (from experiment.yaml) for hyperparameter overrides.

    Returns
    -------
    dict mapping method name → imputer instance
    """
    cfg = config or {}
    imp_cfg = cfg.get("imputation", {})

    imputers = {}
    for name, cls in _REGISTRY.items():
        method_key = name.lower().replace("-", "_")
        method_cfg = imp_cfg.get(method_key, {})
        try:
            imputers[name] = cls(seed=seed, **method_cfg)
        except TypeError:
            # Some imputers don't accept all kwargs — use defaults
            imputers[name] = cls(seed=seed)

    return imputers
