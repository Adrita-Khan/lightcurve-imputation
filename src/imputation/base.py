"""
Abstract base class defining the common imputer API.

Every imputation method must inherit from ``ImputerBase`` and implement
the ``impute`` method.  The ``fit_impute`` wrapper records wall-clock
time and peak memory usage for the evaluation module.
"""

from __future__ import annotations

import abc
import time
from typing import Optional

import numpy as np
import psutil
import os


class ImputerBase(abc.ABC):
    """Abstract base class for all imputation methods.

    Subclasses must implement :meth:`impute`.

    Parameters
    ----------
    name : str
        Human-readable method name for logging and table generation.
    seed : int or None
        Random seed for stochastic methods.  Deterministic methods ignore it.
    """

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name = name
        self.seed = seed
        self._runtime_s: float = 0.0
        self._memory_mb: float = 0.0

    @abc.abstractmethod
    def impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        """Fill missing values and return the completed flux vector.

        Parameters
        ----------
        t : np.ndarray, shape (N,)
            Time vector (uniform cadence assumed, but not required).
        flux : np.ndarray, shape (N,)
            Flux vector with NaN at ``missing_idx`` positions.
        missing_idx : np.ndarray of int
            Indices of missing cadences.
        period_est : float or None
            Lomb–Scargle period estimate from the *observed* cadences.
            Methods that use phase features should use this value.

        Returns
        -------
        imputed : np.ndarray, shape (N,)
            Flux vector with all NaN values replaced by imputed estimates.
        """

    def fit_impute(
        self,
        t: np.ndarray,
        flux: np.ndarray,
        missing_idx: np.ndarray,
        period_est: Optional[float] = None,
    ) -> np.ndarray:
        """Call :meth:`impute` while recording wall-clock time and memory.

        The recorded values are stored in ``self._runtime_s`` and
        ``self._memory_mb`` and are retrieved by the evaluation harness.

        Parameters
        ----------
        (same as :meth:`impute`)

        Returns
        -------
        imputed : np.ndarray, shape (N,)
        """
        proc = psutil.Process(os.getpid())
        mem_before = proc.memory_info().rss / (1024 ** 2)  # MiB

        t0 = time.perf_counter()
        result = self.impute(t, flux, missing_idx, period_est)
        self._runtime_s = time.perf_counter() - t0

        mem_after = proc.memory_info().rss / (1024 ** 2)
        self._memory_mb = max(0.0, mem_after - mem_before)

        return result

    @property
    def runtime_s(self) -> float:
        """Wall-clock time of the last :meth:`fit_impute` call (seconds)."""
        return self._runtime_s

    @property
    def memory_mb(self) -> float:
        """Approximate peak memory increase of the last :meth:`fit_impute` call (MiB)."""
        return self._memory_mb

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, seed={self.seed})"
