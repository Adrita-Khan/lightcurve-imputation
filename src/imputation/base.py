"""
Base class for all imputation methods.

Every imputer implements a single `impute(flux, mask, time)` method that
receives the gapped flux vector (NaN at missing positions), the binary
observed mask, and the time vector, and returns a fully imputed flux vector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseImputer(ABC):
    """
    Abstract base class for light-curve imputation.

    Subclasses must override `impute`.  The signature is intentionally
    minimal so all thirteen methods share the same interface.
    """

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed

    @abstractmethod
    def impute(
        self,
        flux: np.ndarray,
        mask: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        flux : np.ndarray, shape (N,)
            Flux vector with NaN at missing positions (mask == False).
        mask : np.ndarray of bool, shape (N,)
            True at observed positions.
        time : np.ndarray, shape (N,)
            Cadence timestamps (Kepler BKJD or similar).

        Returns
        -------
        np.ndarray, shape (N,)
            Fully imputed flux vector (no NaN).
        """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(seed={self.seed})"
