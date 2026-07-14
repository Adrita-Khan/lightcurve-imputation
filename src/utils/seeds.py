"""Deterministic seed sequence generation for reproducible experiments."""

from __future__ import annotations

import numpy as np


def make_seed_sequence(n: int, base_seed: int = 42) -> list[int]:
    """Generate a list of ``n`` deterministic integer seeds.

    Seeds are derived from a ``np.random.SeedSequence`` rooted at
    ``base_seed``, ensuring that the same ``base_seed`` always produces
    the same set of per-realisation seeds regardless of platform.

    Parameters
    ----------
    n : int
        Number of seeds to generate.
    base_seed : int
        Root seed.

    Returns
    -------
    list of int
    """
    ss = np.random.SeedSequence(base_seed)
    return [int(c.generate_state(1)[0]) for c in ss.spawn(n)]
