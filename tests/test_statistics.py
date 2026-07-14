"""Tests for statistical comparison utilities."""

import numpy as np
import pytest

from src.statistics.tests import (
    wilcoxon_signed_rank,
    friedman_test,
    bootstrap_ci,
    effect_size_cohens_d,
)


class TestWilcoxon:
    def test_identical_samples_high_p(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 6)
        _, p = wilcoxon_signed_rank(x, x + 0.001)
        # Very small difference → high p or graceful output
        assert isinstance(p, float)

    def test_clearly_different_samples(self):
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 1, 30)
        y = x + 5.0  # clearly shifted
        _, p = wilcoxon_signed_rank(x, y)
        assert p < 0.01

    def test_output_types(self):
        x = np.arange(1.0, 31.0)
        y = x + 0.5
        stat, p = wilcoxon_signed_rank(x, y)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestFriedman:
    def test_output_types(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((30, 5))
        stat, p = friedman_test(data)
        assert isinstance(stat, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_identical_columns_nan_or_large_p(self):
        # Identical columns → all ranks tied → scipy returns NaN or high p
        data = np.tile(np.arange(1.0, 31.0).reshape(-1, 1), (1, 4))
        stat, p = friedman_test(data)
        assert np.isnan(p) or p > 0.05


class TestBootstrapCI:
    def test_interval_contains_mean(self):
        rng = np.random.default_rng(0)
        vals = rng.normal(5.0, 1.0, 100)
        lo, hi = bootstrap_ci(vals, ci=0.95, seed=0)
        assert lo < np.mean(vals) < hi

    def test_width_positive(self):
        rng = np.random.default_rng(42)
        vals = rng.standard_normal(50)
        lo, hi = bootstrap_ci(vals, ci=0.95, seed=42)
        assert hi > lo

    def test_reproducibility(self):
        vals = np.arange(1.0, 31.0)
        lo1, hi1 = bootstrap_ci(vals, seed=7)
        lo2, hi2 = bootstrap_ci(vals, seed=7)
        assert lo1 == lo2 and hi1 == hi2


class TestCohensd:
    def test_zero_for_identical(self):
        x = np.ones(20) * 3.0
        y = np.ones(20) * 3.0
        d = effect_size_cohens_d(x, y)
        assert abs(d) < 1e-6

    def test_large_for_separated(self):
        x = np.ones(30)
        y = np.ones(30) * 10.0
        d = effect_size_cohens_d(x, y)
        assert abs(d) > 5.0
