"""Tests for the missing-data injection module."""

import numpy as np
import pytest

from src.missingness.injector import inject_gaps


@pytest.fixture
def sample_flux():
    rng = np.random.default_rng(0)
    return rng.standard_normal(500) + 1.0


class TestInjectGaps:
    def test_output_shapes(self, sample_flux):
        gapped, missing_idx, true_vals = inject_gaps(sample_flux, p=0.20, seed=0)
        assert gapped.shape == sample_flux.shape
        assert len(missing_idx) == len(true_vals)

    def test_fraction_approximate(self, sample_flux):
        """Actual missing fraction should be close to target."""
        p = 0.30
        gapped, missing_idx, _ = inject_gaps(sample_flux, p=p, seed=0)
        actual = np.isnan(gapped).sum() / len(gapped)
        assert abs(actual - p) < 0.05

    def test_nan_at_missing_positions(self, sample_flux):
        gapped, missing_idx, _ = inject_gaps(sample_flux, p=0.20, seed=0)
        assert np.all(np.isnan(gapped[missing_idx]))

    def test_observed_not_nan(self, sample_flux):
        gapped, missing_idx, _ = inject_gaps(sample_flux, p=0.20, seed=0)
        obs_mask = ~np.isnan(gapped)
        assert np.all(np.isfinite(gapped[obs_mask]))

    def test_true_vals_match_original(self, sample_flux):
        gapped, missing_idx, true_vals = inject_gaps(sample_flux, p=0.20, seed=0)
        np.testing.assert_array_equal(true_vals, sample_flux[missing_idx])

    def test_reproducibility(self, sample_flux):
        _, idx1, vals1 = inject_gaps(sample_flux, p=0.30, seed=42)
        _, idx2, vals2 = inject_gaps(sample_flux, p=0.30, seed=42)
        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_array_equal(vals1, vals2)

    def test_different_seeds_differ(self, sample_flux):
        _, idx1, _ = inject_gaps(sample_flux, p=0.30, seed=1)
        _, idx2, _ = inject_gaps(sample_flux, p=0.30, seed=2)
        assert not np.array_equal(idx1, idx2)

    def test_missing_idx_sorted(self, sample_flux):
        _, missing_idx, _ = inject_gaps(sample_flux, p=0.20, seed=0)
        assert list(missing_idx) == sorted(missing_idx)

    def test_invalid_fraction_raises(self, sample_flux):
        with pytest.raises(ValueError):
            inject_gaps(sample_flux, p=1.5, seed=0)
        with pytest.raises(ValueError):
            inject_gaps(sample_flux, p=0.0, seed=0)

    def test_fractions(self, sample_flux):
        for p in [0.10, 0.30, 0.50]:
            gapped, _, _ = inject_gaps(sample_flux, p=p, seed=0)
            actual = np.isnan(gapped).sum() / len(gapped)
            assert abs(actual - p) < 0.06
