"""Unit tests for feature extraction and evaluation metrics."""

import numpy as np
import pytest
from src.features.extraction import extract_features, FEATURE_NAMES, N_FEATURES
from src.evaluation.metrics import (
    rmse, mae, accuracy_loss, relative_period_error,
    period_recovery_rate, feature_distortion, bootstrap_ci,
    safe_missingness_threshold,
)


def _sinusoid(N=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20, N)
    f = 1.0 + 0.25 * np.sin(2 * np.pi * t / 3.0) + 0.02 * rng.standard_normal(N)
    return t, f


class TestFeatureExtraction:

    def test_output_shape(self):
        t, f = _sinusoid()
        phi = extract_features(t, f)
        assert phi.shape == (N_FEATURES,), f"Expected ({N_FEATURES},), got {phi.shape}"

    def test_no_nan(self):
        t, f = _sinusoid()
        phi = extract_features(t, f)
        assert not np.any(np.isnan(phi)), "Feature vector should have no NaN"

    def test_no_inf(self):
        t, f = _sinusoid()
        phi = extract_features(t, f)
        assert not np.any(np.isinf(phi)), "Feature vector should have no Inf"

    def test_feature_names_count(self):
        assert len(FEATURE_NAMES) == N_FEATURES == 35

    def test_reproducibility(self):
        t, f = _sinusoid()
        phi1 = extract_features(t, f)
        phi2 = extract_features(t, f)
        np.testing.assert_array_equal(phi1, phi2)

    def test_degenerate_short_flux(self):
        """Should not crash on very short flux vectors."""
        t = np.linspace(0, 1, 3)
        f = np.array([1.0, 1.05, 0.98])
        phi = extract_features(t, f)
        assert phi.shape == (N_FEATURES,)

    def test_constant_flux(self):
        """Constant flux should return zero for most variability indices."""
        t = np.linspace(0, 10, 300)
        f = np.ones(300)
        phi = extract_features(t, f)
        assert phi.shape == (N_FEATURES,)
        assert not np.any(np.isnan(phi))

    def test_ls_period_positive(self):
        """LS period (log10) should be finite for a periodic signal."""
        t, f = _sinusoid()
        phi = extract_features(t, f)
        # phi[0] = log10(period), phi[1] = ls_power
        assert np.isfinite(phi[0])
        assert phi[1] >= 0

    def test_amplitude_positive(self):
        """Phase-folded amplitude should be positive."""
        t, f = _sinusoid()
        phi = extract_features(t, f)
        assert phi[2] >= 0  # phase_amplitude

    def test_std_positive(self):
        t, f = _sinusoid()
        phi = extract_features(t, f)
        assert phi[5] >= 0  # flux_std

    @pytest.mark.parametrize("n", [50, 200, 500, 4000])
    def test_various_lengths(self, n):
        t, f = _sinusoid(N=n)
        phi = extract_features(t, f)
        assert phi.shape == (N_FEATURES,)
        assert not np.any(np.isnan(phi))


class TestMetrics:

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_mae_known_value(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([2.0, 2.0])
        assert mae(y_true, y_pred) == pytest.approx(2.0)

    def test_accuracy_loss_positive(self):
        assert accuracy_loss(0.85, 0.80) == pytest.approx(0.05)

    def test_accuracy_loss_zero(self):
        assert accuracy_loss(0.85, 0.85) == pytest.approx(0.0)

    def test_relative_period_error(self):
        assert relative_period_error(2.5, 2.5) == pytest.approx(0.0)
        assert relative_period_error(2.5, 2.525) == pytest.approx(0.01)

    def test_period_recovery_rate_perfect(self):
        p_true = np.array([2.0, 3.0, 1.5])
        p_imp  = np.array([2.0, 3.0, 1.5])
        assert period_recovery_rate(p_true, p_imp) == pytest.approx(1.0)

    def test_period_recovery_rate_none(self):
        p_true = np.array([2.0, 2.0])
        p_imp  = np.array([4.0, 4.0])  # 100% error
        assert period_recovery_rate(p_true, p_imp) == pytest.approx(0.0)

    def test_feature_distortion_shape(self):
        phi_imp  = np.random.randn(50, 35)
        phi_true = np.random.randn(50, 35)
        dist = feature_distortion(phi_imp, phi_true)
        assert dist.shape == (35,)

    def test_feature_distortion_zero_on_perfect(self):
        phi = np.random.randn(30, 35)
        dist = feature_distortion(phi, phi)
        np.testing.assert_array_almost_equal(dist, 0.0)

    def test_bootstrap_ci_returns_tuple(self):
        values = np.random.randn(30) * 0.05 + 0.85
        lo, hi = bootstrap_ci(values, n_bootstrap=200, seed=0)
        assert lo < hi
        assert lo < np.mean(values) < hi

    def test_bootstrap_ci_coverage(self):
        """95% CI should contain the true mean in most trials."""
        rng = np.random.default_rng(0)
        true_mean = 0.85
        covered = 0
        for _ in range(100):
            vals = rng.normal(true_mean, 0.02, 30)
            lo, hi = bootstrap_ci(vals, n_bootstrap=500, seed=0)
            if lo <= true_mean <= hi:
                covered += 1
        assert covered >= 88, f"Coverage {covered}/100 is below expected ~95"

    def test_safe_threshold_interpolation(self):
        acc_loss = {0.1: 0.02, 0.3: 0.06, 0.5: 0.12}
        p_star = safe_missingness_threshold(acc_loss, threshold=0.05)
        assert 0.1 < p_star < 0.3

    def test_safe_threshold_never_exceeded(self):
        acc_loss = {0.1: 0.01, 0.3: 0.02, 0.5: 0.04}
        p_star = safe_missingness_threshold(acc_loss, threshold=0.05)
        assert not np.isfinite(p_star)

    def test_safe_threshold_exceeded_at_first(self):
        acc_loss = {0.1: 0.08, 0.3: 0.15, 0.5: 0.20}
        p_star = safe_missingness_threshold(acc_loss, threshold=0.05)
        assert 0.0 < p_star <= 0.1
