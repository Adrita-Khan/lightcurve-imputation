"""Unit tests for imputation methods."""

import numpy as np
import pytest
from src.imputation.classical import (
    MeanFillImputer, ForwardFillImputer, LinearImputer, SplineImputer
)
from src.imputation.registry import get_imputer, get_all_imputers, ALL_METHOD_NAMES
from src.data.gap_injection import inject_gaps


def _make_sinusoid(N=300, seed=42):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 15, N)
    f = 1.0 + 0.25 * np.sin(2 * np.pi * t / 2.5) + 0.01 * rng.standard_normal(N)
    return t, f


def _get_gapped(flux, time=None, p=0.3, seed=7):
    if time is None:
        time = np.linspace(0, 15, len(flux))
    gapped, mask, gt = inject_gaps(flux, p=p, seed=seed)
    return gapped, mask, gt, time


class TestClassicalImputers:

    def test_mean_fill_no_nan(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = MeanFillImputer().impute(g, mask, t)
        assert not np.any(np.isnan(out)), "Output should have no NaN"

    def test_mean_fill_shape(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = MeanFillImputer().impute(g, mask, t)
        assert out.shape == f.shape

    def test_mean_fill_observed_unchanged(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = MeanFillImputer().impute(g, mask, t)
        np.testing.assert_array_almost_equal(out[mask], f[mask])

    def test_mean_fill_imputed_is_mean(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = MeanFillImputer().impute(g, mask, t)
        expected_mean = float(np.nanmean(g[mask]))
        np.testing.assert_array_almost_equal(out[~mask], expected_mean)

    def test_forward_fill_no_nan(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = ForwardFillImputer().impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_linear_no_nan(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = LinearImputer().impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_linear_exact_at_observed(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = LinearImputer().impute(g, mask, t)
        np.testing.assert_array_almost_equal(out[mask], f[mask])

    def test_spline_no_nan(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = SplineImputer().impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_spline_exact_at_observed(self):
        t, f = _make_sinusoid()
        g, mask, _, _ = _get_gapped(f)
        out = SplineImputer().impute(g, mask, t)
        np.testing.assert_array_almost_equal(out[mask], f[mask], decimal=5)

    def test_linear_better_than_mean_on_smooth(self):
        """
        Linear interpolation should outperform mean-fill on a high-amplitude,
        noise-free sinusoid at low missingness (only scattered points missing,
        so linear can bridge gaps accurately).
        """
        # Pure sinusoid — no noise — amplitude 0.5 around mean 1.0
        N = 800
        t = np.linspace(0, 30, N)
        f = 1.0 + 0.5 * np.sin(2 * np.pi * t / 5.0)

        # 10% scattered gaps (block_ratio=0) so linear interpolation is easy
        g, mask, gt = inject_gaps(f, p=0.10, seed=7, block_ratio=0.0)
        miss = np.where(~mask)[0]

        out_mean   = MeanFillImputer().impute(g, mask, t)
        out_linear = LinearImputer().impute(g, mask, t)

        rmse_mean   = np.sqrt(np.mean((out_mean[miss]   - gt) ** 2))
        rmse_linear = np.sqrt(np.mean((out_linear[miss] - gt) ** 2))
        assert rmse_linear < rmse_mean, (
            f"Linear RMSE ({rmse_linear:.4f}) should be < mean RMSE ({rmse_mean:.4f}) "
            f"on noise-free sinusoid with scattered gaps"
        )

    @pytest.mark.parametrize("p", [0.1, 0.3, 0.5])
    def test_all_classical_no_nan_at_all_fractions(self, p):
        t, f = _make_sinusoid(N=300)
        g, mask, _, _ = _get_gapped(f, p=p)
        for cls in [MeanFillImputer, ForwardFillImputer, LinearImputer, SplineImputer]:
            out = cls().impute(g, mask, t)
            assert not np.any(np.isnan(out)), f"{cls.__name__} produced NaN at p={p}"


class TestRegistryAndMLImputers:

    def test_registry_all_names(self):
        assert len(ALL_METHOD_NAMES) == 13

    def test_get_imputer_returns_correct_type(self):
        from src.imputation.classical import MeanFillImputer
        imp = get_imputer("Mean_Fill", seed=0)
        assert isinstance(imp, MeanFillImputer)

    def test_get_imputer_unknown_raises(self):
        with pytest.raises(ValueError):
            get_imputer("NonExistentMethod")

    @pytest.mark.parametrize("name", ["Mean_Fill", "Forward_Fill", "Linear", "Spline"])
    def test_classical_imputers_via_registry(self, name):
        t, f = _make_sinusoid(N=200)
        g, mask, gt, _ = _get_gapped(f, p=0.3)
        imp = get_imputer(name, seed=42)
        out = imp.impute(g, mask, t)
        assert out.shape == f.shape
        assert not np.any(np.isnan(out))

    def test_knn_no_nan(self):
        t, f = _make_sinusoid(N=300)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("KNN_Impute", seed=42)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_rf_no_nan(self):
        t, f = _make_sinusoid(N=300)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("RF_Impute", seed=42)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_mf_no_nan(self):
        t, f = _make_sinusoid(N=200)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("MF_Impute", seed=42)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_ts_mice_no_nan(self):
        t, f = _make_sinusoid(N=200)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("TS_MICE", seed=42)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_rnn_no_nan(self):
        pytest.importorskip("torch")
        t, f = _make_sinusoid(N=150)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("RNN_Impute", seed=42, n_epochs=3)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_saits_no_nan(self):
        pytest.importorskip("torch")
        t, f = _make_sinusoid(N=100)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("SAITS", seed=42, n_epochs=3)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_gain_no_nan(self):
        pytest.importorskip("torch")
        t, f = _make_sinusoid(N=100)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("GAIN_Impute", seed=42, n_epochs=5)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_gb_mice_no_nan(self):
        pytest.importorskip("xgboost")
        t, f = _make_sinusoid(N=200)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("GB_MICE", seed=42, n_iter=2, n_chains=2)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))

    def test_gp_no_nan(self):
        pytest.importorskip("george")
        t, f = _make_sinusoid(N=150)
        g, mask, _, _ = _get_gapped(f, p=0.2)
        imp = get_imputer("GP_Matern32", seed=42, n_restarts=2)
        out = imp.impute(g, mask, t)
        assert not np.any(np.isnan(out))
