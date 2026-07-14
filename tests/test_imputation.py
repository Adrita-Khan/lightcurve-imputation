"""Tests for imputation methods."""

import numpy as np
import pytest

from src.simulation.generator import generate_synthetic_lightcurve
from src.missingness.injector import inject_gaps
from src.imputation.deterministic import (
    MeanFillImputer,
    ForwardFillImputer,
    LinearInterpImputer,
    SplineInterpImputer,
)
from src.imputation.ts_mice import TSMICEImputer
from src.imputation.knn_imputer import KNNImputer
from src.imputation.rf_imputer import RFImputer
from src.imputation.mf_imputer import MFImputer
from src.imputation.registry import get_imputer, list_imputers


@pytest.fixture(scope="module")
def signal_fixture():
    t, flux, params = generate_synthetic_lightcurve(N=300, seed=0)
    return t, flux, params


@pytest.fixture(scope="module")
def gapped_fixture(signal_fixture):
    t, flux, _ = signal_fixture
    gapped, missing_idx, true_vals = inject_gaps(flux, p=0.20, seed=1)
    return t, gapped, missing_idx, true_vals


def _check_imputer(imputer, t, gapped, missing_idx):
    """Run an imputer and verify no NaN remain and output shape matches."""
    imputed = imputer.fit_impute(t, gapped, missing_idx, period_est=1.0)
    assert imputed.shape == gapped.shape, "Shape mismatch"
    assert not np.any(np.isnan(imputed)), f"{imputer.name}: NaN in output"
    assert imputer.runtime_s >= 0.0
    return imputed


class TestMeanFill:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        _check_imputer(MeanFillImputer(), t, gapped, missing_idx)

    def test_missing_set_to_mean(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = MeanFillImputer()
        imputed = imp.impute(t, gapped, missing_idx)
        expected_mean = float(np.nanmean(gapped))
        np.testing.assert_allclose(imputed[missing_idx], expected_mean)

    def test_observed_unchanged(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = MeanFillImputer()
        imputed = imp.impute(t, gapped, missing_idx)
        obs_mask = ~np.isnan(gapped)
        np.testing.assert_array_equal(imputed[obs_mask], gapped[obs_mask])


class TestForwardFill:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        _check_imputer(ForwardFillImputer(), t, gapped, missing_idx)

    def test_observed_unchanged(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = ForwardFillImputer()
        imputed = imp.impute(t, gapped, missing_idx)
        obs_mask = ~np.isnan(gapped)
        np.testing.assert_array_equal(imputed[obs_mask], gapped[obs_mask])


class TestLinearInterp:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        _check_imputer(LinearInterpImputer(), t, gapped, missing_idx)

    def test_observed_unchanged(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = LinearInterpImputer()
        imputed = imp.impute(t, gapped, missing_idx)
        obs_mask = ~np.isnan(gapped)
        np.testing.assert_allclose(imputed[obs_mask], gapped[obs_mask], atol=1e-10)

    def test_better_than_mean(self, gapped_fixture, signal_fixture):
        t, gapped, missing_idx, true_vals = gapped_fixture
        mean_imp = MeanFillImputer().impute(t, gapped, missing_idx)
        lin_imp = LinearInterpImputer().impute(t, gapped, missing_idx)
        mean_rmse = np.sqrt(np.mean((mean_imp[missing_idx] - true_vals) ** 2))
        lin_rmse = np.sqrt(np.mean((lin_imp[missing_idx] - true_vals) ** 2))
        assert lin_rmse <= mean_rmse + 0.05, "Linear interp should beat mean fill (or be close)"


class TestSplineInterp:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        _check_imputer(SplineInterpImputer(), t, gapped, missing_idx)


class TestTSMICE:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = TSMICEImputer(L=3, n_chains=2, max_iter=3, seed=0)
        _check_imputer(imp, t, gapped, missing_idx)

    def test_reproducibility(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp1 = TSMICEImputer(L=3, n_chains=2, max_iter=2, seed=42)
        imp2 = TSMICEImputer(L=3, n_chains=2, max_iter=2, seed=42)
        out1 = imp1.impute(t, gapped, missing_idx, period_est=1.0)
        out2 = imp2.impute(t, gapped, missing_idx, period_est=1.0)
        np.testing.assert_allclose(out1, out2, atol=1e-8)


class TestKNNImputer:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        _check_imputer(KNNImputer(k=3, W=5), t, gapped, missing_idx)

    def test_output_shape(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = KNNImputer(k=5, W=10)
        out = imp.impute(t, gapped, missing_idx)
        assert out.shape == gapped.shape


class TestRFImputer:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = RFImputer(n_estimators=5, L=3, seed=0)
        _check_imputer(imp, t, gapped, missing_idx)


class TestMFImputer:
    def test_no_nan(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        imp = MFImputer(rank=5, max_iter=20, seed=0)
        _check_imputer(imp, t, gapped, missing_idx)

    def test_low_rank_periodic_signal(self, signal_fixture):
        """Pure sinusoid = rank-2 Hankel; MF should recover it well at 10% missing."""
        t, flux, _ = signal_fixture
        gapped, missing_idx, true_vals = inject_gaps(flux, p=0.10, seed=99)
        imp = MFImputer(rank=10, max_iter=100, seed=0)
        imputed = imp.impute(t, gapped, missing_idx)
        rmse = float(np.sqrt(np.mean((imputed[missing_idx] - true_vals) ** 2)))
        assert rmse < 0.20, f"MF-Impute RMSE too high on periodic signal: {rmse:.4f}"


class TestRegistry:
    def test_list_imputers(self):
        names = list_imputers()
        assert len(names) == 13
        assert "gp_matern" in names
        assert "saits" in names

    def test_get_known_imputer(self):
        imp = get_imputer("mean_fill")
        assert imp.name == "Mean-Fill"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_imputer("does_not_exist")

    def test_all_deterministic_imputers_run(self, gapped_fixture):
        t, gapped, missing_idx, _ = gapped_fixture
        for key in ["mean_fill", "forward_fill", "linear_interp", "spline_interp"]:
            imp = get_imputer(key)
            out = imp.impute(t, gapped, missing_idx)
            assert not np.any(np.isnan(out)), f"{key} produced NaN"

    def test_seed_passed_to_imputer(self):
        imp = get_imputer("ts_mice", params={"L": 3, "n_chains": 2, "max_iter": 2}, seed=123)
        assert imp.seed == 123
