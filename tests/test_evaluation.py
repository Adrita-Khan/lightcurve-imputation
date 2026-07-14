"""Tests for evaluation metrics and aggregation."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_rmse,
    compute_mae,
    compute_mse,
    compute_relative_error,
    compute_period_recovery,
    compute_amplitude_error,
    compute_phase_error,
    evaluate_imputation,
)
from src.evaluation.aggregator import aggregate_results, summarise_results
from src.simulation.generator import generate_synthetic_lightcurve


@pytest.fixture(scope="module")
def clean_signal():
    t, flux, params = generate_synthetic_lightcurve(N=500, seed=0, sigma_eps=0.0)
    return t, flux, params


class TestPointwiseMetrics:
    def test_rmse_zero_for_perfect(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(x, x) == pytest.approx(0.0, abs=1e-12)

    def test_rmse_positive(self):
        true = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.2])
        assert compute_rmse(true, pred) > 0

    def test_mae_zero_for_perfect(self):
        x = np.ones(10)
        assert compute_mae(x, x) == pytest.approx(0.0, abs=1e-12)

    def test_mse_relationship_to_rmse(self):
        true = np.random.default_rng(0).standard_normal(100)
        pred = true + 0.1
        assert compute_mse(true, pred) == pytest.approx(compute_rmse(true, pred) ** 2, rel=1e-6)

    def test_relative_error_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        assert compute_relative_error(x, x) == pytest.approx(0.0, abs=1e-10)

    def test_relative_error_positive(self):
        true = np.array([1.0, 2.0])
        pred = np.array([2.0, 4.0])
        assert compute_relative_error(true, pred) > 0


class TestSignalPropertyMetrics:
    def test_period_recovery_perfect(self, clean_signal):
        t, flux, params = clean_signal
        recovered, rel_err = compute_period_recovery(t, flux, true_period=params["P0"])
        assert recovered is True
        assert rel_err < 0.01

    def test_period_recovery_wrong(self, clean_signal):
        t, flux, params = clean_signal
        # Inject constant signal (no period)
        constant = np.ones_like(flux) * 1.0
        # With a constant signal, LS period is undefined; just check it returns something
        recovered, rel_err = compute_period_recovery(t, constant, true_period=params["P0"])
        assert isinstance(recovered, bool)
        assert isinstance(rel_err, float)

    def test_amplitude_error_perfect(self, clean_signal):
        t, flux, params = clean_signal
        err = compute_amplitude_error(flux, params["A"])
        assert err < 0.05  # small error for noise-free signal

    def test_amplitude_error_zero_for_constant(self):
        # A constant signal has zero amplitude, true amplitude = 0.1 → large error
        constant = np.ones(100)
        err = compute_amplitude_error(constant, true_amplitude=0.1)
        assert err > 0.5

    def test_phase_error_returns_float(self, clean_signal):
        t, flux, params = clean_signal
        err = compute_phase_error(t, flux, params["P0"], params["phi0"])
        assert isinstance(err, float)
        assert 0.0 <= err <= 0.5  # normalised to [0, 0.5] period fraction


class TestEvaluateImputation:
    def test_full_metric_dict(self, clean_signal):
        t, flux, params = clean_signal
        missing_idx = np.arange(10, 50)
        true_vals = flux[missing_idx]
        result = evaluate_imputation(
            t=t,
            flux_imputed=flux,
            missing_idx=missing_idx,
            true_vals=true_vals,
            true_period=params["P0"],
            true_amplitude=params["A"],
            true_phase=params["phi0"],
        )
        expected_keys = {
            "rmse", "mae", "mse", "relative_error",
            "period_recovered", "period_rel_err",
            "amplitude_error", "phase_error",
            "runtime_s", "memory_mb",
        }
        assert expected_keys <= set(result.keys())
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)


class TestAggregation:
    @pytest.fixture
    def sample_results(self):
        rows = []
        for method in ["Mean-Fill", "Linear-Interp"]:
            for frac in [0.10, 0.30]:
                for seed in range(5):
                    rows.append({
                        "method": method,
                        "fraction": frac,
                        "seed": seed,
                        "rmse": np.random.default_rng(seed).uniform(0.01, 0.20),
                        "mae": np.random.default_rng(seed + 100).uniform(0.01, 0.15),
                        "period_recovered": int(np.random.default_rng(seed).uniform() > 0.3),
                        "runtime_s": 0.1,
                        "memory_mb": 5.0,
                    })
        return rows

    def test_aggregate_returns_dataframe(self, sample_results):
        df = aggregate_results(sample_results)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 20  # 2 methods × 2 fractions × 5 seeds

    def test_summarise_has_mean_std(self, sample_results):
        df = aggregate_results(sample_results)
        summary = summarise_results(df)
        assert "rmse_mean" in summary.columns
        assert "rmse_std" in summary.columns
        assert "prr" in summary.columns

    def test_summarise_groups_correctly(self, sample_results):
        df = aggregate_results(sample_results)
        summary = summarise_results(df)
        # Should have 2 methods × 2 fractions = 4 rows
        assert len(summary) == 4
