"""Tests for the simulation module."""

import numpy as np
import pytest

from src.simulation.generator import generate_synthetic_lightcurve


class TestGenerateSyntheticLightcurve:
    def test_output_shapes(self):
        t, flux, params = generate_synthetic_lightcurve(N=100)
        assert t.shape == (100,)
        assert flux.shape == (100,)

    def test_time_vector(self):
        t, _, _ = generate_synthetic_lightcurve(N=50, dt=0.5)
        assert np.allclose(t[0], 0.0)
        assert np.allclose(np.diff(t), 0.5)

    def test_reproducibility(self):
        _, flux1, _ = generate_synthetic_lightcurve(N=200, seed=7)
        _, flux2, _ = generate_synthetic_lightcurve(N=200, seed=7)
        np.testing.assert_array_equal(flux1, flux2)

    def test_different_seeds(self):
        _, flux1, _ = generate_synthetic_lightcurve(N=200, seed=1)
        _, flux2, _ = generate_synthetic_lightcurve(N=200, seed=2)
        assert not np.allclose(flux1, flux2)

    def test_no_nan(self):
        _, flux, _ = generate_synthetic_lightcurve(N=500)
        assert not np.any(np.isnan(flux))

    def test_signal_range(self):
        """Flux should be approximately mu0 ± A (with noise headroom)."""
        t, flux, _ = generate_synthetic_lightcurve(N=5000, A=0.1, mu0=1.0, sigma_eps=0.01, seed=0)
        assert flux.mean() == pytest.approx(1.0, abs=0.05)
        assert flux.max() < 1.5
        assert flux.min() > 0.5

    def test_sinusoidal_model(self):
        t, flux, _ = generate_synthetic_lightcurve(N=200, model="sinusoidal", sigma_eps=0.0, seed=0)
        expected = 1.0 + 0.1 * np.sin(2 * np.pi * t / 1.0)
        np.testing.assert_allclose(flux, expected, atol=1e-10)

    def test_multi_harmonic_model(self):
        t, flux, _ = generate_synthetic_lightcurve(
            N=100, model="multi_harmonic", harmonics=[1, 2], sigma_eps=0.0, seed=0
        )
        assert not np.any(np.isnan(flux))

    def test_eclipsing_binary_model(self):
        _, flux, _ = generate_synthetic_lightcurve(N=100, model="eclipsing_binary", seed=0)
        assert not np.any(np.isnan(flux))

    def test_sawtooth_model(self):
        _, flux, _ = generate_synthetic_lightcurve(N=100, model="sawtooth", seed=0)
        assert not np.any(np.isnan(flux))

    def test_custom_model(self):
        fn = lambda t: np.ones_like(t) * 2.5
        _, flux, _ = generate_synthetic_lightcurve(N=50, model="custom", custom_func=fn, sigma_eps=0.0, seed=0)
        np.testing.assert_allclose(flux, 2.5, atol=1e-10)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown signal model"):
            generate_synthetic_lightcurve(N=10, model="bogus")

    def test_params_dict_keys(self):
        _, _, params = generate_synthetic_lightcurve(N=10)
        for key in ("N", "dt", "A", "P0", "phi0", "mu0", "sigma_eps", "seed", "model"):
            assert key in params
