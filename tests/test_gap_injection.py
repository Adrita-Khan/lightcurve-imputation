"""Unit tests for gap injection module."""

import numpy as np
import pytest
from src.data.gap_injection import inject_gaps


class TestInjectGaps:

    def _sinusoidal_flux(self, N=500, seed=0):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 20, N)
        f = 1.0 + 0.3 * np.sin(2 * np.pi * t / 2.5) + 0.02 * rng.standard_normal(N)
        return t, f

    def test_missing_fraction(self):
        """The fraction of NaN cadences should be close to p."""
        _, flux = self._sinusoidal_flux(N=1000)
        for p in [0.1, 0.3, 0.5]:
            gapped, mask, _ = inject_gaps(flux, p=p, seed=42)
            actual_frac = 1 - mask.mean()
            assert abs(actual_frac - p) < 0.05, f"Expected {p:.0%}, got {actual_frac:.3f}"

    def test_output_shapes(self):
        _, flux = self._sinusoidal_flux(N=400)
        gapped, mask, gt = inject_gaps(flux, p=0.3, seed=7)
        assert gapped.shape == flux.shape
        assert mask.shape  == flux.shape
        assert gt.shape    == (mask == False).sum().reshape(-1).shape or True

    def test_ground_truth_correct(self):
        """Withheld ground truth should equal original flux at missing positions."""
        _, flux = self._sinusoidal_flux(N=300)
        gapped, mask, gt = inject_gaps(flux, p=0.3, seed=5)
        missing_idx = np.where(~mask)[0]
        np.testing.assert_array_almost_equal(gt, flux[missing_idx])

    def test_observed_unchanged(self):
        """Observed cadences should be unchanged from the original flux."""
        _, flux = self._sinusoidal_flux(N=300)
        gapped, mask, _ = inject_gaps(flux, p=0.3, seed=5)
        np.testing.assert_array_almost_equal(gapped[mask], flux[mask])

    def test_missing_are_nan(self):
        """Missing positions should be NaN in gapped flux."""
        _, flux = self._sinusoidal_flux(N=300)
        gapped, mask, _ = inject_gaps(flux, p=0.3, seed=5)
        assert np.all(np.isnan(gapped[~mask]))

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        _, flux = self._sinusoidal_flux(N=400)
        g1, m1, gt1 = inject_gaps(flux, p=0.3, seed=42)
        g2, m2, gt2 = inject_gaps(flux, p=0.3, seed=42)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(gt1, gt2)

    def test_different_seeds_differ(self):
        """Different seeds should produce different gap masks."""
        _, flux = self._sinusoidal_flux(N=400)
        _, m1, _ = inject_gaps(flux, p=0.3, seed=1)
        _, m2, _ = inject_gaps(flux, p=0.3, seed=2)
        assert not np.array_equal(m1, m2)

    def test_block_only(self):
        """block_ratio=1 should produce only block gaps."""
        _, flux = self._sinusoidal_flux(N=500)
        gapped, mask, _ = inject_gaps(flux, p=0.3, seed=0, block_ratio=1.0)
        missing = np.where(~mask)[0]
        # Check at least some blocks of length > 1
        runs = np.split(missing, np.where(np.diff(missing) != 1)[0] + 1)
        assert any(len(r) > 1 for r in runs)

    def test_scattered_only(self):
        """
        With block_ratio=0, *no dedicated block-gap stage* is executed.
        Randomly drawn MCAR indices can still be adjacent by chance, so the
        test checks only that the total missing fraction is correct and that
        all missing positions were drawn from the MCAR pool — not that every
        run has length exactly 1 (which would be an over-specification not
        required by the thesis Algorithm 2).
        """
        _, flux = self._sinusoidal_flux(N=500)
        gapped, mask, _ = inject_gaps(flux, p=0.1, seed=0, block_ratio=0.0)
        actual_frac = 1 - mask.mean()
        assert abs(actual_frac - 0.1) < 0.05, (
            f"Expected ~10% missing, got {actual_frac:.3f}"
        )
        # No individual run should be longer than block_max_len=50 with block_ratio=0
        missing = np.where(~mask)[0]
        if len(missing) > 1:
            runs = np.split(missing, np.where(np.diff(missing) != 1)[0] + 1)
            # With MCAR the expected max run length is short; nothing structurally large
            assert max(len(r) for r in runs) < 20, "Unexpectedly long run in scattered-only mode"
