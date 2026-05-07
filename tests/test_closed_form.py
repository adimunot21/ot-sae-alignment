"""Tests for closed-form Wasserstein distances.

The Bures-Wasserstein formula is itself an oracle for our LP and Sinkhorn
solvers — but it has its own analytical sanity checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from ot_primitives.closed_form import bures_wasserstein


class TestBuresWassersteinHandComputed:
    """Cases where the answer is determined by the formula's structure."""

    def test_identical_gaussians_zero(self) -> None:
        """W_2(alpha, alpha) = 0."""
        rng = np.random.default_rng(0)
        d = 4
        mu = rng.standard_normal(d)
        L = rng.standard_normal((d, d))
        cov = L @ L.T + 0.1 * np.eye(d)
        d_w2 = bures_wasserstein(mu, cov, mu, cov)
        np.testing.assert_allclose(d_w2, 0.0, atol=1e-8)

    def test_same_cov_translation_only(self) -> None:
        """When covariances match, distance reduces to ||mu1 - mu2||."""
        rng = np.random.default_rng(1)
        d = 3
        mu1 = rng.standard_normal(d)
        mu2 = rng.standard_normal(d)
        L = rng.standard_normal((d, d))
        cov = L @ L.T + 0.5 * np.eye(d)
        d_w2 = bures_wasserstein(mu1, cov, mu2, cov)
        expected = float(np.linalg.norm(mu1 - mu2))
        np.testing.assert_allclose(d_w2, expected, rtol=1e-10)

    def test_zero_mean_isotropic_1d(self) -> None:
        """1-D zero-mean Gaussians: W_2 between N(0, s1^2) and N(0, s2^2) = |s1 - s2|."""
        s1, s2 = 1.0, 4.0
        d_w2 = bures_wasserstein(
            np.array([0.0]),
            np.array([[s1**2]]),
            np.array([0.0]),
            np.array([[s2**2]]),
        )
        np.testing.assert_allclose(d_w2, abs(s1 - s2), rtol=1e-10)

    def test_2d_isotropic(self) -> None:
        """Isotropic Gaussians N(0, s_i^2 I) in 2D: W_2 = sqrt(2) |s1 - s2|.

        Each axis contributes (s1 - s2)^2; total is 2 * (s1 - s2)^2; W_2 is the sqrt.
        """
        s1, s2 = 1.0, 3.0
        d = 2
        d_w2 = bures_wasserstein(
            np.zeros(d),
            s1**2 * np.eye(d),
            np.zeros(d),
            s2**2 * np.eye(d),
        )
        np.testing.assert_allclose(d_w2, np.sqrt(2.0) * abs(s1 - s2), rtol=1e-10)


class TestBuresWassersteinSymmetry:
    """W_2 is a metric — symmetric and non-negative."""

    def test_symmetric(self) -> None:
        rng = np.random.default_rng(2)
        d = 5
        mu1 = rng.standard_normal(d)
        mu2 = rng.standard_normal(d)
        L1 = rng.standard_normal((d, d))
        L2 = rng.standard_normal((d, d))
        cov1 = L1 @ L1.T + 0.1 * np.eye(d)
        cov2 = L2 @ L2.T + 0.1 * np.eye(d)
        d_12 = bures_wasserstein(mu1, cov1, mu2, cov2)
        d_21 = bures_wasserstein(mu2, cov2, mu1, cov1)
        np.testing.assert_allclose(d_12, d_21, rtol=1e-10)

    def test_non_negative(self) -> None:
        for seed in range(5):
            rng = np.random.default_rng(seed)
            d = 3
            mu1 = rng.standard_normal(d)
            mu2 = rng.standard_normal(d)
            L1 = rng.standard_normal((d, d))
            L2 = rng.standard_normal((d, d))
            cov1 = L1 @ L1.T + 0.5 * np.eye(d)
            cov2 = L2 @ L2.T + 0.5 * np.eye(d)
            d_w2 = bures_wasserstein(mu1, cov1, mu2, cov2)
            assert d_w2 >= 0.0


class TestBuresWassersteinAgainstSamples:
    """The closed form must agree with sample-based estimators in the limit."""

    def test_lp_estimate_converges_to_bures(self) -> None:
        """For 2D Gaussians, exact_ot on n samples approaches the Bures answer."""
        from ot_primitives.costs import squared_euclidean_cost
        from ot_primitives.exact import exact_ot

        rng = np.random.default_rng(0)
        d = 2
        mu1 = np.zeros(d)
        mu2 = np.array([1.5, 0.0])
        cov1 = np.eye(d)
        cov2 = 0.5 * np.eye(d)

        bures_dist = bures_wasserstein(mu1, cov1, mu2, cov2)
        bures_w2_squared = bures_dist**2

        # Sample-based estimate at n=300. Single seed; we just want order of
        # magnitude agreement (sample-based W_2 has O(n^{-1/d}) bias).
        n = 300
        L1 = np.linalg.cholesky(cov1)
        L2 = np.linalg.cholesky(cov2)
        X = mu1 + rng.standard_normal((n, d)) @ L1.T
        Y = mu2 + rng.standard_normal((n, d)) @ L2.T

        a = np.full(n, 1.0 / n)
        b = np.full(n, 1.0 / n)
        C = squared_euclidean_cost(X, Y)
        cost_lp, _ = exact_ot(a, b, C)

        # Loose tolerance: empirical W_2^2 has bias ~n^{-1/d}; in 2D at n=300
        # we expect ~5-15% relative error.
        rel_err = abs(cost_lp - bures_w2_squared) / abs(bures_w2_squared)
        assert rel_err < 0.20, f"empirical W_2^2 = {cost_lp}, Bures W_2^2 = {bures_w2_squared}"


class TestBuresWassersteinValidation:
    def test_dim_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            bures_wasserstein(
                np.zeros(2),
                np.eye(2),
                np.zeros(3),
                np.eye(3),
            )

    def test_cov_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match=r"cov1, cov2 must be"):
            bures_wasserstein(
                np.zeros(3),
                np.eye(2),  # mu is 3-d, cov is 2x2
                np.zeros(3),
                np.eye(3),
            )

    def test_asymmetric_cov_rejected(self) -> None:
        cov_asym = np.array([[1.0, 0.5], [0.0, 1.0]])  # not symmetric
        with pytest.raises(ValueError, match="cov1 is not symmetric"):
            bures_wasserstein(
                np.zeros(2),
                cov_asym,
                np.zeros(2),
                np.eye(2),
            )
