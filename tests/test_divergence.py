"""Tests for Sinkhorn divergence.

Properties verified:
1. S_eps(alpha, alpha) = 0 (the headline property).
2. S_eps(alpha, beta) >= 0 (non-negativity).
3. S_eps(alpha, beta) = S_eps(beta, alpha) (symmetry).
4. S_eps -> W_p^p as eps -> 0 (low-eps limit).
5. Cost-form and dual-form agree at converged Sinkhorn.
6. Cross-validation against POT's empirical_sinkhorn_divergence on convex hulls
   of the regime where both implementations should agree.
"""

from __future__ import annotations

import numpy as np
import ot as pot
import pytest

from ot_primitives.costs import squared_euclidean_cost
from ot_primitives.divergence import (
    sinkhorn_divergence,
    sinkhorn_divergence_from_points,
)
from ot_primitives.exact import exact_ot

# Tolerances calibrated to Sinkhorn convergence floor.
_TOL_SELF_DUAL = 1e-9  # dual form should give ~exact zero on S_eps(alpha, alpha)
_TOL_SELF_COST = 1e-4  # cost form is approximate, depends on Sinkhorn convergence
_TOL_VS_LP = 5e-2  # divergence at small eps approximates W_2^2
_TOL_VS_POT = 1e-3  # cross-validation against POT


class TestSinkhornDivergenceSelfZero:
    """S_eps(alpha, alpha) = 0 — the headline property."""

    @pytest.mark.parametrize("eps", [0.01, 0.1, 1.0, 10.0])
    def test_dual_form_self_zero(self, rng: np.random.Generator, eps: float) -> None:
        """Dual form: S_eps(alpha, alpha) = 0 to machine precision."""
        n, d = 20, 3
        a = rng.dirichlet(np.ones(n))
        X = rng.standard_normal((n, d))
        C_xx = squared_euclidean_cost(X, X)

        s = sinkhorn_divergence(
            a,
            a,
            C_xx,
            C_xx,
            C_xx,
            eps=eps,
            form="dual",
            max_iter=5000,
            tol=1e-12,
        )
        assert abs(s) < _TOL_SELF_DUAL

    @pytest.mark.parametrize("eps", [0.1, 1.0])
    def test_cost_form_self_near_zero(self, rng: np.random.Generator, eps: float) -> None:
        """Cost form: S_eps(alpha, alpha) ≈ 0, with looser tolerance."""
        n, d = 20, 3
        a = rng.dirichlet(np.ones(n))
        X = rng.standard_normal((n, d))
        C_xx = squared_euclidean_cost(X, X)

        s = sinkhorn_divergence(
            a,
            a,
            C_xx,
            C_xx,
            C_xx,
            eps=eps,
            form="cost",
            max_iter=5000,
            tol=1e-12,
        )
        assert abs(s) < _TOL_SELF_COST


class TestSinkhornDivergenceProperties:
    """Symmetry, non-negativity, agreement of forms."""

    def test_symmetric(self, rng: np.random.Generator) -> None:
        """S_eps(alpha, beta) = S_eps(beta, alpha)."""
        n, m, d = 10, 12, 2
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d))

        s_xy = sinkhorn_divergence_from_points(X, Y, eps=0.1, max_iter=5000)
        s_yx = sinkhorn_divergence_from_points(Y, X, eps=0.1, max_iter=5000)

        np.testing.assert_allclose(s_xy, s_yx, rtol=1e-8)

    def test_non_negative(self, rng: np.random.Generator) -> None:
        """S_eps(alpha, beta) >= 0 across a range of inputs."""
        for trial in range(5):
            rng_t = np.random.default_rng(trial)
            n, m, d = 12, 15, 2
            X = rng_t.standard_normal((n, d))
            Y = rng_t.standard_normal((m, d)) + 1.0
            s = sinkhorn_divergence_from_points(X, Y, eps=0.1, max_iter=5000)
            assert s >= -1e-9, f"trial {trial}: S_eps = {s} (negative)"

    def test_cost_and_dual_differ_by_entropy_gap(self, rng: np.random.Generator) -> None:
        """The two forms compute different objects.

        ``S_dual = S_cost - eps * (H(P_xy) - 0.5 H(P_xx) - 0.5 H(P_yy))``.
        For typical distinct measures the entropy gap is positive, so
        ``S_dual < S_cost``. We just check the inequality holds — the exact
        gap value depends on the measures and is not a clean assertion target.
        """
        n, m, d = 10, 12, 2
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d)) + 0.5

        s_dual = sinkhorn_divergence_from_points(
            X, Y, eps=0.1, form="dual", max_iter=10000, tol=1e-12
        )
        s_cost = sinkhorn_divergence_from_points(
            X, Y, eps=0.1, form="cost", max_iter=10000, tol=1e-12
        )
        # Both should be positive (different alpha and beta).
        assert s_dual > 0
        assert s_cost > 0
        # Cost form upper-bounds dual form by the entropy gap.
        assert s_dual < s_cost


class TestSinkhornDivergenceLimits:
    """Limit behavior of S_eps."""

    def test_approaches_w2_squared_as_eps_small(self, rng: np.random.Generator) -> None:
        """As eps -> 0, S_eps(alpha, beta) -> W_2^2(alpha, beta)."""
        n, m, d = 15, 15, 2
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d)) + 1.0

        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        C_xy = squared_euclidean_cost(X, Y)
        cost_lp, _ = exact_ot(a, b, C_xy)
        # Note: exact_ot returns <P, C>; since C is squared Euclidean here,
        # this *is* W_2^2.
        w2_squared_true = cost_lp

        s = sinkhorn_divergence_from_points(X, Y, eps=0.005, max_iter=20000, tol=1e-12)
        rel_err = abs(s - w2_squared_true) / abs(w2_squared_true)
        assert rel_err < _TOL_VS_LP


class TestSinkhornDivergenceAgainstPOT:
    """Cross-validate against POT's reference implementation."""

    @pytest.mark.parametrize("eps", [0.05, 0.1, 0.5])
    def test_matches_pot(self, rng: np.random.Generator, eps: float) -> None:
        n, m, d = 20, 25, 3
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((m, d)) + 0.3

        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)

        s_ours = sinkhorn_divergence_from_points(
            X,
            Y,
            a=a,
            b=b,
            eps=eps,
            form="cost",
            max_iter=10000,
            tol=1e-12,
        )

        # POT's API for the divergence on point clouds.
        s_pot = float(
            pot.bregman.empirical_sinkhorn_divergence(
                X, Y, reg=eps, a=a, b=b, numIterMax=10000, stopThr=1e-12
            )
        )

        np.testing.assert_allclose(s_ours, s_pot, rtol=_TOL_VS_POT, atol=1e-6)


class TestSinkhornDivergenceValidation:
    def test_invalid_form_raises(self) -> None:
        a = np.array([1.0])
        b = np.array([1.0])
        C = np.zeros((1, 1))
        with pytest.raises(ValueError, match="form must be"):
            sinkhorn_divergence(a, b, C, C, C, eps=0.1, form="invalid")  # type: ignore[arg-type]

    def test_shape_mismatch_xx(self) -> None:
        a = np.array([0.5, 0.5])
        b = np.array([1.0])
        C_xy = np.zeros((2, 1))
        C_xx = np.zeros((3, 3))  # should be (2, 2)
        C_yy = np.zeros((1, 1))
        with pytest.raises(ValueError, match="C_xx"):
            sinkhorn_divergence(a, b, C_xy, C_xx, C_yy, eps=0.1)

    def test_shape_mismatch_yy(self) -> None:
        a = np.array([0.5, 0.5])
        b = np.array([1.0])
        C_xy = np.zeros((2, 1))
        C_xx = np.zeros((2, 2))
        C_yy = np.zeros((2, 2))  # should be (1, 1)
        with pytest.raises(ValueError, match="C_yy"):
            sinkhorn_divergence(a, b, C_xy, C_xx, C_yy, eps=0.1)
