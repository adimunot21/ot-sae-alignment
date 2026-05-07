"""Tests for log-domain Sinkhorn.

Properties verified:
1. Marginal constraints satisfied to tolerance.
2. As eps -> 0, Sinkhorn cost approaches the LP cost.
3. Plan structure: P_ij = exp((f_i + g_j - C_ij)/eps) holds at convergence.
4. Symmetry: sinkhorn(a, b, C) cost equals sinkhorn(b, a, C^T) cost.
5. Cross-validation against POT's sinkhorn2 over a sweep of eps.
6. Numerical stability: log-domain works at eps where multiplicative form fails.
"""

from __future__ import annotations

import warnings

import numpy as np
import ot as pot
import pytest

from ot_primitives._legacy import sinkhorn_multiplicative
from ot_primitives.costs import squared_euclidean_cost
from ot_primitives.exact import exact_ot
from ot_primitives.sinkhorn import sinkhorn

_TOL_MARGINAL = 1e-7
_TOL_VS_LP = 1e-2  # Sinkhorn at small eps is close to but not equal to LP.
_TOL_VS_POT = 1e-4


class TestSinkhornBasic:
    """Marginal satisfaction, plan structure, basic properties."""

    def test_marginal_constraints(self, rng: np.random.Generator) -> None:
        n, m = 15, 20
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 3))
        Y = rng.standard_normal((m, 3))
        C = squared_euclidean_cost(X, Y)

        result = sinkhorn(a, b, C, eps=0.1)

        np.testing.assert_allclose(result.plan.sum(axis=1), a, atol=_TOL_MARGINAL)
        np.testing.assert_allclose(result.plan.sum(axis=0), b, atol=_TOL_MARGINAL)
        assert result.converged

    def test_plan_from_potentials(self, rng: np.random.Generator) -> None:
        """Reconstructing P from (f, g) recovers the same plan."""
        n, m = 12, 10
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)
        eps = 0.1

        result = sinkhorn(a, b, C, eps=eps)
        plan_from_potentials = np.exp((result.f[:, None] + result.g[None, :] - C) / eps)
        np.testing.assert_allclose(result.plan, plan_from_potentials, rtol=1e-12)

    def test_symmetry(self, rng: np.random.Generator) -> None:
        """sinkhorn(a, b, C) and sinkhorn(b, a, C^T) give same cost."""
        n, m = 8, 11
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)

        c_forward = sinkhorn(a, b, C, eps=0.1).cost
        c_backward = sinkhorn(b, a, C.T, eps=0.1).cost
        np.testing.assert_allclose(c_forward, c_backward, rtol=1e-8)

    def test_plan_non_negative(self, rng: np.random.Generator) -> None:
        n, m = 10, 10
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)

        result = sinkhorn(a, b, C, eps=0.05)
        assert result.plan.min() >= 0.0


class TestSinkhornApproachesLP:
    """As eps -> 0, Sinkhorn cost approaches the exact LP cost."""

    def test_small_eps_near_lp(self, rng: np.random.Generator) -> None:
        """At eps=0.005, Sinkhorn cost should be within 1% of LP cost."""
        n, m = 20, 20
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)

        cost_lp, _ = exact_ot(a, b, C)
        cost_sk = sinkhorn(a, b, C, eps=0.005, max_iter=5000).cost

        rel_err = abs(cost_sk - cost_lp) / abs(cost_lp)
        assert rel_err < _TOL_VS_LP

    def test_monotone_in_eps(self, rng: np.random.Generator) -> None:
        """Sinkhorn cost increases monotonically with eps for typical inputs."""
        n, m = 15, 15
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)

        epsilons = [0.01, 0.05, 0.2, 1.0, 5.0]
        costs = [sinkhorn(a, b, C, eps=e, max_iter=5000).cost for e in epsilons]
        # Costs should be non-decreasing (entropy regularization makes the
        # plan more spread out, raising the transport cost).
        for c1, c2 in zip(costs, costs[1:], strict=False):
            assert c1 <= c2 + 1e-9


class TestSinkhornAgainstPOT:
    """Cross-validate against POT's reference implementation."""

    @pytest.mark.parametrize("eps", [0.01, 0.05, 0.1, 0.5, 1.0])
    def test_cost_matches_pot(self, rng: np.random.Generator, eps: float) -> None:
        n, m = 15, 18
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)

        cost_ours = sinkhorn(a, b, C, eps=eps, max_iter=5000, tol=1e-12).cost
        cost_pot = float(pot.sinkhorn2(a, b, C, reg=eps, numItermax=5000, stopThr=1e-12))

        np.testing.assert_allclose(cost_ours, cost_pot, rtol=_TOL_VS_POT)


class TestSinkhornStability:
    """Log-domain works where multiplicative breaks."""

    def test_log_domain_stable_at_small_eps(self, rng: np.random.Generator) -> None:
        """eps=0.001 with unit-scale costs: log-domain returns a valid result."""
        n, m = 10, 10
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)

        result = sinkhorn(a, b, C, eps=0.001, max_iter=10000)
        assert np.isfinite(result.cost)
        assert np.all(np.isfinite(result.plan))
        assert result.plan.min() >= 0.0

    def test_multiplicative_breaks_at_small_eps(self, rng: np.random.Generator) -> None:
        """Sanity check: the multiplicative form does fail where log-domain succeeds.

        We construct a case with large costs / small eps where exp(-C/eps)
        underflows. If the multiplicative form *doesn't* break here, our
        motivation for log-domain is suspect.
        """
        n, m = 10, 10
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2)) * 100  # large costs
        Y = rng.standard_normal((m, 2)) * 100
        C = squared_euclidean_cost(X, Y)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            cost_mult, plan_mult = sinkhorn_multiplicative(a, b, C, eps=0.01, n_iter=1000)

        # Multiplicative form should produce NaN or a non-finite plan here.
        # If this assertion fails, the regime isn't extreme enough.
        assert (
            not np.all(np.isfinite(plan_mult))
            or np.any(plan_mult < -1e-6)
            or not np.isfinite(cost_mult)
        )

        # Log-domain should still work.
        result = sinkhorn(a, b, C, eps=0.01, max_iter=10000)
        assert np.isfinite(result.cost)


class TestSinkhornValidation:
    def test_negative_eps_raises(self) -> None:
        a = np.array([1.0])
        b = np.array([1.0])
        C = np.zeros((1, 1))
        with pytest.raises(ValueError, match="eps must be positive"):
            sinkhorn(a, b, C, eps=-0.1)

    def test_zero_eps_raises(self) -> None:
        a = np.array([1.0])
        b = np.array([1.0])
        C = np.zeros((1, 1))
        with pytest.raises(ValueError, match="eps must be positive"):
            sinkhorn(a, b, C, eps=0.0)

    def test_invalid_max_iter_raises(self) -> None:
        a = np.array([1.0])
        b = np.array([1.0])
        C = np.zeros((1, 1))
        with pytest.raises(ValueError, match="max_iter must be"):
            sinkhorn(a, b, C, eps=0.1, max_iter=0)
