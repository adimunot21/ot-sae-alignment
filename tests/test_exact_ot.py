"""Tests for the LP-based exact optimal transport solver.

Three oracles cross-check ``exact_ot``:
1. Hand-computed answers on tiny problems.
2. The 1-D closed form (``wasserstein_1d``) on 1-D inputs.
3. POT's ``ot.emd2`` (network-simplex solver) on random problems.

If any two disagree, one of them has a bug. POT and our LP solver implement
different algorithms (network simplex vs interior-point/HiGHS), so agreement
between them is strong evidence of correctness for both.
"""

from __future__ import annotations

import numpy as np
import ot as pot  # POT — the de facto OT library
import pytest

from ot_primitives._utils import marginal_violation
from ot_primitives.costs import squared_euclidean_cost
from ot_primitives.exact import exact_ot, wasserstein_1d

# Tolerances:
# - For hand-computed answers we use 1e-10 because there's no reason for slop.
# - For the LP-vs-1D-closed-form comparison we allow 1e-8 because the LP solver
#   uses an interior-point method with its own tolerances.
# - For LP-vs-POT we use 1e-6 relative tolerance: two different algorithms,
#   different floating-point paths.
_TOL_HAND = 1e-10
_TOL_VS_CLOSED_FORM = 1e-8
_TOL_VS_POT = 1e-6


class TestExactOTHandComputed:
    """Tiny problems where the answer can be verified by inspection."""

    def test_two_diracs(self) -> None:
        """OT between δ_0 and δ_3 with cost C = |x − y|^2 = 9."""
        a = np.array([1.0])
        b = np.array([1.0])
        C = np.array([[9.0]])
        cost, plan = exact_ot(a, b, C)
        np.testing.assert_allclose(cost, 9.0, atol=_TOL_HAND)
        np.testing.assert_allclose(plan, [[1.0]], atol=_TOL_HAND)

    def test_uniform_2_to_2_monotone(self) -> None:
        """Uniform 2-point measures with monotone optimal plan.

        Source: 0.5 at x=0, 0.5 at x=1. Target: 0.5 at y=2, 0.5 at y=3.
        Squared cost gives C = [[4, 9], [1, 4]]. Optimal plan ships
        0 → 2 (cost 4) and 1 → 3 (cost 4), total cost = 0.5*4 + 0.5*4 = 4.
        """
        a = np.array([0.5, 0.5])
        b = np.array([0.5, 0.5])
        C = np.array([[4.0, 9.0], [1.0, 4.0]])
        cost, plan = exact_ot(a, b, C)
        np.testing.assert_allclose(cost, 4.0, atol=_TOL_HAND)
        expected_plan = np.array([[0.5, 0.0], [0.0, 0.5]])
        np.testing.assert_allclose(plan, expected_plan, atol=_TOL_HAND)


class TestExactOTMarginals:
    """The recovered plan must satisfy marginal constraints."""

    def test_marginals_equal(self, rng: np.random.Generator) -> None:
        n, m = 20, 20
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)
        _, plan = exact_ot(a, b, C)
        assert marginal_violation(plan, a, b) < _TOL_VS_CLOSED_FORM

    def test_marginals_unequal_size(self, rng: np.random.Generator) -> None:
        """Rectangular case: n != m, non-uniform marginals."""
        n, m = 8, 12
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 3))
        Y = rng.standard_normal((m, 3))
        C = squared_euclidean_cost(X, Y)
        _, plan = exact_ot(a, b, C)
        assert plan.shape == (n, m)
        assert marginal_violation(plan, a, b) < _TOL_VS_CLOSED_FORM

    def test_plan_non_negative(self, rng: np.random.Generator) -> None:
        n, m = 10, 15
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)
        _, plan = exact_ot(a, b, C)
        assert plan.min() >= -1e-12  # tiny negative round-off acceptable


class TestExactOTAgainst1DClosedForm:
    """On 1-D inputs, the LP must match ``wasserstein_1d`` to high precision."""

    @pytest.mark.parametrize("n", [5, 20, 50])
    @pytest.mark.parametrize("p", [1.0, 2.0])
    def test_lp_matches_sort_formula(self, rng: np.random.Generator, n: int, p: float) -> None:
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)

        # 1-D closed form.
        wp_closed = wasserstein_1d(x, y, p=p)

        # LP form: cost matrix is |x_i - y_j|^p, marginals uniform 1/n.
        a = np.full(n, 1.0 / n)
        b = np.full(n, 1.0 / n)
        C = np.abs(x[:, None] - y[None, :]) ** p
        cost_lp, _ = exact_ot(a, b, C)
        wp_lp = cost_lp ** (1.0 / p)

        np.testing.assert_allclose(wp_lp, wp_closed, rtol=_TOL_VS_CLOSED_FORM)


class TestExactOTAgainstPOT:
    """Cross-validate against POT's network-simplex solver."""

    @pytest.mark.parametrize("seed", list(range(5)))
    @pytest.mark.parametrize("n,m", [(10, 10), (15, 8), (5, 20)])
    def test_lp_matches_pot(self, seed: int, n: int, m: int) -> None:
        rng = np.random.default_rng(seed)
        a = rng.dirichlet(np.ones(n))
        b = rng.dirichlet(np.ones(m))
        X = rng.standard_normal((n, 2))
        Y = rng.standard_normal((m, 2))
        C = squared_euclidean_cost(X, Y)

        cost_ours, _ = exact_ot(a, b, C)
        cost_pot = float(pot.emd2(a, b, C))

        np.testing.assert_allclose(cost_ours, cost_pot, rtol=_TOL_VS_POT)


class TestExactOTValidation:
    """Input validation."""

    def test_marginal_does_not_sum_to_one(self) -> None:
        a = np.array([0.5, 0.4])  # sums to 0.9
        b = np.array([0.5, 0.5])
        C = np.zeros((2, 2))
        with pytest.raises(ValueError, match="must sum to 1"):
            exact_ot(a, b, C)

    def test_negative_marginal(self) -> None:
        a = np.array([0.6, -0.1, 0.5])
        b = np.array([1.0])
        C = np.zeros((3, 1))
        with pytest.raises(ValueError, match="non-negative"):
            exact_ot(a, b, C)

    def test_cost_shape_mismatch(self) -> None:
        a = np.array([0.5, 0.5])
        b = np.array([1.0])
        C = np.zeros((3, 1))  # should be (2, 1)
        with pytest.raises(ValueError, match="must have shape"):
            exact_ot(a, b, C)

    def test_non_finite_cost(self) -> None:
        a = np.array([1.0])
        b = np.array([1.0])
        C = np.array([[np.inf]])
        with pytest.raises(ValueError, match="non-finite"):
            exact_ot(a, b, C)
