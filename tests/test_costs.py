"""Tests for ground cost functions."""

from __future__ import annotations

import numpy as np
import pytest

from ot_primitives.costs import squared_euclidean_cost


class TestSquaredEuclideanCost:
    """``squared_euclidean_cost`` correctness and shape behavior."""

    def test_zero_when_identical_points(self) -> None:
        """C[i, i] should be exactly zero when X == Y."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 3))
        C = squared_euclidean_cost(X, X)
        np.testing.assert_allclose(np.diag(C), 0.0, atol=1e-10)

    def test_matches_explicit_loop(self) -> None:
        """Vectorized form agrees with naive (i, j) loop on a small case."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((6, 2))
        Y = rng.standard_normal((4, 2))
        C_fast = squared_euclidean_cost(X, Y)

        C_slow = np.zeros((6, 4))
        for i in range(6):
            for j in range(4):
                d = X[i] - Y[j]
                C_slow[i, j] = float(d @ d)

        np.testing.assert_allclose(C_fast, C_slow, rtol=1e-12, atol=1e-12)

    def test_non_negative(self) -> None:
        """Cost matrix entries are non-negative even for near-coincident points."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((20, 5))
        Y = X + rng.standard_normal((20, 5)) * 1e-9  # near-coincident
        C = squared_euclidean_cost(X, Y)
        assert C.min() >= 0.0

    def test_shape(self) -> None:
        rng = np.random.default_rng(3)
        X = rng.standard_normal((7, 4))
        Y = rng.standard_normal((11, 4))
        C = squared_euclidean_cost(X, Y)
        assert C.shape == (7, 11)

    def test_known_distance(self) -> None:
        """C[i, j] = sum of squared coordinate differences, by hand."""
        X = np.array([[0.0, 0.0]])
        Y = np.array([[3.0, 4.0]])
        C = squared_euclidean_cost(X, Y)
        # ||(0,0) - (3,4)||^2 = 9 + 16 = 25
        np.testing.assert_allclose(C, [[25.0]], rtol=1e-12)

    def test_dim_mismatch_raises(self) -> None:
        X = np.zeros((3, 2))
        Y = np.zeros((3, 4))
        with pytest.raises(ValueError, match="same dimension"):
            squared_euclidean_cost(X, Y)

    def test_non_2d_raises(self) -> None:
        X = np.zeros(3)
        Y = np.zeros((3, 2))
        with pytest.raises(ValueError, match="2-D"):
            squared_euclidean_cost(X, Y)
