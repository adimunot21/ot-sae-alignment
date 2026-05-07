"""Tests for the 1-D closed-form Wasserstein distance."""

from __future__ import annotations

import numpy as np
import pytest

from ot_primitives.exact import wasserstein_1d


class TestWasserstein1D:
    """Closed-form 1-D W_p — sanity checks against hand-computed answers."""

    def test_two_diracs(self) -> None:
        """W_p between {0} and {3} is 3 for any p >= 1."""
        for p in [1.0, 2.0, 3.0]:
            d = wasserstein_1d(np.array([0.0]), np.array([3.0]), p=p)
            np.testing.assert_allclose(d, 3.0, rtol=1e-12)

    def test_translation_invariance(self) -> None:
        """W_p(x, x + c) = |c| for any constant c."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(50)
        c = 2.5
        d = wasserstein_1d(x, x + c, p=2.0)
        np.testing.assert_allclose(d, abs(c), rtol=1e-12)

    def test_symmetric(self) -> None:
        """W_p(x, y) == W_p(y, x)."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(30)
        y = rng.standard_normal(30)
        d_xy = wasserstein_1d(x, y, p=2.0)
        d_yx = wasserstein_1d(y, x, p=2.0)
        np.testing.assert_allclose(d_xy, d_yx, rtol=1e-12)

    def test_zero_when_equal(self) -> None:
        """W_p(x, x) = 0."""
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        d = wasserstein_1d(x, x, p=2.0)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_zero_when_permuted(self) -> None:
        """W_p(x, π(x)) = 0 for any permutation π — only the multiset matters."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(50)
        perm = rng.permutation(50)
        d = wasserstein_1d(x, x[perm], p=2.0)
        np.testing.assert_allclose(d, 0.0, atol=1e-12)

    def test_sort_by_hand_p2(self) -> None:
        """Tiny case computable by hand."""
        x = np.array([3.0, 1.0])
        y = np.array([4.0, 2.0])
        # Sorted: x = (1, 3), y = (2, 4). Diffs (1, 1). W_2^2 = (1 + 1) / 2 = 1.
        # W_2 = 1.
        d = wasserstein_1d(x, y, p=2.0)
        np.testing.assert_allclose(d, 1.0, rtol=1e-12)

    def test_sort_by_hand_p1(self) -> None:
        """Same case with p=1."""
        x = np.array([3.0, 1.0])
        y = np.array([4.0, 2.0])
        # Sorted diffs (1, 1). W_1 = (1 + 1) / 2 = 1.
        d = wasserstein_1d(x, y, p=1.0)
        np.testing.assert_allclose(d, 1.0, rtol=1e-12)

    def test_unequal_length_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            wasserstein_1d(np.zeros(3), np.zeros(4))

    def test_non_1d_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            wasserstein_1d(np.zeros((3, 2)), np.zeros(3))

    def test_invalid_p_raises(self) -> None:
        with pytest.raises(ValueError, match="p must be"):
            wasserstein_1d(np.zeros(3), np.zeros(3), p=0.5)
