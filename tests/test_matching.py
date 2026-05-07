"""Sanity tests for matching functions on synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ot_sae.matching import (
    activation_hungarian,
    cosine_hungarian,
    fused_gw_matching,
    gw_matching,
)


class TestCosineHungarianRecoversPermutation:
    def test_identity_match(self) -> None:
        """Two identical decoder matrices: matching should be identity."""
        rng = np.random.default_rng(0)
        W = torch.tensor(rng.standard_normal((20, 8)), dtype=torch.float32)
        matching = cosine_hungarian(W, W)
        assert np.array_equal(matching, np.arange(20))

    def test_permutation_recovered(self) -> None:
        """Permuted decoder: matching should be the inverse permutation."""
        rng = np.random.default_rng(1)
        n = 25
        W = torch.tensor(rng.standard_normal((n, 6)), dtype=torch.float32)
        perm = rng.permutation(n)
        W_perm = W[perm]
        matching = cosine_hungarian(W, W_perm)
        # If A[i] should be matched to B[matching[i]], and B[j] = A[perm[j]],
        # then matching[i] should be the j with perm[j] == i, i.e. argsort(perm).
        expected = np.argsort(perm)
        assert np.array_equal(matching, expected)


class TestActivationHungarianRecoversPermutation:
    def test_permutation_recovered(self) -> None:
        rng = np.random.default_rng(2)
        n_tokens, n_features = 200, 15
        F = torch.tensor(rng.standard_normal((n_tokens, n_features)), dtype=torch.float32)
        perm = rng.permutation(n_features)
        F_perm = F[:, perm]
        matching = activation_hungarian(F, F_perm)
        expected = np.argsort(perm)
        assert np.array_equal(matching, expected)


class TestGWMatching:
    def test_recovers_permutation_on_distinct_features(self) -> None:
        """GW recovers a permutation when feature directions are distinct."""
        rng = np.random.default_rng(3)
        n = 12
        # Pull rows from a higher-dim space so cosine distances are well-separated.
        W = torch.tensor(rng.standard_normal((n, 32)), dtype=torch.float32)
        perm = rng.permutation(n)
        W_perm = W[perm]

        matching = gw_matching(W, W_perm, epsilon=1e-3, max_iter=2000)
        expected = np.argsort(perm)

        # GW with argmax decoding may not be perfect — allow a few mismatches.
        accuracy = (matching == expected).mean()
        assert accuracy >= 0.8, f"GW accuracy {accuracy} too low"

    def test_handles_unequal_size(self) -> None:
        """GW should run (not crash) when sizes differ."""
        rng = np.random.default_rng(4)
        W_a = torch.tensor(rng.standard_normal((10, 20)), dtype=torch.float32)
        W_b = torch.tensor(rng.standard_normal((15, 20)), dtype=torch.float32)
        matching = gw_matching(W_a, W_b, epsilon=5e-3, max_iter=1000)
        assert matching.shape == (10,)
        assert matching.min() >= 0
        assert matching.max() < 15


class TestFusedGWMatching:
    def test_alpha_1_recovers_cosine_hungarian(self) -> None:
        """At alpha=1 (pure Wasserstein), fused-GW should agree with cosine-Hungarian
        on the easy case where decoder rows are clearly distinguishable."""
        rng = np.random.default_rng(10)
        n = 15
        W = torch.tensor(rng.standard_normal((n, 32)), dtype=torch.float32)
        perm = rng.permutation(n)
        W_perm = W[perm]

        m_cos = cosine_hungarian(W, W_perm)
        m_fgw = fused_gw_matching(W, W_perm, alpha=1.0, epsilon=1e-3, max_iter=2000)

        # Both should recover the same permutation. With argmax decoding, fused-GW
        # may have rare mismatches; allow >=80% agreement.
        agreement = (m_cos == m_fgw).mean()
        assert agreement >= 0.8, f"alpha=1 fused-GW vs Hungarian agreement {agreement} too low"

    def test_handles_unequal_size(self) -> None:
        rng = np.random.default_rng(11)
        W_a = torch.tensor(rng.standard_normal((10, 20)), dtype=torch.float32)
        W_b = torch.tensor(rng.standard_normal((15, 20)), dtype=torch.float32)
        m = fused_gw_matching(W_a, W_b, alpha=0.5, epsilon=5e-3, max_iter=1000)
        assert m.shape == (10,)
        assert m.min() >= 0
        assert m.max() < 15

    def test_invalid_alpha_raises(self) -> None:
        rng = np.random.default_rng(12)
        W_a = torch.tensor(rng.standard_normal((5, 10)), dtype=torch.float32)
        W_b = torch.tensor(rng.standard_normal((5, 10)), dtype=torch.float32)
        with pytest.raises(ValueError, match="alpha must be"):
            fused_gw_matching(W_a, W_b, alpha=1.5)
