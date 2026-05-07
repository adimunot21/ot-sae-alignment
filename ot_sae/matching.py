"""Three ways to match features between two SAEs."""

from __future__ import annotations

import numpy as np
import ot
import torch
from scipy.optimize import linear_sum_assignment


def _normalize_rows(W: torch.Tensor) -> torch.Tensor:
    """L2-normalize rows."""
    norms = W.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return W / norms


def cosine_hungarian(
    W_dec_a: torch.Tensor,
    W_dec_b: torch.Tensor,
) -> np.ndarray:
    """Hungarian matching on cosine similarity between decoder rows.

    Supports rectangular inputs: ``W_dec_a`` and ``W_dec_b`` may have
    different numbers of rows. ``scipy.optimize.linear_sum_assignment``
    handles rectangular cost matrices by leaving the surplus rows/columns
    unmatched.

    Parameters
    ----------
    W_dec_a, W_dec_b
        Decoder weight matrices, shape (n_features, d_model). Sizes can differ.

    Returns
    -------
    matching : np.ndarray of shape (n_features_a,)
        matching[i] = j means feature i in A is matched to feature j in B.
        For i not assigned (only possible if n_a > n_b), matching[i] = -1.
    """
    if W_dec_a.shape[1] != W_dec_b.shape[1]:
        raise ValueError(
            f"cosine_hungarian requires same d_model, got {W_dec_a.shape[1]} vs {W_dec_b.shape[1]}"
        )

    W_a = _normalize_rows(W_dec_a).detach().numpy()
    W_b = _normalize_rows(W_dec_b).detach().numpy()

    sim = W_a @ W_b.T
    cost = -sim

    row_ind, col_ind = linear_sum_assignment(cost)

    # Build full-size output: matching[i] = j or -1 if unassigned.
    n_a = W_a.shape[0]
    matching = np.full(n_a, -1, dtype=np.int64)
    matching[row_ind] = col_ind
    return matching


def activation_hungarian(
    F_a: torch.Tensor,
    F_b: torch.Tensor,
) -> np.ndarray:
    """Hungarian matching on activation-pattern correlation.

    Parameters
    ----------
    F_a, F_b
        Feature activation matrices on a shared token set, shape (n_tokens, n_features).
        Both must have been computed on the *same* tokens in the same order.

    Returns
    -------
    matching : np.ndarray of shape (n_features,)
    """
    if F_a.shape[0] != F_b.shape[0]:
        raise ValueError(
            f"activation_hungarian needs same n_tokens, got {F_a.shape[0]} vs {F_b.shape[0]}"
        )

    # Center each feature column.
    F_a_c = F_a - F_a.mean(dim=0, keepdim=True)
    F_b_c = F_b - F_b.mean(dim=0, keepdim=True)

    # Pearson correlation matrix between columns of F_a and columns of F_b.
    # Allowed to be rectangular: n_features can differ between A and B.
    a_norms = F_a_c.norm(dim=0).clamp(min=1e-12)
    b_norms = F_b_c.norm(dim=0).clamp(min=1e-12)
    F_a_n = F_a_c / a_norms
    F_b_n = F_b_c / b_norms
    corr = (F_a_n.T @ F_b_n).detach().numpy()

    cost = -corr
    row_ind, col_ind = linear_sum_assignment(cost)

    # Rectangular case: features in A that don't get a partner are -1.
    n_a = F_a.shape[1]
    matching = np.full(n_a, -1, dtype=np.int64)
    matching[row_ind] = col_ind
    return matching


def gw_matching(
    W_dec_a: torch.Tensor,
    W_dec_b: torch.Tensor,
    epsilon: float = 5e-3,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Entropic Gromov-Wasserstein matching on within-SAE cosine distance.

    Builds two within-SAE pairwise cosine *distance* matrices (one per SAE).
    Runs entropic GW. Decodes the soft coupling via argmax.

    Parameters
    ----------
    W_dec_a, W_dec_b
        Decoder matrices. Shapes can differ in n_features.
    epsilon
        GW regularization strength.

    Returns
    -------
    matching : np.ndarray of shape (n_features_a,)
        matching[i] = j means feature i in A is best-matched to feature j in B.
    """
    W_a = _normalize_rows(W_dec_a).detach().numpy().astype(np.float64)
    W_b = _normalize_rows(W_dec_b).detach().numpy().astype(np.float64)

    # Within-SAE cosine distance: 1 - cos_sim.
    C1 = 1.0 - W_a @ W_a.T
    C2 = 1.0 - W_b @ W_b.T
    # Numerical floor.
    np.clip(C1, 0.0, 2.0, out=C1)
    np.clip(C2, 0.0, 2.0, out=C2)

    n_a, n_b = C1.shape[0], C2.shape[0]
    p = np.full(n_a, 1.0 / n_a)
    q = np.full(n_b, 1.0 / n_b)

    T = ot.gromov.entropic_gromov_wasserstein(
        C1,
        C2,
        p,
        q,
        loss_fun="square_loss",
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
    )

    matching = np.argmax(T, axis=1)
    return matching


def fused_gw_matching(
    W_dec_a: torch.Tensor,
    W_dec_b: torch.Tensor,
    alpha: float = 0.5,
    epsilon: float = 5e-3,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> np.ndarray:
    r"""Entropic Fused Gromov-Wasserstein matching.

    Combines cross-side cosine distance (Wasserstein term) with within-side
    cosine distance (GW term), weighted by ``alpha``.

    The objective is

        FGW = alpha * <M, T>  +  (1 - alpha) * <|C1 - C2|^2, T \otimes T>  -  eps * H(T)

    where M[i, j] = 1 - cos_sim(W_dec_a[i], W_dec_b[j]) is the cross-side cost,
    C1, C2 are within-side cosine distances.

    Parameters
    ----------
    W_dec_a, W_dec_b
        Decoder matrices, shape (n_features, d_model). Sizes can differ.
    alpha
        Fusion parameter in [0, 1]. alpha=1 is pure Wasserstein, alpha=0 is pure GW.
    epsilon
        Entropic regularization strength.

    Returns
    -------
    matching : np.ndarray of shape (n_features_a,)
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    W_a = _normalize_rows(W_dec_a).detach().numpy().astype(np.float64)
    W_b = _normalize_rows(W_dec_b).detach().numpy().astype(np.float64)

    # Cross-side cost: 1 - cosine similarity. In [0, 2].
    M = 1.0 - W_a @ W_b.T
    np.clip(M, 0.0, 2.0, out=M)

    # Within-side cosine distances.
    C1 = 1.0 - W_a @ W_a.T
    C2 = 1.0 - W_b @ W_b.T
    np.clip(C1, 0.0, 2.0, out=C1)
    np.clip(C2, 0.0, 2.0, out=C2)

    n_a, n_b = C1.shape[0], C2.shape[0]
    p = np.full(n_a, 1.0 / n_a)
    q = np.full(n_b, 1.0 / n_b)

    # POT's entropic fused-GW. Note: POT's "alpha" weights the GW term, not
    # the Wasserstein term — opposite of the convention used in our docstring.
    # We pass (1 - alpha) so our convention matches the docstring.
    T = ot.gromov.entropic_fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p,
        q,
        loss_fun="square_loss",
        epsilon=epsilon,
        alpha=1.0 - alpha,
        max_iter=max_iter,
        tol=tol,
    )

    return np.argmax(T, axis=1)
