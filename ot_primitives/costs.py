"""Ground cost functions.

Cost matrices in OT measure how expensive it is to transport mass from each
source point to each target point. The most common choice in ML is squared
Euclidean distance, which makes OT correspond to W_2 (the 2-Wasserstein
distance) and is what Brenier's theorem is stated for.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def squared_euclidean_cost(X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
    """Squared Euclidean cost matrix.

    Computes ``C[i, j] = ||X[i] - Y[j]||_2^2`` for two point clouds.

    Uses the algebraic identity
    ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x, y>``
    rather than broadcasting subtraction, which is faster for large n/m and
    is the standard formulation in scikit-learn / POT / OTT-JAX.

    Parameters
    ----------
    X
        Source points, shape ``(n, d)``.
    Y
        Target points, shape ``(m, d)``.

    Returns
    -------
    C
        Cost matrix of shape ``(n, m)``, dtype float64.
        Entries are clipped at zero to suppress small negative values from
        floating-point cancellation when X[i] ≈ Y[j].
    """
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = np.asarray(Y, dtype=np.float64)
    if X_arr.ndim != 2 or Y_arr.ndim != 2:
        raise ValueError(f"X and Y must be 2-D; got shapes {X_arr.shape}, {Y_arr.shape}")
    if X_arr.shape[1] != Y_arr.shape[1]:
        raise ValueError(
            f"X and Y must have same dimension; got {X_arr.shape[1]} vs {Y_arr.shape[1]}"
        )

    X_sq = np.sum(X_arr * X_arr, axis=1, keepdims=True)  # (n, 1)
    Y_sq = np.sum(Y_arr * Y_arr, axis=1, keepdims=True).T  # (1, m)
    cross = X_arr @ Y_arr.T  # (n, m)
    C = X_sq + Y_sq - 2.0 * cross
    return np.clip(C, a_min=0.0, a_max=None)
