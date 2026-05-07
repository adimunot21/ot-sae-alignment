"""Exact optimal transport via the Kantorovich linear program.

Two implementations live here:

* ``exact_ot`` — the general Kantorovich LP solved via SciPy's HiGHS solver.
  Works in any dimension at any pair of marginal weights.

* ``wasserstein_1d`` — closed-form W_p^p between equal-weight 1-D empirical
  measures via the sort-based monotone-rearrangement formula.

The 1-D closed form is exact and acts as an oracle for the LP solver: on 1-D
inputs they must agree to numerical precision, otherwise one of them has a bug.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

from ot_primitives._utils import as_probability_vector, validate_cost_matrix


def exact_ot(
    a: ArrayLike,
    b: ArrayLike,
    C: ArrayLike,
) -> tuple[float, NDArray[np.float64]]:
    """Exact Kantorovich optimal transport via linear programming.

    Solves

        min_P  <P, C>_F   subject to   P 1 = a,  P^T 1 = b,  P >= 0,

    where the unknown ``P`` is the (n, m) transport plan and the inner product
    is the Frobenius (entrywise) inner product.

    The LP is encoded for SciPy's HiGHS solver as follows. Let ``p = P.ravel()``
    of length ``n*m`` (row-major). The objective vector is ``c = C.ravel()``.
    The marginal constraints become a sparse equality system:

        [I_n ⊗ 1_m^T]      [a]
        [          ] p  =  [ ]
        [1_n^T ⊗ I_m]      [b]

    where the first ``n`` rows enforce row-sums and the next ``m`` rows enforce
    column-sums. The bounds ``P_ij >= 0`` are passed via ``bounds=(0, None)``.

    Parameters
    ----------
    a
        Source probability vector of length ``n``. Non-negative, sums to 1.
    b
        Target probability vector of length ``m``. Non-negative, sums to 1.
    C
        Cost matrix of shape ``(n, m)``. Must be finite.

    Returns
    -------
    cost
        Optimal transport cost ``<P*, C>_F`` (a scalar).
    plan
        Optimal transport plan ``P*`` of shape ``(n, m)``.

    Raises
    ------
    ValueError
        If shapes are inconsistent or marginals are invalid.
    RuntimeError
        If the LP solver fails to converge (very rare for well-posed inputs).

    Notes
    -----
    Tractable up to roughly n = m ≈ 500 in this implementation. Beyond that,
    use network-simplex (POT's ``ot.emd``) or switch to entropic regularization
    (``sinkhorn`` in sub-phase 1.2).
    """
    a_arr = as_probability_vector(a, name="a")
    b_arr = as_probability_vector(b, name="b")
    n = a_arr.size
    m = b_arr.size
    C_arr = validate_cost_matrix(C, n=n, m=m, name="C")

    # Flatten the cost matrix in row-major order to match the LP variable layout.
    c_obj = C_arr.ravel()

    # Build the equality constraint matrix [A_row; A_col] of shape (n + m, n*m).
    #
    # A_row has shape (n, n*m): row i has ones in columns [i*m, i*m + m), zeros
    # elsewhere — this enforces sum_j P_ij = a_i.
    #
    # A_col has shape (m, n*m): row j has ones in columns [j, j + m, j + 2m, ...] —
    # this enforces sum_i P_ij = b_j.
    A_row = np.kron(np.eye(n), np.ones((1, m)))  # (n, n*m)
    A_col = np.kron(np.ones((1, n)), np.eye(m))  # (m, n*m)
    A_eq = np.vstack([A_row, A_col])  # (n + m, n*m)
    b_eq = np.concatenate([a_arr, b_arr])  # (n + m,)

    result = linprog(
        c=c_obj,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )

    if not result.success:
        raise RuntimeError(f"linprog failed: status={result.status}, message={result.message!r}")

    plan = result.x.reshape(n, m)
    cost = float(result.fun)
    return cost, plan


def wasserstein_1d(
    x: ArrayLike,
    y: ArrayLike,
    p: float = 2.0,
) -> float:
    """Closed-form W_p between two 1-D empirical measures with equal weights.

    For two point clouds ``x = (x_1, ..., x_n)`` and ``y = (y_1, ..., y_n)`` of
    equal size, treated as the empirical measures (1/n) Σ δ_{x_i} and
    (1/n) Σ δ_{y_i}, the p-Wasserstein distance has the closed form

        W_p(α, β)^p  =  (1/n) Σ_i  |x_(i) − y_(i)|^p

    where x_(i), y_(i) are the i-th order statistics (sorted in increasing
    order). The optimal transport plan matches sorted x to sorted y in order;
    this is the 1-D specialization of cyclical monotonicity of the optimal
    transport plan. See ``docs/derivations.md`` for the derivation.

    Parameters
    ----------
    x, y
        1-D arrays of equal length ``n``.
    p
        Order of the Wasserstein distance. Must be ``p >= 1``.

    Returns
    -------
    distance
        ``W_p(α, β)``, the p-th root of the average sorted-difference power.

    Raises
    ------
    ValueError
        If inputs are not 1-D, lengths differ, or ``p < 1``.

    Notes
    -----
    Restricted to the equal-weight, equal-size case. The general 1-D formula
    uses CDF inverses and works for arbitrary marginals; we don't need it for
    Phase 1.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError(f"x and y must be 1-D; got shapes {x_arr.shape}, {y_arr.shape}")
    if x_arr.size != y_arr.size:
        raise ValueError(f"x and y must have equal length; got {x_arr.size} and {y_arr.size}")
    if p < 1.0:
        raise ValueError(f"p must be >= 1, got {p}")

    x_sorted = np.sort(x_arr)
    y_sorted = np.sort(y_arr)
    diffs = np.abs(x_sorted - y_sorted)
    return float(np.mean(diffs**p) ** (1.0 / p))
