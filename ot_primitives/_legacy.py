"""Deprecated implementations kept for educational reference only.

Nothing in this module should be used in production code or tests of the
main library. It exists so the README / paper writeup can reference the
"natural" form of Sinkhorn before justifying the log-domain implementation.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ot_primitives._utils import as_probability_vector, validate_cost_matrix


def sinkhorn_multiplicative(
    a: ArrayLike,
    b: ArrayLike,
    C: ArrayLike,
    eps: float,
    n_iter: int = 1000,
    tol: float = 1e-9,
) -> tuple[float, NDArray[np.float64]]:
    """Multiplicative-form Sinkhorn — DEPRECATED, kept for reference only.

    Implements Sinkhorn–Knopp scaling on ``K = exp(-C / eps)`` directly.
    Underflows whenever ``min(C) / eps`` exceeds about 700 (the float64
    overflow threshold for ``exp``), which is roughly ``eps < min(C) / 700``.
    For typical OT problems with cost magnitudes ~1, this means the
    multiplicative form fails for ``eps`` smaller than about 1e-3.

    Use ``ot_primitives.sinkhorn`` instead. This function emits a
    ``DeprecationWarning`` on call.

    Parameters
    ----------
    a, b
        Probability vectors.
    C
        Cost matrix.
    eps
        Regularization strength.
    n_iter
        Maximum Sinkhorn iterations.
    tol
        Marginal-violation tolerance for early stopping.

    Returns
    -------
    cost
        Transport cost ``<P, C>_F`` of the converged plan.
    plan
        The (possibly NaN-poisoned) transport plan.
    """
    warnings.warn(
        "sinkhorn_multiplicative is deprecated and underflows at small eps; "
        "use ot_primitives.sinkhorn (log-domain) instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    a_arr = as_probability_vector(a, name="a")
    b_arr = as_probability_vector(b, name="b")
    n, m = a_arr.size, b_arr.size
    C_arr = validate_cost_matrix(C, n=n, m=m, name="C")

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    K = np.exp(-C_arr / eps)
    u = np.ones(n)
    v = np.ones(m)

    for _ in range(n_iter):
        u = a_arr / (K @ v)
        v = b_arr / (K.T @ u)
        plan = u[:, None] * K * v[None, :]
        violation = float(
            np.abs(plan.sum(axis=1) - a_arr).sum() + np.abs(plan.sum(axis=0) - b_arr).sum()
        )
        if violation < tol:
            break

    plan = u[:, None] * K * v[None, :]
    cost = float(np.sum(plan * C_arr))
    return cost, plan
