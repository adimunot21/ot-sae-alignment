"""Log-domain Sinkhorn for entropic-regularized optimal transport.

The entropic-regularized Kantorovich problem is

    min_P  <P, C>_F - eps * H(P)
    subject to  P 1 = a,  P^T 1 = b,  P >= 0,

where H(P) = -sum_ij P_ij (log P_ij - 1) is the discrete entropy of the plan.
The optimal plan has the structure P*_ij = u_i K_ij v_j with K_ij = exp(-C_ij/eps),
and Sinkhorn (1967) finds u, v by alternating projections.

This module implements the algorithm in log-domain — substituting f = eps log u
and g = eps log v throughout — so that the dual potentials remain finite even
when eps is small enough to underflow K = exp(-C/eps) to zero.

References
----------
Cuturi (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."
Schmitzer (2019), "Stabilized Sparse Scaling Algorithms for Entropy Regularized
    Transport Problems." — the log-domain stabilization recipe.
Peyré & Cuturi (2019), "Computational Optimal Transport," chapter 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp

from ot_primitives._utils import as_probability_vector, validate_cost_matrix


@dataclass(frozen=True)
class SinkhornResult:
    """Result of a Sinkhorn run.

    Attributes
    ----------
    cost
        Transport cost ``<P, C>_F`` of the converged regularized plan.
        NOT the regularized objective ``<P, C>_F - eps H(P)``.
    plan
        Transport plan ``P`` of shape ``(n, m)``.
    f, g
        Log-domain dual potentials. ``P_ij = exp((f_i + g_j - C_ij) / eps)``.
        Useful for downstream computations that need the duals (e.g. the
        Sinkhorn divergence in sub-phase 1.3).
    n_iter
        Number of iterations actually performed.
    converged
        Whether the marginal-violation criterion was met before ``max_iter``.
    marginal_violation
        Final L1 marginal violation, ``||P 1 - a||_1 + ||P^T 1 - b||_1``.
    """

    cost: float
    plan: NDArray[np.float64]
    f: NDArray[np.float64]
    g: NDArray[np.float64]
    n_iter: int
    converged: bool
    marginal_violation: float


def sinkhorn(
    a: ArrayLike,
    b: ArrayLike,
    C: ArrayLike,
    eps: float,
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> SinkhornResult:
    """Log-domain entropic-regularized optimal transport.

    Solves the entropic-regularized Kantorovich problem and returns the
    transport cost of the regularized plan, the plan itself, and the
    log-domain dual potentials.

    The implementation uses the recursion

        f_i  <-  -eps * logsumexp_j ( (g_j - C_ij) / eps )  +  eps * log(a_i)
        g_j  <-  -eps * logsumexp_i ( (f_i - C_ij) / eps )  +  eps * log(b_j)

    iterated until the L1 marginal violation drops below ``tol`` or
    ``max_iter`` iterations have elapsed. The plan is recovered as
    ``P_ij = exp((f_i + g_j - C_ij) / eps)``.

    Parameters
    ----------
    a
        Source probability vector of length ``n``.
    b
        Target probability vector of length ``m``.
    C
        Cost matrix of shape ``(n, m)``. Must be finite.
    eps
        Regularization strength. Must be positive. Smaller values give
        sharper plans (closer to the LP solution) at the cost of more
        iterations to converge.
    max_iter
        Maximum number of Sinkhorn iterations.
    tol
        Marginal-violation L1 tolerance for early stopping.

    Returns
    -------
    result
        A ``SinkhornResult`` with the converged plan, cost, dual potentials,
        and convergence diagnostics.

    Raises
    ------
    ValueError
        If inputs are invalid (negative marginals, wrong shapes, non-positive
        ``eps``, etc.).

    Notes
    -----
    Stable for ``eps`` down to at least ``1e-3`` for unit-magnitude costs.
    Convergence is geometric in iteration count for fixed ``eps``; the rate
    degrades as ``eps -> 0``. See Altschuler, Weed, Rigollet (2017) for
    formal rates.
    """
    a_arr = as_probability_vector(a, name="a")
    b_arr = as_probability_vector(b, name="b")
    n, m = a_arr.size, b_arr.size
    C_arr = validate_cost_matrix(C, n=n, m=m, name="C")

    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")

    log_a = np.log(a_arr)
    log_b = np.log(b_arr)

    # Dual potentials in log-domain. Initialized to zero — equivalent to
    # u = v = 1 in multiplicative form.
    f = np.zeros(n)
    g = np.zeros(m)

    converged = False
    final_violation = np.inf

    for it in range(1, max_iter + 1):  # noqa: B007
        # Update f: M_ij = (g_j - C_ij) / eps; f_i = -eps * logsumexp_j M_ij + eps log a_i.
        M = (g[None, :] - C_arr) / eps
        f = -eps * logsumexp(M, axis=1) + eps * log_a

        # Update g: M_ij = (f_i - C_ij) / eps; g_j = -eps * logsumexp_i M_ij + eps log b_j.
        M = (f[:, None] - C_arr) / eps
        g = -eps * logsumexp(M, axis=0) + eps * log_b

        # Marginal-violation check. Reconstructing the plan each iteration is
        # O(n*m) and dominates the iteration cost — but it's the right
        # convergence criterion (see Schmitzer 2019, Peyré & Cuturi §4.4).
        log_P = (f[:, None] + g[None, :] - C_arr) / eps
        plan = np.exp(log_P)
        row_sums = plan.sum(axis=1)
        col_sums = plan.sum(axis=0)
        violation = float(np.abs(row_sums - a_arr).sum() + np.abs(col_sums - b_arr).sum())

        if violation < tol:
            converged = True
            final_violation = violation
            break

        final_violation = violation

    log_P = (f[:, None] + g[None, :] - C_arr) / eps
    plan = np.exp(log_P)
    cost = float(np.sum(plan * C_arr))

    return SinkhornResult(
        cost=cost,
        plan=plan,
        f=f,
        g=g,
        n_iter=it,
        converged=converged,
        marginal_violation=final_violation,
    )
