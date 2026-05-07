"""Sinkhorn divergence — debiased entropic optimal transport.

The entropic OT cost is biased: ``OT_eps(alpha, alpha) > 0``. Genevay, Peyré,
and Cuturi (2018) introduced the *Sinkhorn divergence* to remove this bias:

    S_eps(alpha, beta)
        = OT_eps(alpha, beta)
          - 0.5 * OT_eps(alpha, alpha)
          - 0.5 * OT_eps(beta, beta).

Properties (Feydy et al. 2019):
- S_eps(alpha, alpha) = 0 exactly (when computed from dual potentials).
- S_eps(alpha, beta) >= 0.
- S_eps interpolates between W_2^2 (eps -> 0) and MMD^2 (eps -> infinity)
  with a kernel determined by the cost.

Two forms are implemented; they compute related but DIFFERENT objects:

* ``"cost"``: subtract transport costs ``<P, C>`` of the three regularized
  plans. This is what POT's ``empirical_sinkhorn_divergence`` computes.

* ``"dual"``: subtract regularized OT objectives ``<P, C> - eps * H(P)``,
  computed efficiently from the dual potentials as ``<f, a> + <g, b>``
  (the constant ``eps`` shifts cancel across the three terms). This is the
  formal definition in Feydy et al. 2019.

The two forms differ by an entropy-gap term:

    S_dual = S_cost - eps * (H(P_xy) - 0.5 H(P_xx) - 0.5 H(P_yy)) .

They both equal zero on self-self, both approach W_2^2 as eps -> 0, and both
are non-negative. For typical distinct measures, S_dual < S_cost (the
cross-coupling P_xy is more spread out than the self-couplings).

See POT issue #383 for the corresponding distinction in that library.

References
----------
Genevay, Peyré, Cuturi (2018), "Learning Generative Models with Sinkhorn
    Divergences."
Feydy, Séjourné, Vialard, Amari, Trouvé, Peyré (2019),
    "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences."
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ot_primitives._utils import as_probability_vector, validate_cost_matrix
from ot_primitives.sinkhorn import sinkhorn


def sinkhorn_divergence(
    a: ArrayLike,
    b: ArrayLike,
    C_xy: ArrayLike,
    C_xx: ArrayLike,
    C_yy: ArrayLike,
    eps: float,
    *,
    form: Literal["cost", "dual"] = "cost",
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> float:
    """Sinkhorn divergence ``S_eps(alpha, beta)``.

    Computes the debiased Sinkhorn divergence between two empirical measures
    ``alpha = sum_i a_i delta_{x_i}`` and ``beta = sum_j b_j delta_{y_j}``
    via three Sinkhorn problems.

    Parameters
    ----------
    a, b
        Source and target probability vectors of length n and m.
    C_xy
        Cross cost matrix of shape (n, m): cost from x_i to y_j.
    C_xx
        Self-cost matrix of shape (n, n): cost from x_i to x_j.
    C_yy
        Self-cost matrix of shape (m, m): cost from y_i to y_j.
    eps
        Regularization strength (shared across all three Sinkhorn runs).
    form
        Which form to compute. ``"dual"`` is recommended; ``"cost"`` is
        included for pedagogical comparison.
    max_iter
        Per-Sinkhorn-run iteration cap.
    tol
        Per-Sinkhorn-run marginal-violation tolerance.

    Returns
    -------
    divergence
        ``S_eps(alpha, beta)``, a non-negative scalar.

    Raises
    ------
    ValueError
        If shapes are inconsistent or marginals are invalid.
    """
    a_arr = as_probability_vector(a, name="a")
    b_arr = as_probability_vector(b, name="b")
    n, m = a_arr.size, b_arr.size

    C_xy_arr = validate_cost_matrix(C_xy, n=n, m=m, name="C_xy")
    C_xx_arr = validate_cost_matrix(C_xx, n=n, m=n, name="C_xx")
    C_yy_arr = validate_cost_matrix(C_yy, n=m, m=m, name="C_yy")

    if form not in ("cost", "dual"):
        raise ValueError(f"form must be 'cost' or 'dual', got {form!r}")

    res_xy = sinkhorn(a_arr, b_arr, C_xy_arr, eps=eps, max_iter=max_iter, tol=tol)
    res_xx = sinkhorn(a_arr, a_arr, C_xx_arr, eps=eps, max_iter=max_iter, tol=tol)
    res_yy = sinkhorn(b_arr, b_arr, C_yy_arr, eps=eps, max_iter=max_iter, tol=tol)

    if form == "cost":
        # Conceptually direct: subtract transport costs of the three plans.
        return float(res_xy.cost - 0.5 * res_xx.cost - 0.5 * res_yy.cost)

    # Dual-potential form. The regularized objective at the optimum equals
    # <f, a> + <g, b> by Lagrangian duality (Peyré & Cuturi §4.4).
    obj_xy = float(res_xy.f @ a_arr + res_xy.g @ b_arr)
    obj_xx = float(res_xx.f @ a_arr + res_xx.g @ a_arr)
    obj_yy = float(res_yy.f @ b_arr + res_yy.g @ b_arr)
    return float(obj_xy - 0.5 * obj_xx - 0.5 * obj_yy)


def sinkhorn_divergence_from_points(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    a: ArrayLike | None = None,
    b: ArrayLike | None = None,
    eps: float = 0.1,
    *,
    form: Literal["cost", "dual"] = "cost",
    max_iter: int = 1000,
    tol: float = 1e-9,
) -> float:
    """Sinkhorn divergence from raw point clouds, with squared-Euclidean cost.

    Convenience wrapper that computes the three required cost matrices
    automatically. Uses uniform marginals if ``a`` or ``b`` is ``None``.

    Parameters
    ----------
    X
        Source points of shape (n, d).
    Y
        Target points of shape (m, d).
    a
        Source probability vector. Defaults to uniform ``1/n``.
    b
        Target probability vector. Defaults to uniform ``1/m``.
    eps
        Regularization strength.

    Returns
    -------
    divergence
        ``S_eps(alpha, beta)``.
    """
    from ot_primitives.costs import squared_euclidean_cost

    n, m = X.shape[0], Y.shape[0]
    a = np.full(n, 1.0 / n) if a is None else a
    b = np.full(m, 1.0 / m) if b is None else b

    C_xy = squared_euclidean_cost(X, Y)
    C_xx = squared_euclidean_cost(X, X)
    C_yy = squared_euclidean_cost(Y, Y)

    return sinkhorn_divergence(
        a, b, C_xy, C_xx, C_yy, eps=eps, form=form, max_iter=max_iter, tol=tol
    )
