"""Internal utilities — marginal validators and shape checkers.

Not part of the public API. Used by other modules to sanity-check inputs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Numerical tolerances for marginal validation.
# Tight enough to catch bugs, loose enough to not trip on float64 round-off.
_MARGINAL_RTOL = 1e-9
_MARGINAL_ATOL = 1e-12


def as_probability_vector(a: ArrayLike, name: str = "a") -> NDArray[np.float64]:
    """Validate and return ``a`` as a 1-D float64 probability vector.

    Checks that ``a`` is 1-D, non-negative, and sums to 1 within tolerance.

    Parameters
    ----------
    a
        Array-like to validate.
    name
        Name used in error messages (``"a"``, ``"b"``, etc.).

    Returns
    -------
    a_arr
        ``a`` cast to ``np.float64`` 1-D array.

    Raises
    ------
    ValueError
        If ``a`` is not 1-D, contains negatives, or does not sum to 1.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    if a_arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a_arr.shape}")
    if np.any(a_arr < 0):
        raise ValueError(f"{name} must be non-negative, got min {a_arr.min()}")
    total = a_arr.sum()
    if not np.isclose(total, 1.0, rtol=_MARGINAL_RTOL, atol=_MARGINAL_ATOL):
        raise ValueError(f"{name} must sum to 1, got {total}")
    return a_arr


def validate_cost_matrix(C: ArrayLike, n: int, m: int, name: str = "C") -> NDArray[np.float64]:
    """Validate and return ``C`` as an (n, m) float64 cost matrix.

    Checks shape and that all entries are finite. Does not enforce non-negativity:
    OT is well-defined for arbitrary real costs (only the optimal value depends on
    the costs; the optimal plan is invariant under additive shifts of c that
    preserve a + b structure — see Peyré & Cuturi §2.3).

    Parameters
    ----------
    C
        Cost matrix.
    n, m
        Expected number of source and target points.
    name
        Name used in error messages.

    Returns
    -------
    C_arr
        ``C`` cast to ``np.float64`` 2-D array of shape ``(n, m)``.
    """
    C_arr = np.asarray(C, dtype=np.float64)
    if C_arr.shape != (n, m):
        raise ValueError(f"{name} must have shape ({n}, {m}), got {C_arr.shape}")
    if not np.all(np.isfinite(C_arr)):
        raise ValueError(f"{name} contains non-finite entries (nan or inf)")
    return C_arr


def marginal_violation(
    P: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64]
) -> float:
    """L1 marginal violation of plan ``P`` w.r.t. marginals ``a``, ``b``.

    Returns ``||P 1 − a||_1 + ||P^T 1 − b||_1``, the sum of absolute differences
    between the plan's row/column sums and the target marginals.

    Used as a Sinkhorn convergence criterion in sub-phase 1.2. Defined here so
    the exact-OT tests can also report it for diagnostic purposes.
    """
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    return float(np.abs(row_sums - a).sum() + np.abs(col_sums - b).sum())
