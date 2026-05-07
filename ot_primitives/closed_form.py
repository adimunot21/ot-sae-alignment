"""Closed-form Wasserstein distances.

Some problems admit exact analytical answers without any optimization. These
serve as oracles for the iterative and LP-based solvers in the rest of the
library.

Currently implemented:

* ``bures_wasserstein``: W_2 between multivariate Gaussians via the
  Bures-Wasserstein formula.

The 1-D sort-based formula lives in ``ot_primitives.exact.wasserstein_1d``
because it interleaves with the LP-based ``exact_ot`` testing infrastructure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import sqrtm

# Tolerance for the imaginary-part check on matrix square roots.
# scipy's sqrtm uses Schur decomposition and can produce small imaginary
# components even for real symmetric inputs due to floating-point noise.
# Anything below this is rounded away; above it, we raise.
_IMAG_TOL = 1e-8


def _real_matrix_sqrt(M: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute matrix square root, returning a real array.

    Validates that any imaginary component is below ``_IMAG_TOL``.

    Raises
    ------
    RuntimeError
        If ``sqrtm`` returns a result with a non-trivial imaginary part —
        a sign that ``M`` is not actually positive semi-definite.
    """
    sqrt = sqrtm(M)
    if np.iscomplexobj(sqrt):
        max_imag = float(np.max(np.abs(sqrt.imag)))
        if max_imag > _IMAG_TOL:
            raise RuntimeError(
                f"sqrtm returned a complex matrix with max |imag| = {max_imag:.2e}; "
                f"input may not be positive semi-definite."
            )
        sqrt = sqrt.real
    return np.asarray(sqrt, dtype=np.float64)


def bures_wasserstein(
    mu1: ArrayLike,
    cov1: ArrayLike,
    mu2: ArrayLike,
    cov2: ArrayLike,
) -> float:
    """Closed-form W_2 between two multivariate Gaussians.

    For ``alpha = N(mu1, cov1)`` and ``beta = N(mu2, cov2)``, the squared
    2-Wasserstein distance is

        W_2^2(alpha, beta) =
            ||mu1 - mu2||^2
            + tr(cov1 + cov2 - 2 * (cov1^(1/2) cov2 cov1^(1/2))^(1/2)).

    The covariance term is the Bures metric on positive semi-definite
    matrices. See Olkin & Pukelsheim (1982); Peyré & Cuturi (2019), §2.6.

    Parameters
    ----------
    mu1, mu2
        Mean vectors of length ``d``.
    cov1, cov2
        Covariance matrices of shape ``(d, d)``. Must be symmetric and
        positive semi-definite. (We check symmetry; PSD-ness is checked
        implicitly via the matrix-square-root call.)

    Returns
    -------
    distance
        ``W_2(alpha, beta)``, the 2-Wasserstein distance (not squared).

    Raises
    ------
    ValueError
        If shapes are inconsistent or covariances are not symmetric.
    RuntimeError
        If covariances are not PSD (manifesting as complex sqrtm output).

    Notes
    -----
    Returns the distance, not the squared distance, to match the convention
    of ``wasserstein_1d``. To get ``W_2^2`` (which is what the Sinkhorn
    cost computes when the cost matrix is squared Euclidean), square the
    output.
    """
    mu1_arr = np.asarray(mu1, dtype=np.float64)
    mu2_arr = np.asarray(mu2, dtype=np.float64)
    cov1_arr = np.asarray(cov1, dtype=np.float64)
    cov2_arr = np.asarray(cov2, dtype=np.float64)

    if mu1_arr.ndim != 1 or mu2_arr.ndim != 1:
        raise ValueError(f"mu1, mu2 must be 1-D; got shapes {mu1_arr.shape}, {mu2_arr.shape}")
    if mu1_arr.size != mu2_arr.size:
        raise ValueError(f"mu1, mu2 must have same length; got {mu1_arr.size} and {mu2_arr.size}")
    d = mu1_arr.size
    if cov1_arr.shape != (d, d) or cov2_arr.shape != (d, d):
        raise ValueError(f"cov1, cov2 must be ({d}, {d}); got {cov1_arr.shape}, {cov2_arr.shape}")

    # Symmetrize defensively against tiny float asymmetries in user-provided
    # covariances. If the asymmetry is large, that's a real bug — flag it.
    asym1 = float(np.max(np.abs(cov1_arr - cov1_arr.T)))
    asym2 = float(np.max(np.abs(cov2_arr - cov2_arr.T)))
    if asym1 > 1e-6:
        raise ValueError(f"cov1 is not symmetric (max asymmetry {asym1:.2e})")
    if asym2 > 1e-6:
        raise ValueError(f"cov2 is not symmetric (max asymmetry {asym2:.2e})")
    cov1_arr = 0.5 * (cov1_arr + cov1_arr.T)
    cov2_arr = 0.5 * (cov2_arr + cov2_arr.T)

    # Mean term.
    mean_term = float(np.sum((mu1_arr - mu2_arr) ** 2))

    # Covariance term: tr(cov1 + cov2 - 2 * (cov1^(1/2) cov2 cov1^(1/2))^(1/2)).
    cov1_sqrt = _real_matrix_sqrt(cov1_arr)
    inner = cov1_sqrt @ cov2_arr @ cov1_sqrt
    inner_sqrt = _real_matrix_sqrt(inner)
    cov_term = float(np.trace(cov1_arr) + np.trace(cov2_arr) - 2.0 * np.trace(inner_sqrt))

    # Numerical floor: very small negative values can appear from sqrtm noise.
    # The true squared distance is non-negative.
    w2_squared = max(mean_term + cov_term, 0.0)
    return float(np.sqrt(w2_squared))
