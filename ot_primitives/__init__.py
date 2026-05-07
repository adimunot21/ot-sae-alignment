"""Optimal transport primitives implemented from scratch.

Phase 1 of the OT-SAE alignment project. See docs/PROJECT_PLAN_PHASE_1.md.
"""

from ot_primitives.closed_form import bures_wasserstein
from ot_primitives.costs import squared_euclidean_cost
from ot_primitives.divergence import (
    sinkhorn_divergence,
    sinkhorn_divergence_from_points,
)
from ot_primitives.exact import exact_ot, wasserstein_1d
from ot_primitives.sinkhorn import SinkhornResult, sinkhorn

__version__ = "0.1.0.dev0"

__all__ = [
    "SinkhornResult",
    "bures_wasserstein",
    "exact_ot",
    "sinkhorn",
    "sinkhorn_divergence",
    "sinkhorn_divergence_from_points",
    "squared_euclidean_cost",
    "wasserstein_1d",
]
