"""Project code for OT-based SAE alignment.

This package uses POT (Python Optimal Transport) directly. The from-scratch
implementations in `ot_primitives` are didactic and not used here.
"""

from ot_sae.activations import collect_sae_features, top_active_features
from ot_sae.evaluation import evaluate_matching
from ot_sae.matching import (
    activation_hungarian,
    cosine_hungarian,
    fused_gw_matching,
    gw_matching,
)

__all__ = [
    "collect_sae_features",
    "top_active_features",
    "cosine_hungarian",
    "activation_hungarian",
    "gw_matching",
    "fused_gw_matching",
    "evaluate_matching",
]
