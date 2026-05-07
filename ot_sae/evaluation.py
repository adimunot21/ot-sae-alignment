"""Evaluation: how good is a feature matching?"""

from __future__ import annotations

import numpy as np
import torch


def evaluate_matching(
    matching: np.ndarray,
    F_a_eval: torch.Tensor,
    F_b_eval: torch.Tensor,
) -> dict:
    """Compute matched-pair activation correlation on held-out features.

    Parameters
    ----------
    matching
        Output of one of the matching functions. Shape (n_features_a,).
    F_a_eval, F_b_eval
        Held-out feature activations on the SAME held-out tokens.
        Shapes (n_eval_tokens, n_features_a) and (n_eval_tokens, n_features_b).

    Returns
    -------
    A dict with summary statistics:
      mean_corr, median_corr, q25_corr, q75_corr,
      frac_above_0p3, frac_above_0p5, frac_above_0p7,
      n_pairs : the count of (i, matching[i]) pairs evaluated.
    """
    if F_a_eval.shape[0] != F_b_eval.shape[0]:
        raise ValueError("F_a_eval and F_b_eval must have same n_tokens")

    F_a_c = F_a_eval - F_a_eval.mean(dim=0, keepdim=True)
    F_b_c = F_b_eval - F_b_eval.mean(dim=0, keepdim=True)
    a_norms = F_a_c.norm(dim=0).clamp(min=1e-12)
    b_norms = F_b_c.norm(dim=0).clamp(min=1e-12)
    F_a_n = F_a_c / a_norms
    F_b_n = F_b_c / b_norms

    # For each i, correlation between feature i in A and feature matching[i] in B.
    # Skip i where matching[i] == -1 (rectangular case with surplus on A side).
    matching_arr = np.asarray(matching)
    matched_mask = matching_arr >= 0
    matching_t = torch.as_tensor(matching_arr.clip(min=0), dtype=torch.long)
    matched_b = F_b_n[:, matching_t]  # (n_tokens, n_features_a)
    correlations = (F_a_n * matched_b).sum(dim=0).numpy()  # (n_features_a,)
    correlations = correlations[matched_mask]

    # If a feature in A never fires on eval tokens, its correlation is 0/nan;
    # filter those out.
    finite_mask = np.isfinite(correlations)
    correlations = correlations[finite_mask]

    return {
        "mean_corr": float(correlations.mean()),
        "median_corr": float(np.median(correlations)),
        "q25_corr": float(np.quantile(correlations, 0.25)),
        "q75_corr": float(np.quantile(correlations, 0.75)),
        "frac_above_0p3": float((correlations > 0.3).mean()),
        "frac_above_0p5": float((correlations > 0.5).mean()),
        "frac_above_0p7": float((correlations > 0.7).mean()),
        "n_pairs": int(len(correlations)),
    }
