"""Visualize the bias-variance tradeoff of entropic regularization.

For two random 2D point clouds, computes the exact LP optimal transport cost
once, then computes Sinkhorn cost across a sweep of regularization strengths.
Plots both on a log-scale x-axis.

Expected behavior:
- At small eps (left of plot): Sinkhorn cost approaches the LP cost (dashed line).
- At moderate eps: gap grows; iterations stay fast.
- At large eps: cost approaches <a a^T, C> (uniform plan, no real transport).

Run from repo root:
    python scripts/cost_vs_epsilon.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ot_primitives.costs import squared_euclidean_cost
from ot_primitives.exact import exact_ot
from ot_primitives.sinkhorn import sinkhorn

OUT_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed=0)

    n, m = 50, 50
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    X = rng.standard_normal((n, 2))
    Y = rng.standard_normal((m, 2)) + np.array([2.0, 0.0])
    C = squared_euclidean_cost(X, Y)

    cost_lp, _ = exact_ot(a, b, C)

    # Cost of the uniform "no-transport" plan: <a b^T, C>.
    cost_uniform = float(np.outer(a, b).reshape(-1) @ C.reshape(-1))

    epsilons = np.geomspace(1e-3, 1e2, 25)
    rows = []
    for eps in epsilons:
        result = sinkhorn(a, b, C, eps=float(eps), max_iter=20000, tol=1e-10)
        rows.append(
            {
                "eps": float(eps),
                "cost": result.cost,
                "n_iter": result.n_iter,
                "converged": result.converged,
                "marginal_violation": result.marginal_violation,
            }
        )
        print(
            f"eps={eps:9.5f}  cost={result.cost:.6f}  "
            f"iters={result.n_iter:5d}  converged={result.converged}"
        )

    # Persist results so the plot is regenerable without rerunning Sinkhorn.
    payload = {
        "config": {
            "n": n,
            "m": m,
            "seed": 0,
            "y_shift": [2.0, 0.0],
        },
        "cost_lp": cost_lp,
        "cost_uniform": cost_uniform,
        "sweep": rows,
    }
    json_path = OUT_DIR / "cost_vs_epsilon.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {json_path}")

    # Plot.
    eps_arr = np.array([r["eps"] for r in rows])
    cost_arr = np.array([r["cost"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(eps_arr, cost_arr, "o-", label="Sinkhorn cost")
    ax.axhline(cost_lp, linestyle="--", color="black", label=f"LP cost ({cost_lp:.3f})")
    ax.axhline(
        cost_uniform,
        linestyle=":",
        color="gray",
        label=f"uniform-plan cost ({cost_uniform:.3f})",
    )
    ax.set_xlabel(r"$\varepsilon$")
    ax.set_ylabel("transport cost")
    ax.set_title(r"Sinkhorn cost vs $\varepsilon$ (n=m=50, 2D Gaussians)")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.tight_layout()

    plot_path = OUT_DIR / "cost_vs_epsilon.png"
    fig.savefig(plot_path, dpi=140)
    print(f"saved {plot_path}")


if __name__ == "__main__":
    main()
