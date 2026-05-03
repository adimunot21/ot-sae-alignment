"""Smoke test for the Phase 0 environment.

Verifies that all required dependencies import and that a minimal OT computation
runs and gives the right answer. Exit 0 on success, 1 on failure.

This script does NOT exercise the `ot_primitives` library (which is intentionally
empty at the end of Phase 0). It validates the tools we'll use to build it.
"""

from __future__ import annotations

import sys


def main() -> int:
    # Imports
    try:
        import matplotlib
        import numpy as np
        import ot  # POT
        import scipy
        import scipy.optimize  # noqa: F401
        import skimage  # noqa: F401
    except ImportError as exc:
        print(f"FAIL: import error: {exc}", file=sys.stderr)
        return 1

    print(f"Python      : {sys.version.split()[0]}")
    print(f"NumPy       : {np.__version__}")
    print(f"SciPy       : {scipy.__version__}")
    print(f"Matplotlib  : {matplotlib.__version__}")
    print(f"POT         : {ot.__version__}")

    # Minimal OT problem with a known answer.
    # Source measure: 0.5 mass at x=0, 0.5 mass at x=1.
    # Target measure: 0.5 mass at y=2, 0.5 mass at y=3.
    # Cost: squared Euclidean.
    # By inspection (or by the LP), the optimal plan is monotone:
    #   0 -> 2 with mass 0.5  (cost 4)
    #   1 -> 3 with mass 0.5  (cost 4)
    # Total cost = 0.5 * 4 + 0.5 * 4 = 4.0.
    a = np.array([0.5, 0.5])
    b = np.array([0.5, 0.5])
    C = np.array([[4.0, 9.0], [1.0, 4.0]])
    expected_cost = 4.0

    plan = ot.emd(a, b, C)
    cost = float(np.sum(plan * C))

    if abs(cost - expected_cost) > 1e-10:
        print(
            f"FAIL: expected OT cost {expected_cost}, got {cost}",
            file=sys.stderr,
        )
        return 1

    print()
    print("OT smoke test (uniform 2-point measures, squared-Euclidean cost):")
    print(f"  expected cost = {expected_cost}")
    print(f"  POT cost      = {cost}")
    print(f"  plan          = {plan.tolist()}")
    print()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
