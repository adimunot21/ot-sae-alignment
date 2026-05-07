# Derivations

By-hand derivations for the closed-form cases that act as oracles in the test suite.

## 1-D Wasserstein via monotone rearrangement

**Claim.** Let α = (1/n) Σ_i δ_{x_i} and β = (1/n) Σ_i δ_{y_i} be empirical
measures on the real line with equal weights and the same number of atoms.
Let x_(1) ≤ ... ≤ x_(n) and y_(1) ≤ ... ≤ y_(n) denote the order statistics.
For any p ≥ 1,

    W_p(α, β)^p = (1/n) Σ_i |x_(i) − y_(i)|^p .

The optimal plan matches sorted source atoms to sorted target atoms in order
(the *monotone rearrangement*).

**Proof sketch.** The optimal coupling π* between two measures on ℝ for any
convex cost c(x, y) of the form c(x − y) — which includes |x − y|^p for p ≥ 1
— is supported on a *monotone non-decreasing set*: if (x, y) and (x', y') are
both in the support of π*, then x ≤ x' implies y ≤ y'. This is a special case
of *cyclical monotonicity* of optimal couplings (Villani 2008, §5; Peyré &
Cuturi 2019, §2.6).

The intuition: suppose the support of π contained two points (x, y) and
(x', y') with x < x' but y > y'. Then "swapping" — sending x → y' and x' → y
instead — would strictly reduce the cost when c is convex in (x − y),
contradicting optimality.

For empirical measures with equal weights and equal numbers of atoms, the
unique monotone coupling is the sort-and-match coupling: x_(i) ↔ y_(i). The
transported cost is then the average of |x_(i) − y_(i)|^p, and W_p is the
p-th root.

This generalizes to non-equal weights via CDF inverses:

    W_p(α, β)^p = ∫_0^1 |F_α^{-1}(u) − F_β^{-1}(u)|^p du .

We don't implement the general form in Phase 1.

## Bures-Wasserstein for Gaussians

**Claim.** For α = N(μ_1, Σ_1) and β = N(μ_2, Σ_2),

    W_2^2(α, β) = ||μ_1 − μ_2||^2 + tr(Σ_1 + Σ_2 − 2(Σ_1^(1/2) Σ_2 Σ_1^(1/2))^(1/2)) .

The covariance term is the *Bures metric* on positive semi-definite matrices.
Reference: Olkin & Pukelsheim (1982); Peyré & Cuturi (2019), §2.6.

**Verification of structural cases.**

*Identical Gaussians* (μ_1 = μ_2, Σ_1 = Σ_2 = Σ):
The mean term is zero. For the covariance term, Σ^(1/2) Σ Σ^(1/2) = Σ^2,
whose square root is Σ. So tr(Σ + Σ − 2Σ) = 0. Total: 0. ✓

*Same covariance, different means* (Σ_1 = Σ_2 = Σ):
The covariance term collapses to zero by the same calculation. Distance is
||μ_1 − μ_2||. ✓

*Diagonal covariances*: when both Σ_i are diagonal, all matrix square roots
are diagonal too, and the formula factors across coordinates. In each
coordinate, the contribution is (σ_{1,i} − σ_{2,i})^2 + (μ_{1,i} − μ_{2,i})^2.

**Why the matrix-square-root form?**
Brenier's theorem gives the optimal map between Gaussians as a linear map
T(x) = A x + b where A satisfies AΣ_1 A^T = Σ_2, leading to
A = Σ_1^(-1/2) (Σ_1^(1/2) Σ_2 Σ_1^(1/2))^(1/2) Σ_1^(-1/2). The cost of this
optimal map evaluates to the Bures formula above. The full derivation is
beyond Phase 1; we cite Peyré & Cuturi §2.6 and treat the formula as a
black-box oracle.

**Numerical implementation note.**
``scipy.linalg.sqrtm`` uses Schur decomposition and can produce small
imaginary components from real symmetric inputs. We take the real part
when the imaginary component is below 1e-8 and raise otherwise.
