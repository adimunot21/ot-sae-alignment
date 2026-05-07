# 01 — Background: What is Optimal Transport?

Skip this if you already know OT well. Read it if you've heard "Wasserstein distance" thrown around without knowing where it comes from.

## The motivating image

You have a hundred piles of sand on a beach, scattered in different spots. You want to move all the sand into a hundred holes located elsewhere on the same beach. Each pile has a known amount of sand in it; each hole has a fixed capacity. Moving sand costs effort proportional to the distance traveled.

What's the cheapest plan to move all the sand into all the holes?

That's optimal transport. The piles and the holes are *measures* — distributions of mass over locations. The cheapest plan to convert one into the other is the *transport plan*. The total cost of that plan is the **Wasserstein distance** between the source and target distributions.

## Why this generalizes to ML

Two probability distributions are exactly two piles of sand. Just abstract mass instead of literal sand, and abstract "distance between locations" instead of physical distance.

So OT lets you measure how *different* two distributions are by asking how much rearranging it would take to turn one into the other. This is fundamentally different from comparing them point by point (which is what KL divergence does). Point-by-point comparisons fail when distributions don't overlap — KL says "infinity, useless" if two Gaussians sit far apart on the number line. OT says "they're far apart, here's how far," which is what you actually want.

## The two formulations

There are two ways to set up the problem mathematically. Both come from old French mathematicians.

### Monge (1781) — deterministic, sometimes impossible

Each pile of sand goes entirely to exactly one hole. You're looking for a function `T(x)` that says "the sand at location x goes to location T(x)." This is clean and intuitive.

It's also sometimes impossible. If you have one big pile and two small holes, Monge can't split the pile. There's no valid `T`.

### Kantorovich (1942) — splits allowed

Mass can be split. Instead of a function, look for a *coupling* `π(x, y)` that says "this much mass flows from x to y." Mathematically, a coupling is a joint probability distribution whose marginals are the source and target.

This formulation always has a solution. And — crucially — it's a **linear program**. The objective (total cost) is linear in the coupling, the constraints (mass conservation, non-negativity) are linear, and linear programs are well-understood. Any standard solver handles them.

For finite measures with `n` source points and `m` target points, the LP is:

```
minimize    Σᵢⱼ Pᵢⱼ Cᵢⱼ                  (total cost)
subject to  Σⱼ Pᵢⱼ = aᵢ for each i        (rows sum to source mass)
            Σᵢ Pᵢⱼ = bⱼ for each j        (columns sum to target mass)
            Pᵢⱼ ≥ 0                       (no negative sand)
```

`P` is the unknown coupling matrix. `C[i, j]` is the cost of moving one unit from source point `i` to target point `j` — typically squared Euclidean distance.

## The 1D special case

In one dimension, OT has a stunning closed form: **sort both sets of points and match them in order.** No optimization needed.

Concretely: take source samples `[x_1, ..., x_n]` and target samples `[y_1, ..., y_n]`, all with equal weight 1/n. Sort both. The optimal plan matches `x_(i)` (the i-th smallest source) with `y_(i)` (the i-th smallest target). The Wasserstein-p distance is just:

```
W_p^p = (1/n) Σᵢ |x_(i) - y_(i)|^p
```

The reason this is optimal: any plan that "crosses" — that sends a smaller source point to a larger target while a larger source goes to a smaller target — can be uncrossed at lower cost. So no optimal plan crosses. Mathematically this is called *cyclical monotonicity*.

This 1D formula is our **oracle**. Any OT algorithm we run on a 1D problem must agree with this formula. If it doesn't, the algorithm is wrong.

## Why we need approximations: Sinkhorn

The Kantorovich LP is exact and beautiful, but it's slow. The cost scales as roughly `O((nm)³)` in the worst case. Fine at n=m=20. Painful at n=m=200. Intractable at n=m=2000.

Real ML problems involve thousands of points. We need a faster method.

Cuturi (2013) proposed adding a small entropy term to the objective:

```
minimize  Σᵢⱼ Pᵢⱼ Cᵢⱼ - ε · H(P)
```

where `H(P) = -Σ Pᵢⱼ log Pᵢⱼ` is the entropy of the plan. This is **entropic OT**. Adding entropy makes the problem strongly convex, which is good news for optimization. And the optimal plan turns out to have a beautiful structure: `P*ᵢⱼ = uᵢ · exp(-Cᵢⱼ/ε) · vⱼ`. You find the scaling vectors `u, v` by alternating multiplications.

This is **Sinkhorn's algorithm**. It converges in milliseconds even at n=m=10,000.

The price you pay is bias. The plan you get is a smoothed-out version of the true Kantorovich plan. As `ε → 0`, the smoothed plan approaches the exact LP solution. As `ε → ∞`, the plan becomes uniform — no transport optimization at all.

So `ε` is a knob:
- **Small ε** → close to the true LP, sharper plan, slower convergence, numerically dangerous.
- **Large ε** → smoothed plan, faster convergence, biased away from the truth.

The numerical danger at small ε is real. `exp(-C/ε)` underflows to zero in float64 once `C/ε` exceeds about 700. That's why everyone implements Sinkhorn in *log-domain*, where the underflow doesn't happen. But that's an implementation detail; the math is the same.

## A subtle problem with naive Sinkhorn: the bias

Here's a thing that should bother you. The entropic OT cost from a measure to itself is *not zero*. That is, `OT_ε(α, α) > 0`.

Think about it: if the source and target are the same distribution, the *true* optimal plan is the identity (don't move anything). But the entropy regularizer prefers a spread-out plan, so the optimal regularized plan is some smeared-out near-identity. Its transport cost is positive — and that positive number measures how much the regularizer's preference for spreading-out penalizes us, not how different α is from itself.

So `OT_ε` is not a clean distance. Using it as a loss function in ML is fine, but using it as "the distance between two distributions" is misleading.

The fix is the **Sinkhorn divergence** (Genevay, Peyré, Cuturi 2018):

```
S_ε(α, β) = OT_ε(α, β) - ½ · OT_ε(α, α) - ½ · OT_ε(β, β)
```

You compute three Sinkhorn problems instead of one. The two self-self terms measure the bias — how much each distribution "spreads against itself" under the regularizer. Subtracting half of each cancels the bias exactly. The result has three properties:

1. **Self-zero.** `S_ε(α, α) = 0` exactly.
2. **Non-negative.** `S_ε(α, β) ≥ 0`.
3. **Interpolation.** As `ε → 0`, `S_ε → W₂²` (Wasserstein). As `ε → ∞`, `S_ε → MMD²` (Maximum Mean Discrepancy, a popular kernel-based distance).

The third property is genuinely beautiful. Wasserstein and MMD aren't competing tools — they're the same tool at different settings, with `ε` as the dial.

## Bures-Wasserstein for Gaussians

One last thing worth knowing about. For two multivariate Gaussians `N(μ₁, Σ₁)` and `N(μ₂, Σ₂)`, the squared 2-Wasserstein distance has a closed form:

```
W_2² = ‖μ₁ - μ₂‖² + tr(Σ₁ + Σ₂ - 2·(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})
```

The covariance term involves matrix square roots — there's a numerical implementation issue but the math is clean. This is called the **Bures metric** on positive semi-definite matrices, and the same object appears in quantum information theory as the fidelity between density operators.

For our purposes: the Bures formula is *another oracle*. Sample from two known Gaussians, run any OT solver on the samples, and the result must converge to the Bures value as the sample count grows. We'll use this in Phase 1.

## Summary in one paragraph

Optimal transport asks: how much would it cost to rearrange one distribution into another? The Kantorovich LP answers it exactly but slowly. Sinkhorn answers approximately but fast, biased by an entropy regularizer with knob ε. The Sinkhorn divergence cancels the bias and beautifully interpolates between Wasserstein (ε→0) and MMD (ε→∞). For Gaussians, you can compute the answer in closed form, which gives us a way to test sample-based solvers.

Next chapter: Gromov-Wasserstein. The variant that doesn't need both distributions to live in the same coordinate system.
