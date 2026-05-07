# 02 — Background: What is Gromov-Wasserstein?

If you've read [chapter 1](01_background_OT.md), you know that standard OT requires a way to compute "distance from x to y" where x is in the source and y is in the target. That works fine when x and y live in the same space — both are 2D points, or both are 100-dimensional embeddings of the same kind. The cost matrix `C[i, j] = ‖xᵢ - yⱼ‖²` is well-defined.

But what if they don't?

## The motivating example

Suppose I have two 3D point clouds. Standard OT works: distance between a 3D point and another 3D point is just Euclidean distance.

Now suppose I have one 3D point cloud and one 50D point cloud. They might represent the same underlying object (say, the same molecule represented two different ways), but the coordinates aren't comparable. Asking "what's the distance from this 3D point to that 50D point" doesn't even make sense.

Or: I have two graphs. Graph A has 100 nodes; graph B has 80 nodes. Each node has no inherent coordinates — it's just a vertex with some neighbors. There's no "distance from node 47 in A to node 23 in B."

These are **incomparable spaces**. Standard OT doesn't apply.

## The Gromov-Wasserstein idea

Mémoli (2011) found a way around this. The trick: instead of comparing source to target *directly*, compare them by their *internal structure*.

Each space gets described by its own pairwise distance matrix. Space A has matrix `C₁` of size n×n, where `C₁[i, k]` is the distance from point i to point k *within space A*. Space B has matrix `C₂` of size m×m, computed entirely within B.

Crucially, `C₁` and `C₂` don't need to "know about" each other. They can have different sizes, different units, different definitions of distance.

GW asks: find a coupling `T` that *preserves pairwise distances*. For each pair (i, k) in space A, mapped under T to roughly the pair (j, l) in space B, the within-A distance `C₁[i, k]` should be roughly equal to the within-B distance `C₂[j, l]`.

Mathematically:

```
GW(α, β) = min_T  Σᵢⱼₖₗ |C₁[i, k] - C₂[j, l]|² · T[i, j] · T[k, l]
```

The objective sums over every pair-of-pairs (i,k) and (j,l), looking at how close `C₁[i, k]` is to `C₂[j, l]` and weighting by how much T maps i to j *and* k to l simultaneously.

It's a *quadratic* objective in T. That's harder than the linear OT problem. The optimization is non-convex, has multiple local minima, and is generally messier than standard OT. But it works, and POT (the Python library) has a solid entropic-regularized GW solver that's good enough for most applications.

## What GW recovers, in plain words

Imagine two graphs that you know are isomorphic — same structure, but with different node labels. GW takes the within-graph distance matrices (e.g., shortest-path distances) and finds the relabeling that makes them line up. It does this without ever being told what the right correspondence is — it figures it out from structure alone.

This is the canonical use case. We tested it in Phase 2 of our project, and GW recovers the correspondence at 100% accuracy on clean inputs. (Phase 6 of this document covers this in detail.)

GW is also the right tool for:
- Matching shapes (point clouds) up to rotation/scaling/reflection.
- Matching graphs of different sizes.
- Aligning embeddings from different models, when you can compute pairwise distances within each but not across.

## The crucial caveat (this becomes the project's central tension)

GW only sees within-side structure. It receives `C₁` and `C₂` separately. Each must contain enough structural information for GW to lock onto.

What does "enough structural information" mean? Concretely: the pairwise distance matrix needs to have *variation*. If every pair of points within a space is roughly the same distance from every other pair, then `C₁` is featureless and GW has nothing to match.

Think about this: in graph isomorphism, distance variation is huge — some pairs of nodes are 1 step apart, some are 5, some are 10. The triangles, the paths, the densely-connected clusters all show up clearly in the pairwise distance matrix. There's tons of structure for GW to match.

In contrast, what if you have 1000 points in a high-dimensional space where most pairs are at roughly the same distance from each other? That happens often in high dimensions due to the *concentration of distance* phenomenon — random vectors in d-dimensional space tend to be nearly orthogonal to each other, so most pairwise distances cluster around the same value. In that regime, `C₁` looks nearly featureless, and GW struggles.

We didn't appreciate how much this would matter to us. We will, by Phase 4a.

## Entropic-regularized GW

Just like standard OT, GW has an entropic-regularized version:

```
GW_ε = the GW objective + ε · entropy regularizer
```

Same trade-off as before. Small ε → sharp plan, slow, possibly numerically unstable. Large ε → smoothed plan, fast, biased.

There's an extra subtlety here that doesn't exist in standard OT: the entropy regularizer competes with the GW objective for influence over the plan. If the GW objective itself has small magnitude (which happens when within-side distances are nearly uniform), the regularizer dominates and the plan collapses to uniform regardless of how small ε is. We'll see this happen.

## Fused-GW

There's a hybrid form called **Fused Gromov-Wasserstein** (Vayer et al. 2018) that we'll use in Phase 4a-bis. It's a convex combination of standard OT and standard GW:

```
FGW(α, β) = α · <M, T>  +  (1-α) · GW_objective(C₁, C₂, T)
```

where `M[i, j]` is now a *cross-side* cost — the cost of mapping source point i to target point j directly. Note that this requires the two spaces to share enough structure that you can compute a cross-side cost. So FGW is most useful in settings where:
- You *can* compute cross-side cost meaningfully (e.g., decoder vectors of two SAEs both live in the same residual stream space).
- AND there's also relevant within-side structure you'd like to exploit.

The parameter α controls the mix:
- α=1 → pure standard OT (no GW influence).
- α=0 → pure GW (no cross-side influence).
- α∈(0,1) → blend.

This becomes the workhorse of our Phase 4 experiments.

## Summary

Gromov-Wasserstein generalizes optimal transport to settings where source and target live in different spaces. It works by matching pairwise *within-side* distances rather than absolute positions. Its strength is that it doesn't need a shared coordinate system; its weakness is that it needs the within-side distance distributions to be *informative* — non-uniform, with structure to lock onto.

When within-side distances are nearly featureless, GW fails. We will discover this empirically in [Phase 4a](08_phase_4a.md).

Next chapter: SAEs.
