# 09 — Phase 4a-bis: Fused-GW (A Small Positive Result)

After Phase 4a's negative result, we had a clear diagnosis: pure GW fails because within-SAE structure is too uniform. The fix to try was fused-GW, which combines GW's structural term with a cross-side similarity term.

This chapter is about the fused-GW experiment, the surprise it produced (a small but real improvement over Hungarian), and what it does and doesn't mean.

## What fused-GW does, restated

Fused-GW objective:

```
FGW = α · <M, T>  +  (1-α) · GW_objective(C₁, C₂, T)
```

The first term is standard OT: M is a *cross-side* cost matrix (e.g., M[i,j] = 1 - cos_sim of A's feature i with B's feature j). The second term is GW: it uses within-side distance matrices C₁ and C₂.

The mixing parameter α controls the blend:
- α=1.0 → pure standard OT. With M = 1 - cos_sim and the entropic relaxation, this is essentially "Sinkhorn-Hungarian on cosine distance" — equivalent to Hungarian's logic, just with a soft entropic plan instead of a hard assignment.
- α=0.0 → pure GW. The failure case from Phase 4a.
- α∈(0,1) → blend.

The hope: at some intermediate α, the within-side structural term (even though it's mostly featureless) provides a useful tiebreaker that improves on pure cross-side matching.

## The first sweep

We ran fused-GW at α ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0}. Same setup as Phase 4a — top 2000 features per layer, ε=5e-3, max 500 iterations.

The first result was striking enough that I want to lay it out before doing the multi-seed verification:

| alpha | mean_corr | frac > 0.5 | agreement with cosine-Hungarian |
|---|---|---|---|
| 0.1 | 0.342 | 28.3% | 71.9% |
| 0.3 | 0.332 | 27.8% | 85.4% |
| 0.5 | 0.331 | 27.5% | 90.7% |
| 0.7 | 0.331 | 27.6% | 92.8% |
| 0.9 | 0.331 | 27.6% | 94.1% |
| 1.0 | 0.330 | 27.6% | 94.4% |

Two observations:

**At α=0.1, fused-GW slightly *beats* cosine-Hungarian.** Mean correlation 0.342 vs Hungarian's 0.329. That's a 4% relative improvement. Small in absolute terms, but it's *higher*, and it's at the value where fused-GW is mostly GW with just a sliver of cross-side similarity providing scaffolding.

**The agreement-with-Hungarian column tells a story.** At α=0.1, fused-GW disagrees with Hungarian on about 28% of feature pairs. So it's not just reproducing Hungarian. It's finding genuinely different matches, and those different matches are slightly better on the held-out metric.

**As α increases toward 1, fused-GW converges to Hungarian.** Mean correlation converges to ≈0.33 (essentially Hungarian's value), and agreement converges to 94% (essentially the same matching).

## The interpretation, made concrete

The Phase 4a story was: GW can't work alone because within-SAE structure is too uniform.

The Phase 4a-bis story refines this: GW can't work *alone*, but a small dose of GW combined with a strong dose of cross-side similarity outperforms pure cross-side similarity. The within-side structure, while not self-sufficient, contributes useful signal at the margin.

What's the within-side signal doing? Best guess: when cosine-Hungarian can't decide between two feature matches that look equally good cross-side (similar cross-side cosine similarity), the structural term tilts the decision toward the option whose within-side neighborhoods are more compatible. It's a tiebreaker.

For the ~28% of pairs where Hungarian and fused-GW disagree, the GW component is making the call. And on average, it's making slightly better calls.

## Multi-seed verification

A single-seed result with a 4% improvement is suggestive but not decisive. We needed to verify this held across multiple data samples.

We ran a multi-seed sweep: 3 seeds (different initial document samples from the streaming dataset) × a finer α grid ({0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0}). About 30 minutes of compute.

The aggregated results across 3 seeds:

| method | mean_corr | std | frac > 0.5 |
|---|---|---|---|
| cosine_hungarian | 0.3294 | 0.0019 | 0.2752 |
| activation_hungarian | 0.3112 | 0.0033 | 0.2723 |
| pure_gw (α=0) | 0.0431 | 0.0005 | 0.0173 |
| **fused_gw α=0.05** | **0.3440** | **0.0022** | **0.2843** |
| fused_gw α=0.1 | 0.3397 | 0.0021 | 0.2815 |
| fused_gw α=0.15 | 0.3363 | 0.0024 | 0.2800 |
| fused_gw α=0.2 | 0.3339 | 0.0015 | 0.2782 |
| fused_gw α=0.5 | 0.3311 | 0.0016 | 0.2758 |
| fused_gw α=1.0 | 0.3309 | 0.0011 | 0.2762 |

Three things worth pulling out:

**The optimum is at α=0.05, not α=0.1.** Even more cross-side, even less GW, but still some GW influence. The improvement over Hungarian: 0.0146 in mean correlation, with the std of each estimate at ~0.002. The gap is about 7× the typical std. By any reasonable test, the difference is real.

**The α curve is clean.** Pure GW is at chance, then performance jumps to slightly above Hungarian at α=0.02, peaks at α=0.05, then declines monotonically toward Hungarian-level at α=1.0. This is exactly the shape "GW provides marginal scaffolding signal" predicts.

**The std bars are tight.** Standard deviations across 3 seeds are around 0.002. The gap between α=0.05 fused-GW (0.344) and α=1.0 (0.331) is about 0.013. The gap is real and reproducible across seeds.

## What this means, sharply

The Phase 4a-bis result is a **modest but genuine positive finding**. Specifically:

- Pure GW for SAE matching: dead. Confirmed.
- Fused-GW with mostly cross-side similarity and a small amount of within-side structure: ~4-5% relative improvement over pure cosine-Hungarian.
- The improvement is small but reproducible, with non-overlapping uncertainty bands.

This is the level of finding that would warrant a workshop paper *if* it generalized to other settings. The natural next question: does this same fused-GW recipe (small α, lots of cross-side, little GW) help in the unequal-size case where Hungarian is degraded?

If yes: we have a coherent story. Fused-GW is the right tool whenever Hungarian can run, modestly better in the easy case, substantially better in the hard case. That'd be a real contribution.

If no: the Phase 4a-bis result is a small artifact of the equal-size setting, and the project's main hypothesis (GW helps when sizes are unequal) is unconfirmed.

This question is what Phase 4b answered.

## A side note on baselines

One thing worth flagging here. Across all of Phase 4a, **activation-Hungarian** (using cross-side activation correlation, not cross-side cosine similarity) consistently performed *worse* than cosine-Hungarian. Mean correlation 0.311 vs 0.329. This was a small surprise.

The intuition I had going in was: activation-Hungarian uses *behavioral* information (do features fire on the same tokens?), which is more directly related to "are these the same concept." Cosine-Hungarian uses *geometric* information (do decoder vectors point in similar directions in residual stream?).

Cosine winning over activation makes sense in this specific cross-layer setting:

- Decoder vectors at layers 7 and 8 still live in the same residual stream space, so their cosine comparison is informative.
- Activation patterns at layer 7 vs layer 8 *differ* because the model does computation between them — features at layer 8 fire on slightly different tokens than their layer-7 counterparts because layer 7→8 transformation has happened in between. So activation correlation systematically underestimates concept similarity for cross-layer cases.

This wouldn't necessarily generalize to other settings (like cross-seed matching at the same layer), and the activation-vs-cosine comparison is itself a research question. We didn't pursue it; we noted it and moved on.

## Where we stood at the end of Phase 4a-bis

A modest positive result on the equal-size case, with a clear hypothesis to test next: does fused-GW preserve or extend its advantage in the unequal-size case?

The original project pitch staked everything on this. If unequal-size was where Hungarian fundamentally fails and GW fundamentally helps, fused-GW should *crush* it.

We were about to find out the answer wasn't what we hoped.

Next: [Phase 4b](10_phase_4b.md), where the unequal-size experiment delivered the project's clearest result.
