# 04 — The Original Research Question

This is the project's elevator pitch. It's important to lay it out clearly before we get into what we tried, because — spoiler — the data ended up not supporting the elevator pitch, and we should be honest about that gap.

## The setting

You have two trained SAEs, A and B. Each has a feature dictionary — a list of decoder directions, one per feature. You want a *matching*: for each feature in A, find its corresponding feature in B (or determine that no good match exists).

This is a real, useful problem. People want it for the reasons in [chapter 3](03_background_SAE.md): universality studies, cross-model transfer, scaling-law experiments.

## The current standard tool

Cosine-Hungarian matching. Three steps:

1. Build the cosine-similarity matrix between A's decoder directions and B's. Entry (i, j) is cos-sim of A's feature i with B's feature j.
2. Convert to a cost matrix by negating.
3. Run `scipy.optimize.linear_sum_assignment` to find the assignment that minimizes total cost.

This works when both SAEs have the same dictionary size and live in the same coordinate system (i.e., they're both trained on the same base model).

## Where Hungarian fails

Two real settings:

1. **Unequal dictionary sizes.** If A has 4096 features and B has 24576, Hungarian fundamentally wants a square cost matrix. It does *technically* run on rectangular matrices in modern scipy (it picks the best 4096 matches from the 24576 candidates and ignores the rest), but this loses information about the larger SAE's other 20480 features. Researchers studying feature splitting at different scales ideally want to know "this concept in the smaller SAE corresponds to *these three* concepts in the larger one." Hungarian can't say that.

2. **Cross-model SAEs.** If A is trained on GPT-2-small (768-dim residual stream) and B on GPT-2-medium (1024-dim), the decoder directions live in incomparable coordinate systems. Cosine similarity isn't defined across different dimensions. Hungarian can't run at all.

## The original project pitch (as I sold it)

GW is the natural tool for both cases. GW doesn't compare features *directly* across the two SAEs. Instead, it builds a *within-side* pairwise distance matrix for each SAE separately — within A, how far apart are its decoder directions from each other? Same for B, separately.

Then it finds a coupling that *preserves these within-side distances*. Two features that are far apart within A get matched to two features that are far apart within B. Two features that are close within A get matched to two features that are close within B.

This handles unequal sizes natively (GW's quadratic objective is fine with rectangular T matrices). And it handles cross-model cases natively (the within-side distances in A and B are computed entirely separately; they don't need to share a coordinate system).

The pitch I made: **GW is the right tool because it doesn't need a shared coordinate system. It just needs each SAE's internal pairwise structure.**

That sounded compelling. It still sounds compelling on a slide. But the actual question, which I underweighted at proposal time, is:

> Does the within-side pairwise distance structure of an SAE *carry enough information* for GW to actually find the right matching?

The answer turned out to be no. Or at least: not nearly enough information to compete with Hungarian's direct cross-side cosine signal in the cases where Hungarian *can* compute it.

## What I should have asked at proposal time

In retrospect, the question I should have flagged:

What does the pairwise distance distribution between SAE decoder vectors *look like*?

If most pairs of decoder vectors are nearly orthogonal (cosine similarity ≈ 0), then the within-SAE distance matrix is nearly featureless — std of distances is small relative to mean — and GW has nothing to lock onto.

If decoder vectors form clean clusters in 768-dim space (some pairs very close, some very far), then the within-SAE distance matrix has rich structure and GW should be able to match it across SAEs.

I didn't run this diagnostic before committing to the project. Had I done so, I would have seen the concentration-of-distance phenomenon — high-dimensional vectors with no particular geometric organization tend to be near-orthogonal — and recognized that within-SAE structure was unlikely to be informative.

This is a real lesson: when you propose a research project that hinges on a particular geometric property, *measure that property first*, before committing to weeks of work.

## The hypotheses, made explicit

Phase 4 was structured around three sub-experiments, each testing a specific hypothesis:

**Phase 4a.** Two SAEs of the same size on the same model (different random seeds, or different layers). Hungarian *should* work fine here. The hypothesis: GW at least matches Hungarian. If GW doesn't even tie here, the project pivots.

**Phase 4b.** Two SAEs of different sizes on the same model. Hungarian's rectangular extension is degraded. The hypothesis: GW substantially beats Hungarian's rectangular extension by exploiting structural information that Hungarian discards.

**Phase 4c.** Two SAEs on different models. Hungarian can't run. The hypothesis: GW provides a workable matching where Hungarian gives nothing.

If 4a passes, we proceed to 4b. If 4b passes, we proceed to 4c.

## What actually happened (preview)

- **Phase 4a (initial attempt):** Pure GW failed catastrophically. Mean correlation 0.04 vs Hungarian's 0.33. The project's main hypothesis looked dead on arrival.
- **Phase 4a-bis (fused-GW retry):** Combining GW with a small fraction of cross-side similarity (α=0.05) recovered to 0.34 — slightly *better* than Hungarian. Modest positive result.
- **Phase 4b:** Hungarian's rectangular extension turned out to be much stronger than the project plan assumed. Cosine-Hungarian beat fused-GW across all three size ratios (2x, 4x, 8x). The project's main hypothesis didn't survive the test.

The arc of this project is: the original hypothesis was wrong in a particular way, we discovered why over the course of careful experiments, and we ended up with a careful negative result rather than the positive contribution we hoped for.

The remaining chapters walk through each phase in detail. The negative result is itself useful — it closes off an intuition that lots of researchers might have had — and the experimental machinery we built is reusable for future questions.

Next: [Phase 1](05_phase_1.md), the OT primitives we built from scratch.
