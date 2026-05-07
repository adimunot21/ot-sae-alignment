# 11 — Conclusion: What We Actually Learned

This is the honest summary. What worked, what didn't, what's salvageable, and what the right next step is given where the data took us.

## The headline result, in one paragraph

We set out to show that Gromov-Wasserstein (or fused-GW) provides a better tool than the Hungarian algorithm for matching features across two sparse autoencoders, especially in the cases (unequal dictionary sizes, cross-model) where Hungarian fundamentally struggles. The data does not support this. Across same-layer different-seed and same-layer different-size matching problems, cosine-Hungarian dominates — modestly in some regimes, decisively in others. The within-SAE pairwise distance structure that GW-family methods depend on turns out to be nearly uniform under every metric we tried, leaving GW with insufficient signal to compete with direct cross-side similarity. The one positive finding — fused-GW with α=0.05 beating Hungarian by ~5% in the equal-size case — is real but does not generalize to the unequal-size case the project was designed around.

## The numbers across the project

| Setting | Hungarian | Fused-GW | Winner |
|---|---|---|---|
| Phase 4a — cross-layer (24576 ↔ 24576) | 0.329 ± 0.002 | 0.344 ± 0.002 (α=0.05) | Fused-GW (small margin) |
| Phase 4b — same-layer 2× ratio (6144 ↔ 12288) | 0.660 ± 0.001 | 0.659 ± 0.001 | Hungarian |
| Phase 4b — same-layer 4× ratio (6144 ↔ 24576) | 0.555 ± 0.003 | 0.549 ± 0.003 | Hungarian |
| Phase 4b — same-layer 8× ratio (3072 ↔ 24576) | 0.467 ± 0.005 | 0.465 ± 0.006 | Hungarian (small margin) |

The pattern: fused-GW only wins in the equal-size cross-layer case, by a small margin. In every unequal-size case, Hungarian wins. The advantage of fused-GW does not grow with size mismatch — it inverts.

## What we actually learned

A few real findings, distinct from "the original hypothesis didn't pan out":

**1. The within-SAE distance distribution is the predictor.** Both cosine distance between decoder vectors and co-firing distance between feature activations have means near 1.0 and very small standard deviations (0.12 for cosine, 0.02 for co-firing). This concentration is the structural reason GW fails — the algorithm needs variation in within-side distances to lock onto, and this variation isn't there.

This is generalizable advice. Anyone considering a GW-based method for matching across embedding spaces should *first* compute the within-side distance distribution of their data. If it's concentrated (high mean, low std), GW won't work. This single diagnostic, run early, would have saved this project a lot of time.

**2. The cherry-picking effect makes rectangular Hungarian a stronger baseline than expected.** When Hungarian gets to pick its `n_smaller` best assignments from a pool of `n_larger` candidates, the larger pool helps it. This is the opposite of what naive intuition predicts (more candidates = more chance to pick wrong) — in practice, more candidates means Hungarian can avoid the conflicting cases that hurt it in the square setting. Future projects considering "Hungarian fails in the rectangular case" as a starting point should test rectangular Hungarian directly before assuming it's degraded.

**3. The Sinkhorn divergence cost-form vs dual-form distinction is real and matters.** This came out of Phase 1 and isn't directly related to the matching question, but it's worth flagging: papers and libraries are loose about which form of the divergence they use, and the two forms differ by an entropy-gap term that's nonzero for distinct measures. If you're trying to reproduce a Sinkhorn-divergence result and your numbers don't match, this distinction is a likely culprit.

**4. Operational fluency with the SAE machinery.** By the end of Phase 3, we could load any pretrained SAE on top of a transformer model, extract activations, and characterize what features represent. This isn't a "research finding" but it's a transferable skill — the same machinery could be used for a different research question without redoing Phase 3.

## What the negative result is, precisely

It's important to say this carefully. The negative result is *not* "GW doesn't work for SAEs." That's too broad and oversells the conclusion.

The precise claim is: **for matching features across two SAEs trained on the same base model (same-layer or cross-layer, equal-size or unequal-size), GW-family methods using within-SAE pairwise distance structure (cosine or co-firing) do not outperform Hungarian-family methods using cross-side similarity. Pure GW fails because within-SAE distances are too uniform; fused-GW only beats Hungarian in the equal-size cross-layer case by ~5%, and loses to Hungarian everywhere else.**

What we *haven't* tested:
- Cross-model matching (different base models). Hungarian can't run there. Whether GW would work is unknown from our data.
- Different metrics for within-SAE structure (e.g., random-feature kernels, attention-based distances). Possible, untested.
- Hierarchical refinement methods that combine GW with iterative local search. Possible, untested.

So the negative result is sharp and valid in scope, and there are open directions a future project could explore.

## Was Phase 4c (cross-model matching) worth trying?

I considered this carefully and concluded: no.

The reasons:

**1. The use case is weak.** The interpretability community has largely moved to *training-time* approaches for cross-model feature comparison (joint cross-coders, like Anthropic's recent work), rather than *post-hoc* matching. The practitioners who want cross-model feature alignment have a tool, and post-hoc matching isn't it.

**2. The ground-truth problem is severe.** In Phase 4a/4b, we evaluated matching quality via held-out activation correlation — a proxy that requires the two SAEs to operate on the same residual stream space. In the cross-model case, that shared evaluation surface doesn't exist. We'd be evaluating fused-GW against... what? The only viable evaluation would be activation correlation on shared *tokens*, which is itself an activation-Hungarian-flavored signal — and activation-Hungarian, applied directly, is itself a strong baseline.

So the experimental design becomes circular: we'd be testing GW's ability to recover something that activation-Hungarian *already produces*, and previous experience suggests Hungarian wins.

**3. The structural concentration issue almost certainly persists.** Decoder vectors in 768-dim space are nearly orthogonal, and decoder vectors in 1024-dim space are nearly orthogonal. Both within-side distance matrices would still be nearly uniform. The mechanism that defeated GW in Phases 4a and 4b would still be active.

The honest read: pursuing Phase 4c is more likely to be sunk-cost reasoning than productive research. The project has tested its hypothesis carefully and the hypothesis is not supported. The right move is to acknowledge that and decide what comes next.

## The right next step

Three options, in order of how I'd weight them:

**A. Write up the work as a focused tutorial document.** Not a paper. The repo, this explainer, the code, and the data form a coherent educational package: "Here's what optimal transport is, here's what we tried for SAE matching, here's what we found, here's what to take away." This serves several purposes: (1) it preserves the work for future reference; (2) it gives others a starting point that avoids the dead end we hit; (3) it makes the educational content accessible to people who want to learn OT or mech interp without committing to a research project.

This is the option that matches the user's stated request — write a detailed multi-file explainer, not a paper. That's what I'm doing now.

**B. Pivot the project.** The same OT/POT/sae_lens infrastructure could power a different research question. Some candidates:
- OT-based dataset distance metrics for transfer learning.
- Wasserstein-distance-driven curriculum learning.
- OT for matching latent representations in non-SAE settings.

The cost: we've spent a couple of months on the SAE direction; pivoting means restarting research scoping. The benefit: we know POT well, we know SAEs well, and the infrastructure is reusable.

**C. Write a careful negative-result paper for OTML 2026.** "Limits of structural matching for SAE feature alignment." A real workshop paper with the data we have. Modest, honest contribution. The bar for OTML is not paradigm-shifting results; careful methodological work is welcome.

The honest weighting:
- A is the right move if the goal is "get value from the work that's been done." This explainer is part of that.
- B is the right move if the goal is "use this experience to do real research that produces a strong paper."
- C is the right move if the goal is specifically "publish an OTML 2026 workshop paper with the work we have."

The choices aren't mutually exclusive — A is happening now (this document), and B or C could follow.

## Lessons for future research projects

A few things I'd carry to the next project:

**1. Diagnose before building.** When a project hinges on a particular property of your data (in our case, "the within-side distance matrix has informative variation"), compute that property *first*, before committing weeks of work to building the machinery. A 30-minute diagnostic in week 1 would have predicted Phase 4a's failure.

**2. The "stronger baseline than expected" pattern is common.** I underestimated rectangular Hungarian, and that decision invalidated the project's main hypothesis. Future projects should test the strongest possible version of the baseline before committing to "the baseline is broken in setting X."

**3. Negative results are real but require discipline.** The temptation to keep trying variants ("maybe with a different ε, maybe with a different metric, maybe in a different setting") is sunk-cost reasoning in disguise. After two clean negative results from the same diagnostic explanation, the prior on "next variant works" is low. Better to write up the negative result honestly and move on.

**4. Build the infrastructure before the experiments, but evaluate honestly afterwards.** Phase 1 was overkill; Phase 4 was where the real value was. The OT-from-scratch implementation taught me OT well but didn't contribute to the matching results. POT does what we need. In hindsight, less Phase 1, more careful Phase 0 diagnostic on real SAE data.

**5. The from-scratch-vs-library decision is more nuanced than "always one or the other."** For algorithmic primitives that are well-understood and well-libraried, use libraries. For domain-specific parts of the contribution, write yourself. The dividing line: if writing it teaches you something the library would hide, write it. If writing it just reimplements what's known and tested, use the library.

## Closing

This project did not deliver the contribution we hoped for. The data was honest, and the data said cosine-Hungarian is the right tool for SAE feature matching when both SAEs share a base model. We don't get to argue with that.

What we do have: a careful, well-tested codebase; reusable evaluation infrastructure; a clear negative result with diagnosed cause; transferable understanding of OT and mech interp; and this document, which makes the journey accessible to others.

These are real outcomes. They're not the outcomes of the original pitch, but they're the outcomes of the actual experiments, and they're honest. That has to be enough.
