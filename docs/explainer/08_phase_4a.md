# 08 — Phase 4a: Cross-Layer Matching (The First Negative Result)

This is where the project hit its first real wall. The numbers are clean, the diagnostic story is clear, and the lesson is one of the most important takeaways from the whole project: GW with within-side cosine distance does not work for SAE feature matching.

The chapter is long because the full story matters. We saw the failure, ran diagnostics that explained it, tried a fix, watched the fix fail too, and only then truly understood why.

## Setup

We loaded two pretrained SAEs from the same release (`gpt2-small-res-jb`):
- One trained on **layer 7** of GPT-2-small (residual stream, pre-block).
- One trained on **layer 8** (same hook position, one layer later).

Both have 24,576 features. Both decode into the same 768-dim residual stream. Cosine similarity *across* the two SAEs is meaningful — they're decoder vectors in the same coordinate space.

The matching question: for each feature in the layer-7 SAE, which feature in the layer-8 SAE corresponds to it?

This isn't necessarily a well-defined question for *every* feature. Some concepts at layer 7 might not have a layer-8 counterpart (because layer 8 has constructed something new). But many should — the residual stream is read by every layer, so concepts that exist at layer 7 generally persist into layer 8 and beyond.

## The three methods we compared

**Method 1: Cosine-Hungarian.** L2-normalize the decoder rows. Compute the pairwise cosine similarity matrix between layer-7 decoder rows and layer-8 decoder rows. Run `scipy.optimize.linear_sum_assignment` on the negated similarity matrix.

This is the "standard tool." It uses cross-side information directly: for each (i, j) pair, the cost is "how similar are layer-7 feature i and layer-8 feature j as decoder directions." Hungarian finds the assignment that maximizes total similarity.

**Method 2: Activation-Hungarian.** Run both SAEs on the training portion of our token corpus. For each feature in each SAE, get its 8,000-element activation pattern (one number per training token). Compute the Pearson correlation matrix between layer-7 features' patterns and layer-8 features' patterns. Run Hungarian on the negated correlations.

This uses cross-side information of a different kind: "do features fire on the same tokens?" If they do, they're plausibly the same concept.

**Method 3: Entropic GW (the project's main hypothesis).** Compute pairwise *cosine distance* matrices *within* each SAE separately. The within-layer-7 matrix is 2000×2000 (we restricted to the top-2000 most-active features for tractability). Same for layer 8. Run entropic GW with ε=5e-3 on these two matrices.

This uses *only* within-side information. GW never sees the cross-side cosine similarity. It tries to find a matching that preserves within-side distance structure.

## The evaluation protocol

For each method's matching, on the held-out 30% of tokens (about 4,000 tokens):
- For each (i, π(i)) pair, compute Pearson correlation between feature i in SAE A and feature π(i) in SAE B.
- Report mean, median, quartiles, and the fraction of pairs with correlation > 0.3, > 0.5, > 0.7.

The metric is honest: it asks "after matching, how related are the features in their behavior on data we didn't use to compute the matching?" High correlation means the matching is real; low correlation means the matching is noise.

## The result

Restricted to top-2000 features per SAE. One seed. Single run.

| Method | mean_corr | median_corr | frac > 0.5 |
|---|---|---|---|
| Cosine-Hungarian | 0.329 | 0.260 | 27.5% |
| Activation-Hungarian | 0.314 | 0.234 | 27.4% |
| **Entropic GW** | **0.043** | **0.001** | **1.7%** |

Hungarian methods find real signal: about 28% of feature pairs have held-out correlation above 0.5. That's "this is a real matching" territory. Most of the matched pairs are weakly to moderately correlated (median 0.26), which makes sense — many features at layer 7 don't have direct layer-8 counterparts, so the average correlation is pulled down by the unmatched cases. But a substantial fraction is well-matched.

GW gives essentially zero. Mean correlation 0.043 — not just below the others, but at chance. Median correlation 0.001 — the *typical* matched pair shows zero relationship.

This was unexpected. Phase 2 had GW recovering matchings perfectly on graphs. Why is it failing here?

## The diagnostic, and the moment of clarity

We dug in. Three things to check:

**Is the GW solver converging?** We ran it for more iterations with tighter tolerance. It converged. The "answer" was real — the algorithm wasn't just timing out.

**What does the converged plan look like?** This is where the truth came out. We checked the structure of the GW transport plan T:

- Maximum entry of T: 2.94e-04. (For comparison: the uniform plan would have all entries equal to 1/n² = 2.5e-07. So the plan max is 1000× higher than uniform — but absolutely still tiny.)
- Mean of the top-5 entries per row: 3.15e-06.
- Fraction of total mass concentrated on the argmax of each row: 0.79%.

Translation: the plan is essentially uniform. Every feature in SAE A is matched to every feature in SAE B with roughly equal probability. The argmax decoding extracts a slightly-higher-than-baseline entry, which is essentially noise.

**What does the within-SAE distance distribution look like?** This was the diagnostic I should have run before the project started. We computed the cosine-distance distributions:

- Layer 7 within-SAE distances: mean = 0.991, std = 0.126
- Layer 8 within-SAE distances: mean = 0.992, std = 0.112

Both distributions are concentrated around 0.99 with a tiny spread. That is: almost every pair of decoder vectors within an SAE is *nearly orthogonal* to almost every other pair. The within-SAE distance matrix is nearly featureless.

**This is the explanation.** GW needs informative within-side structure to lock onto. It got featureless distance matrices. The plan collapsed to uniform because there's nothing for the algorithm to differentially favor — every (i, k) pair within SAE A has roughly the same within-A distance as every (j, l) pair within SAE B, so every assignment is roughly equally good (or equally bad). The entropic regularizer, even at small ε=5e-3, dominates the tiny signal in within-side distances and washes everything to uniform.

## A second attempt: co-firing distance

If cosine distance on decoder vectors is too uniform, maybe a different within-SAE distance carries more information.

We tried *co-firing distance*: within each SAE, compute the Pearson correlation between feature i's activation pattern and feature j's activation pattern across the training tokens. Distance = 1 - correlation. Two features that fire on the same tokens have distance 0; two that never co-fire have distance ~1.

This is structurally interesting because co-firing patterns are exactly what activation-Hungarian uses *across* the two SAEs. We're asking: maybe co-firing patterns *within* each SAE are also informative.

The diagnostic on these new distance matrices:

- Layer 7 co-firing distances: mean = 0.999, std = 0.021
- Layer 8 co-firing distances: mean = 0.999, std = 0.022

*Even less* variance than the cosine version. Almost no pair of features in an SAE substantially co-fires with another, because the SAE is sparse — most features are quiet most of the time, so most pairs of features are "both quiet" most of the time, so their correlations are near zero.

We ran GW on these new distance matrices. The plan collapsed harder than before. Maximum entry: 2.6e-07 — exactly uniform, no resolution at all. All matched-pair correlations were near zero or slightly negative.

The diagnosis stuck. SAE within-side distance matrices, under either of the obvious metrics, are too uniform for GW to extract useful information from.

## The deeper issue, made explicit

This is the conceptual lesson worth pulling out. Let me state it clearly.

GW's matching power comes from the *variation* in the within-side distance matrices. If C₁ and C₂ have rich variation — pairs at many different distances — GW can exploit it: a "tightly-clustered pair" in C₁ should match a "tightly-clustered pair" in C₂; a "far-apart pair" in C₁ should match a "far-apart pair" in C₂.

If C₁ and C₂ are nearly featureless — almost all pairs at the same distance — GW has nothing to lock onto. It's like trying to read text written in a single shade of gray.

In the SAE setting:
- Decoder vectors in 768-dim space tend to be near-orthogonal (concentration of distance in high dimensions).
- Co-firing patterns are sparse (most features rarely fire, so most pairs rarely co-fire).
- Both within-SAE distance matrices are featureless.

So GW can't work in this setting. The signal isn't there for GW to extract. This isn't an ε-tuning problem or a numerical issue — even an unregularized exact GW would struggle.

What does work: any method that uses *cross-side* information directly. Cosine-Hungarian uses the fact that decoder vectors live in the same residual stream space, so their cross-side cosine similarity is meaningful. Activation-Hungarian uses the fact that both SAEs see the same tokens, so cross-side activation correlation is meaningful. These methods don't depend on within-side variation; they depend on cross-side comparability.

This insight will determine the next phase's strategy.

## The path forward (as of the end of Phase 4a)

Pure GW is dead for this problem. But there's a natural extension: **fused-GW** (Vayer et al. 2018), which combines a cross-side similarity term with the within-side GW term, weighted by a parameter α.

When α is high, fused-GW behaves like Hungarian (mostly cross-side similarity). When α is low, fused-GW behaves like pure GW (mostly within-side structure). At intermediate α, the two contributions blend.

The question for Phase 4a-bis: does the within-side structure, even though it's mostly featureless, contribute *anything* useful when combined with cross-side similarity? Or is cross-side cosine alone the right answer?

This is the question we'd answer next.

Final note: the negative result of Phase 4a, even if we'd stopped there, is not nothing. "GW with within-SAE cosine distance fails because the within-side structure is too uniform" is a real and useful finding. Future researchers proposing OT-based methods for SAE feature matching can read this and avoid the same dead end.

Next: [Phase 4a-bis](09_phase_4a_bis.md), where fused-GW gave us a small but real positive result.
