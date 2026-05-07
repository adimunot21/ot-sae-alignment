# 06 — Phase 2: Gromov-Wasserstein on Toy Graphs

This is the phase that built our trust in GW. By the end of it, we believed GW was a powerful, well-behaved tool that did what it claimed to do. That belief turned out to be correct in this domain — and misleading when transferred to the SAE setting.

The chapter is short, because the experiment is simple, but the lesson at the end is one of the most important in the whole project.

## What we ran

A single notebook with one core experiment and one robustness study.

**Setup.** Generate a random graph G with 15 nodes using a stochastic block model — three communities of 5 nodes each, with high within-community edge probability (0.7) and low between-community probability (0.1). Compute its shortest-path distance matrix C₁.

Make a copy of G, randomly relabel its nodes (apply a permutation π to the node labels), and compute its shortest-path distance matrix C₂. Mathematically, C₂ is just a permuted version of C₁, so the two graphs are *isomorphic* — same structure, different labels.

Run entropic GW on C₁ and C₂ with uniform marginals. Decode the soft coupling via `argmax`. Compare the recovered matching to the inverse of the relabeling permutation.

## What we saw

100% accuracy. GW recovered the relabeling perfectly. The right-side panel of the visualization (which permutes the recovered coupling by the truth) showed a clean diagonal — exactly what you see when GW gets it right.

The GW loss after convergence was 1.18e-16 — essentially machine-precision zero. The algorithm found that the optimal coupling preserved within-graph distances *exactly*, which makes sense: the two graphs *are* the same graph up to relabeling, so a perfect distance-preserving correspondence exists.

This was the textbook GW experiment, and GW passed it textbook-style.

## Robustness study: edge noise

We added an "edge noise" experiment. Take graph G, randomly toggle some edges to corrupt it, then run GW between the original G and the corrupted version.

We swept the number of edits and ran 5 seeds per noise level. The resulting accuracy curve:

| edits | accuracy mean | std |
|---|---|---|
| 0 | 100% | 0% |
| 2 | 89% | 10% |
| 5 | 45% | 32% |
| 10 | 32% | 5% |
| 20 | 15% | 11% |

A few things worth noting.

**The clean cases stay clean.** Zero edits give 100%; two edits give 89% with 10% std. GW degrades gracefully in this regime — small structural perturbations cause small accuracy drops. This is the desirable property.

**The middle is messy.** At 5 edits, accuracy is 45% with 32% std. This high variance is honest — *which* edges get hit matters a lot. Some edits remove peripheral edges that GW recovers from; some hit critical structural information that GW can't replace. The variance reflects the underlying problem's heterogeneity.

**The high-noise regime collapses.** At 20 edits on a 15-node graph, you've corrupted enough of the structure that GW can't reliably recover anything. Accuracy drops toward 1/15 ≈ 7%, which is chance.

This is a cleanly interpretable curve. GW does what it claims: when the structural information is preserved, it finds the right matching; as structure is destroyed, it degrades smoothly toward chance.

## What this experiment validated

Several things that we relied on later:

1. **Our POT integration works.** The library does what we expect; entropic GW with `argmax` decoding recovers correct matchings on clean inputs.
2. **The pipeline pattern works.** Build distance matrices, run GW, decode via argmax — this is the same pattern we'd use in Phase 4.
3. **GW is a real tool.** It's not magic, it's not buggy, it's not the wrong algorithm. When given informative within-side distances, it does the right thing.

## The lesson I should have extracted, and didn't

Here's the part that became important much later.

Phase 2 succeeded because graph distance matrices are *informative*. The shortest-path distances in a 15-node SBM graph have rich variation: some pairs are 1 step apart (direct neighbors), some are 2-3 (within community via one hop), some are 4-5 (across communities). The histogram of pairwise distances has multiple peaks, clear structure, distinct clusters.

GW locks onto this structural variation. It knows that a pair-of-nodes-at-distance-1 in graph A should be matched to a pair-of-nodes-at-distance-1 in graph B. It uses the variation to pin down the matching.

What I should have asked at the end of Phase 2 — and didn't — is:

> **When we run GW on SAE decoder vectors, will the within-SAE distance matrix have similar informative variation? Or will it be near-uniform?**

Had I run the diagnostic — just compute the cosine-distance distribution of decoder vectors in any SAE — I would have seen that decoder vectors are nearly all orthogonal to each other. The within-SAE distance distribution has mean ≈ 0.99 and std ≈ 0.12. Almost no variation. Almost no structure.

GW on a near-uniform distance matrix has nothing to lock onto. The plan collapses to uniform regardless of how cleverly you tune ε.

This is the *concentration of distance* phenomenon — random vectors in high dimensions are nearly orthogonal to each other. SAE decoder vectors are random-looking enough that the same thing happens. The graphs in Phase 2 had clear non-uniform structure. The SAEs in Phase 4 didn't.

This single diagnostic — "what does the within-side distance distribution look like?" — would have predicted Phase 4a's failure before we ran a single matching experiment. I didn't run it. Lesson: when you're proposing a method that depends on a particular geometric property, *measure that property in your real data first*, before doing anything else.

## Phase 2's status in the project

The notebook is committed. The result is real and reproducible. It validates that we know how to run GW correctly in the easy case.

The deeper lesson about distance distributions is the one that actually mattered for the project's outcome. We discovered it the hard way in Phase 4a — see the next chapter.

Next: [Phase 3](07_phase_3.md), where we got hands-on with real SAEs.
