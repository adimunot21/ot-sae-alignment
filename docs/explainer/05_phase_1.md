# 05 — Phase 1: OT Primitives From Scratch

In retrospect, this phase was the wrong choice for the project's goals — but it was educationally valuable, and the code is preserved as a learning artifact. Let me explain what we did, why we did it, and why we eventually shelved it in favor of using the standard library.

## What we built

A Python package called `ot_primitives` containing from-scratch implementations of:

- **Exact OT (the Kantorovich LP).** Wraps `scipy.optimize.linprog` to solve the linear program. Slow but exact.
- **1D Wasserstein closed form.** The sort-and-match formula. Fast and exact for 1D problems. Acts as our "oracle" for testing.
- **Squared Euclidean cost matrix.** A small but important utility that uses a vectorization trick (`||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩`) to avoid materializing a 3D tensor.
- **Log-domain Sinkhorn.** Entropic OT, implemented with `logsumexp` to avoid float64 underflow at small ε.
- **Multiplicative Sinkhorn.** The "naive" version. Implemented *deliberately* so we could see it break at small ε. Lives in `_legacy.py`.
- **Sinkhorn divergence.** Both the "cost form" (subtracting transport costs) and the "dual form" (subtracting regularized objectives via dual potentials), with documented relationship — they differ by an entropy-gap term.
- **Bures-Wasserstein.** The closed-form formula for two Gaussians. Acts as another oracle.

90 tests pass, 96% coverage. The test suite uses a "three independent oracles" discipline: every primitive gets tested against (a) hand-computed answers on tiny problems, (b) the 1D closed form when applicable, and (c) POT (the standard Python OT library) on random problems. Three independent ways of getting the answer; if they all agree, the implementation is almost certainly right.

## What this taught us, in concrete terms

**The bias-variance trade-off in entropic OT, made visible.** We ran a sweep of Sinkhorn cost vs ε, on a fixed pair of 2D Gaussian distributions, with the LP-exact cost as a horizontal reference line.

What we saw was a clean S-curve. At small ε (1e-3 to 1e-2), Sinkhorn cost matched the LP cost to 4 decimal places — we'd recovered the exact answer with a much faster algorithm. At ε=1, Sinkhorn cost was about 30% higher than the LP cost. At ε=100, Sinkhorn cost approached the cost of the *uniform* plan (every source point sends equal mass to every target point) — no transport optimization at all.

The story is: small ε is close to true OT, large ε is biased. The intuition you read in papers becomes a measurement you can plot.

**Why log-domain matters, made visible.** We took the multiplicative Sinkhorn, ran it on a problem with squared-Euclidean costs of magnitude ~30² = 900, at ε=1.0. The algorithm produced NaN. Same problem with log-domain Sinkhorn: clean answer, returns in milliseconds.

The reason: `exp(-900/1.0) = exp(-900)` underflows to zero in float64. Multiplicative Sinkhorn divides by `K @ v` where K has zero entries; you get zero divided by zero, which is NaN. Log-domain Sinkhorn never computes `exp(-C/ε)` directly — it stays in log space throughout — so underflow doesn't happen.

This was viscerally satisfying. It's not just "the textbook says use log-domain." It's: here's the input that makes the naive version fail, and here's the implementation that doesn't.

**The cost vs dual form distinction in Sinkhorn divergence.** We implemented both forms, ran them both on the same problems, and watched them disagree by the predicted "entropy gap" amount. The dual form gave exactly zero on self-self (`S_ε(α, α) = 0` to machine precision); the cost form gave approximately zero (depending on Sinkhorn's convergence tolerance).

This distinction matters because papers and libraries are loose about which form they mean. POT computes one form; Feydy et al. 2019 define another; the difference is `ε` times an entropy gap that's nonzero for distinct measures. If you're trying to reproduce a result, knowing which form is in use can be the difference between matching and not matching.

**The Bures benchmark.** We implemented the closed-form Wasserstein-2 distance between two Gaussians and used it as an oracle. Sample n points from each Gaussian, compute the LP cost on the samples — as n grows, the sample-based cost converges to the Bures formula. We saw this convergence happen, with the predicted 1/√n-ish rate. A clean, end-to-end sanity check that the whole stack works.

## The educational discipline

We followed three rules throughout Phase 1:

**Rule 1: Tests on every primitive, before using it for anything.** A bug in an OT solver looks normal — you just get slightly off numbers — and propagates everywhere. The test discipline says: write the closed-form sanity check, write the cross-validation against POT, *then* use the primitive in experiments. Not the other way around.

**Rule 2: Three independent oracles.** Hand-computed answers on tiny problems, closed-form formulas where they exist, and POT. If two of three agree, almost certainly correct. If they disagree, find out why before continuing.

**Rule 3: Implement the math, not the optimization.** When the algorithm *is* the OT idea (Sinkhorn iterations, the divergence symmetrization, the Bures formula), write it from scratch. When the algorithm is a standard linear-algebra tool (LP solver, matrix square root, eigendecomposition), call the library. The division kept the code focused — Phase 1 is about understanding OT, not about reinventing optimization.

## Why we shelved it

Halfway through what was supposed to be Phase 4, the actual research, we hit a wall: a benchmark script crashed with an out-of-memory error, and I noticed I was about to rebuild a Bures convergence figure that POT could already produce. Stopping to inspect, the broader pattern became clear:

- We were spending engineering time on debugging numerical stability issues that POT had already solved.
- The from-scratch implementations were strictly slower than POT's (POT uses C extensions; we're pure NumPy). Speed didn't matter for Phase 1, but it would matter for Phase 4 with thousands of features.
- The actual research contribution was *not* about implementing OT well. It was about applying OT to SAE matching. POT had everything the actual research needed.

I made the call to mothball `ot_primitives` and use POT for everything from Phase 2 onward. The decision was: we keep the code as a teaching artifact and as supporting material for the eventual writeup, but don't import any of it from production code.

## What survives

The lessons survive, even though the code doesn't get used:

- A precise understanding of how `ε` controls the bias-variance trade-off.
- Visceral experience with why log-domain matters.
- The cost-vs-dual-form distinction in Sinkhorn divergence — useful when reading papers.
- The Bures formula as an oracle — useful intuition for sanity-checking sample-based estimators.
- The "three independent oracles" testing discipline, which we apply to all later code.

The 90 tests still run, still pass, and the code is publicly visible on the repo. If we ever write the project up as a tutorial paper or workshop appendix, this material is the natural place to start.

## What we'd do differently

If I were starting over: skip Phase 1 entirely. Use POT from day one. Compensate by reading Peyré-Cuturi (the OT textbook) more carefully — get the conceptual fluency from reading and from running POT examples, rather than from re-implementing.

The cost of the wrong call: about 7 work-days. Not catastrophic, but real. The benefit of having made it: I now understand the algorithms one level deeper than I would have, which helps when reading papers. Whether this trade-off was worth it depends on long-term goals; for *this* project, it wasn't.

Next: [Phase 2](06_phase_2.md), where we ran GW on a toy graph problem and saw it work cleanly.
