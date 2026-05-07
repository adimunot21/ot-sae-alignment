# Phase 1 — Status and framing

Phase 1 implemented exact-LP OT, log-domain Sinkhorn, Sinkhorn divergence
(cost and dual forms), and the Bures-Wasserstein closed form from scratch
in NumPy/SciPy, with a test suite that cross-validates against POT and
analytical oracles. 90 tests pass; coverage 96%.

**Status:** the package `ot_primitives` is preserved as a didactic artifact.
Production code from Phase 2 onward uses POT directly. The from-scratch
implementations served their teaching purpose — log-domain stabilization,
the bias of entropic OT, the entropy-gap distinction between cost-form and
dual-form Sinkhorn divergence, the Bures formula — and that material will
be expanded into a paper appendix / standalone tutorial as part of the
project writeup phase.

**What survives in active use:** none of `ot_primitives` is imported by
Phase 2+ code. POT replaces it.

**Why kept in repo at all:** (a) the test suite documents subtle
correctness properties most users of POT don't think about; (b) the
material feeds the eventual tutorial chapter.
