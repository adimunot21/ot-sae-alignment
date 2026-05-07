# OT-SAE Alignment Project — A Detailed Walkthrough

A from-scratch explanation of what we built, what we tried, what worked, and what didn't. Written for someone with hardware-engineering or general CS background but no prior optimal transport (OT) or sparse autoencoder (SAE) knowledge.

This is not a paper. It's a story, told one phase at a time, with the actual numbers we measured and an honest reading of what they meant.

## How to read this

You can read in order, or jump to whichever phase you care about. Everything you need to understand any one chapter is either explained inline or pointed back to a previous chapter.

The chapters get more interesting as the project progresses. The early phases are foundational; the late phases are where the actual research happens and where we hit real findings (some of which weren't what we hoped for).

## Chapters

1. [Background — What is Optimal Transport?](01_background_OT.md) — The conceptual core of OT, including Sinkhorn, Sinkhorn divergence, and the key knobs.
2. [Background — What is Gromov-Wasserstein?](02_background_GW.md) — How GW generalizes OT to "incomparable" spaces, and why we thought it would help us.
3. [Background — What is a Sparse Autoencoder?](03_background_SAE.md) — What SAEs are, what they're for, and why matching their features matters.
4. [The original research question](04_research_question.md) — Why we picked this project and what we hoped to show.
5. [Phase 1 — OT primitives from scratch](05_phase_1.md) — Building the OT machinery from the ground up. (Educational; mothballed in production.)
6. [Phase 2 — Gromov-Wasserstein on toy graphs](06_phase_2.md) — Where GW worked, and why that matters as a sanity check.
7. [Phase 3 — Crash course on SAEs](07_phase_3.md) — Loading a real SAE, looking at what its features represent, building intuition.
8. [Phase 4a — Cross-layer matching (the first negative result)](08_phase_4a.md) — Where naive GW failed catastrophically and why.
9. [Phase 4a-bis — Fused-GW (a small positive result)](09_phase_4a_bis.md) — Combining GW with cross-side similarity, and a 5% improvement over Hungarian.
10. [Phase 4b — Unequal sizes (the second negative result)](10_phase_4b.md) — Where the project's main hypothesis met cleanly with reality.
11. [Conclusion — What we actually learned](11_conclusion.md) — The honest summary of what the data tells us, what's salvageable, and what the right next step is.

## Headline numbers (so you know what's coming)

Three numbers from across the project, for orientation:

- Phase 4a (same-layer, different seeds): **fused-GW @ α=0.05 beat cosine-Hungarian by ~5% relative** (mean correlation 0.344 vs 0.329, with non-overlapping uncertainty bands across 3 seeds).
- Phase 4b (same layer, different sizes, 4× ratio): **cosine-Hungarian beat fused-GW** by 1% absolute (mean correlation 0.555 vs 0.549).
- Phase 4b (same layer, different sizes, 8× ratio): same direction — **cosine-Hungarian still wins**.

The story is: GW-flavored methods are *not* the right tool for SAE feature matching when cosine-Hungarian can run at all. That's not the result we hoped for, but it's a real finding, and the rest of this document explains why.
