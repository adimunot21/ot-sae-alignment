# OT-SAE Alignment

Optimal Transport methods for matching features across Sparse Autoencoders. A research project that ran from Phase 0 through Phase 4b, ending with a careful negative result on the original GW-for-SAE thesis.

**Status: complete (paused).** The original hypothesis — that Gromov-Wasserstein methods would outperform Hungarian-style baselines for SAE feature matching — was tested across equal-size and unequal-size cases and did not survive contact with the data. See `docs/explainer/` for the full story and `docs/POSTMORTEM_AND_NEXT_STEPS.md` for the post-mortem.

## What's here

- `ot_primitives/` — From-scratch OT library built in Phase 1 (didactic; mothballed in production code). Includes exact LP, log-domain Sinkhorn, Sinkhorn divergence (cost & dual forms), Bures-Wasserstein closed form. 90 tests pass.
- `ot_sae/` — Production matching code (built on POT). Cosine-Hungarian, activation-Hungarian, GW, and fused-GW.
- `notebooks/` — Phase 2 (toy graph GW), Phase 3 (SAE basics), Phase 4a (cross-layer matching).
- `scripts/` — Multi-seed alpha sweep (Phase 4a-bis) and unequal-size sweep (Phase 4b).
- `results/` — JSON outputs from all experiments.
- `tests/` — Full test suite (98 tests, ~96% coverage on `ot_primitives`).
- `docs/` — Project plan, post-mortem, status-and-handoff, explainer walkthrough.

## Headline findings

- **Phase 4a (cross-layer, equal-size):** Pure GW failed catastrophically (mean correlation 0.04 vs Hungarian's 0.33). Fused-GW at α=0.05 beat cosine-Hungarian by ~5% (0.344 vs 0.329, multi-seed verified).
- **Phase 4b (same-layer, unequal sizes):** Cosine-Hungarian (rectangular) won across all three size ratios (2x, 4x, 8x). Cherry-picking effect made Hungarian a stronger baseline than expected.
- **Why GW fails:** Within-SAE pairwise distance distributions are concentrated (cosine: mean 0.99, std 0.12; co-firing: mean 0.999, std 0.02). Too uniform for GW to extract useful matching signal.

## Reading order

If you're new to the project, read in this order:

1. `docs/explainer/00_README.md` — the navigation index.
2. `docs/explainer/01_background_OT.md` through `11_conclusion.md` — full walkthrough.
3. `docs/POSTMORTEM_AND_NEXT_STEPS.md` — short post-mortem.
4. The notebooks if you want to see the experiments live.

## Quickstart

```bash
conda env create -f environment.yml
conda activate ot-sae-alignment
pip install -e .
make test    # should print "98 passed"
```

## Note on commit history

The granular commit history of Phases 1 through 4b was lost on May 7 due to an accidental `git init` during a copy-paste error. All working files survived. The single recovery commit captures the end-of-Phase-4b state. The narrative of how the project got here is preserved in `docs/explainer/`, which is the canonical story of the work.

## License

Pending.
