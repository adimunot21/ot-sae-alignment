# Phase 1 — OT Primitives from Scratch

**Project (overall):** Gromov–Wasserstein matching of sparse-autoencoder feature dictionaries across models. Target venue: OTML @ NeurIPS 2026.

**Phase 1 scope (this document):** build a tested, well-documented OT primitives library that becomes the foundation for every later phase. Two weeks at ~40 hrs/week.

This is not a research phase. It produces a library and a foundational understanding, not a result. Treating it as research would set the wrong success bar; treating the library as throwaway code would set the wrong quality bar. The frame is: code I would be willing to publish on its own as `ot-from-scratch` if the larger research project pivoted away.

---

## 1. Why this phase exists

Every later phase reads from this one. If `sinkhorn` has a silent numerical bug, every Phase 5+ matching number is wrong. If `sinkhorn_divergence` doesn't actually debias the way I think it does, my Phase 6 unequal-dictionary results are uninterpretable. The OT-research community catches these bugs in review, brutally. So Phase 1 is not preparation, it's the bedrock the rest of the project sits on.

The pedagogical purpose runs alongside the engineering purpose: I haven't worked seriously with OT since my undergrad thesis, and the field has moved. Implementing Sinkhorn in the log-domain, deriving the Bures formula by hand, and seeing closed-form Wasserstein-2 between Gaussians match my sample-based estimate to four decimal places — that's how I rebuild fluency. The deliverables happen to be the same artifacts a research-quality library needs.

What I'm explicitly *not* doing in Phase 1: writing my own LP solver, my own eigenvalue routine, my own matrix-square-root. Those exist in `scipy` and `numpy`, are battle-tested, and learning them is a different project. The dividing line: if the algorithm is *the OT idea*, write it from scratch (Sinkhorn iterations, the divergence symmetrization, the Bures formula, the 1D sort trick). If the algorithm is a generic optimization/linear-algebra tool that the OT idea uses as a black box, call the library.

## 2. Goal and success criteria

**Goal:** a Python package `ot_primitives` that exposes correct, tested implementations of the foundational OT objects, plus one notebook that demonstrates them all converging to the right answer on cases I can verify by hand or formula.

**Success — concrete and gateable:**

1. **Closed-form agreement.** Implementations agree with closed-form answers to `≥4` decimal places on at least three known-answer cases: 1D Wasserstein-2 via sorting, 2D Wasserstein-2 between Gaussians via Bures, Wasserstein-1 between two Diracs.
2. **Reference agreement.** On 50 randomly-generated 2D problems (n in {32, 64, 128, 256}), our exact-OT cost matches POT's `ot.emd2` to relative tolerance 1e-6, and our log-domain Sinkhorn matches POT's `ot.sinkhorn2` to relative tolerance 1e-4 across ε ∈ {0.01, 0.1, 1.0}.
3. **Sinkhorn-divergence properties.** `S_ε(α, α) ≤ 1e-8` on a battery of test measures (this should be exactly zero in theory, modulo numerical noise — if it's not near zero, the symmetrization is wrong).
4. **Numerical stability.** Log-domain Sinkhorn does not overflow, NaN, or silently fail at ε = 0.001 on n=256 problems where the multiplicative form does.
5. **Test coverage** of the public API ≥ 90% as measured by `coverage.py`.
6. **One headline notebook** (`notebooks/01_phase1_demo.ipynb`) that solves a 2D Gaussian → 2D Gaussian transport with all four methods (LP, multiplicative Sinkhorn, log-domain Sinkhorn, Sinkhorn divergence) and visualizes the resulting plans plus a side-by-side cost-versus-ε curve.

The phase is done when all six pass on a clean clone in CI. Not before.

## 3. Failure modes and mitigations

These are the three things most likely to derail two weeks of work. Each has a concrete tripwire and a concrete response.

**Failure mode 1: numerical instability hides correctness bugs.** Multiplicative-form Sinkhorn underflows at small ε; you start getting NaNs, treat them as "the algorithm doesn't converge here," and move on. Months later your downstream paper has results that depended on broken outputs. *Mitigation:* implement log-domain Sinkhorn first, and only after it passes its tests implement the multiplicative form (purely as an educational artifact). Never use multiplicative Sinkhorn for any test or notebook.

**Failure mode 2: tests look right, aren't testing the right thing.** Easiest version: comparing two implementations of the same wrong formula, both agree, both wrong. Slightly harder version: closed-form sanity check happens to also pass on a buggy implementation because the bug cancels in the symmetric case. *Mitigation:* every primitive is tested against (a) at least one closed-form known-answer case where I derived the answer by hand or from a textbook formula, *and* (b) POT as an independent reference. Two independent oracles. If they ever disagree on the same problem, stop and find out why before continuing.

**Failure mode 3: time spent on optimization, speed, or scope creep when only correctness matters here.** Phase 1 doesn't need to be fast. POT already exists for fast. The value of writing this is correctness and understanding, not throughput. *Mitigation:* explicit time-box per sub-phase below, with a hard rule that if a sub-phase runs over by more than 50%, I write a one-paragraph debrief on why and either descope or cut. No starting on stretch features (low-rank Sinkhorn, multi-marginal extensions, GPU acceleration) until the core list is green.

A fourth failure mode worth naming: **reading paralysis.** OT has a deep literature, and "let me read one more chapter of Villani" is a perfectly defensible-sounding way to spend a week not writing code. *Mitigation:* the reading list in §5 is the reading list. New papers don't get added during Phase 1 unless they're directly required to fix a bug.

## 4. Reading list for this phase

These are the only sources I'm planning to engage seriously during Phase 1. Other foundational works (Villani, Santambrogio) are excellent but theoretical-heavy; I'll mine them later if needed.

- **Peyré & Cuturi, *Computational Optimal Transport* (2019).** *The* practitioner reference. Chapters 2 (Theoretical Foundations), 3 (Algorithmic Foundations: LP and network simplex), and 4 (Entropic Regularization) are the core for Phase 1. Maybe ~12 hours of careful reading.
- **Cuturi (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."** The paper that made entropic OT the default in ML. Read for the algorithm and the dual interpretation.
- **Feydy et al. (2019), "Interpolating between Optimal Transport and MMD using Sinkhorn Divergences."** Defines the Sinkhorn divergence we'll implement. Read for the symmetrization argument and why it's not just a heuristic.
- **Genevay, Chizat, Bach, Cuturi, Peyré (2018), "Sample Complexity of Sinkhorn Divergences."** Background for why ε matters statistically. Skim for intuition; full theorem proofs not needed in Phase 1.
- **Schmitzer (2019), "Stabilized Sparse Scaling Algorithms for Entropy Regularized Transport Problems."** The reference for log-domain stabilization. The numerical tricks I'll implement are from here.
- **Knight (2008), "The Sinkhorn–Knopp Algorithm: Convergence and Applications."** Numerical-analysis-flavored treatment of Sinkhorn. Useful for intuition about when it converges and how fast.
- **Altschuler, Weed, Rigollet (2017), "Near-linear time approximation algorithms for optimal transport via Sinkhorn iteration."** Establishes the rate at which Sinkhorn converges to the LP solution as a function of ε. Useful when interpreting the cost-versus-ε curve in the headline notebook.
- *(Optional reach)* **Chizat, Roussillon, Léger, Vialard, Peyré (2020), "Faster Wasserstein Distance Estimation with the Sinkhorn Divergence."** Makes the case for the divergence as an estimator. Read in week 2 if there's time.

## 5. Architecture

Single Python package, `ot_primitives`. Flat-ish module structure — Phase 1 is small enough that fancy organization is overhead.

```
ot-sae-alignment/                 # repo root, will host all phases
├── ot_primitives/
│   ├── __init__.py               # public API
│   ├── exact.py                  # LP-based exact OT via scipy.linprog
│   ├── sinkhorn.py               # log-domain (default) + multiplicative
│   ├── divergence.py             # Sinkhorn divergence with symmetrization
│   ├── closed_form.py            # 1D, Bures formula for Gaussians
│   ├── costs.py                  # squared-Euclidean and other ground costs
│   └── _utils.py                 # logsumexp helpers, marginal checks
├── tests/
│   ├── conftest.py               # shared fixtures, RNG seeding
│   ├── test_exact.py
│   ├── test_sinkhorn.py
│   ├── test_divergence.py
│   ├── test_closed_form.py
│   └── test_against_pot.py       # cross-validation against POT
├── notebooks/
│   └── 01_phase1_demo.ipynb
├── docs/
│   ├── PROJECT_PLAN_PHASE_1.md   # this file
│   └── derivations.md            # by-hand derivations for closed-form tests
├── pyproject.toml
├── environment.yml
├── requirements.txt
├── .gitignore
├── .pre-commit-config.yaml
└── README.md
```

The `closed_form.py` module is small but important. It contains the Bures formula and the 1D sort-based Wasserstein computation. These are the *oracles*. Tests in `test_closed_form.py` verify the oracles against the textbook formulas (e.g., Bures applied to identical covariances should give zero). Tests in `test_sinkhorn.py` then use the oracles to verify Sinkhorn.

Public API target — what `from ot_primitives import …` should expose at end of phase:

```python
exact_ot(a, b, C) -> (cost, plan)
sinkhorn(a, b, C, eps, n_iter=1000, log_domain=True) -> (cost, plan)
sinkhorn_divergence(a, b, C_xx, C_xy, C_yy, eps) -> cost
wasserstein_1d(x_samples, y_samples, p=2) -> distance     # closed form
bures_wasserstein(mu1, cov1, mu2, cov2) -> distance       # closed form
squared_euclidean_cost(X, Y) -> C
```

That's it. No grand abstractions, no plugin systems. Add complexity only when later phases demand it.

## 6. Sub-phase breakdown

Two weeks of calendar time, ~80–100 working hours. Time estimates are upper bounds; if a sub-phase finishes early, the buffer rolls forward.

### Sub-phase 1.0 — Environment and repo setup. ~4–6 hours.

Phase 0 from your master template, scoped to this project. Conda env, `.gitignore`, `pyproject.toml`, pre-commit hooks, smoke test, GitHub repo, first commit. I'll give you the exact commands for this in the next message after you approve the plan. Reminder: GitHub repo must be created on github.com first.

**Done when:** `pytest` runs (with zero tests, exit code 0), `ruff check .` passes, smoke test prints "ok," initial commit pushed to `main`.

### Sub-phase 1.1 — LP-based exact OT and 1D closed form. ~16–20 hours.

Implement `exact_ot(a, b, C)` by reformulating the Kantorovich LP as a `scipy.optimize.linprog` call. Implement `wasserstein_1d` using the sort-based formula (for empirical measures with equal weights, W_p^p between µ and ν is the L_p distance between their sorted samples). Derive the 1D formula by hand in `docs/derivations.md`.

Tests: (i) `exact_ot` on two Diracs returns the ground cost between them; (ii) `exact_ot` against `wasserstein_1d` agrees on 1D problems; (iii) `exact_ot` agrees with `ot.emd2` on 50 random 2D problems.

**Done when:** all three test groups pass, code is documented, committed.

### Sub-phase 1.2 — Sinkhorn from scratch. ~20–24 hours.

Implement multiplicative-form Sinkhorn first as a pedagogical exercise, then immediately implement log-domain Sinkhorn using `scipy.special.logsumexp`. Document why we use log-domain (and reference Schmitzer 2019). Convergence criterion: marginal violation `‖P 1 − a‖_1 + ‖P^T 1 − b‖_1 < tol`.

Tests: (i) Sinkhorn at small ε approaches LP solution (relative cost gap < 1e-3 at ε=0.01 for moderate-size problems); (ii) marginals of the recovered plan satisfy constraints to tol; (iii) log-domain doesn't NaN at ε=0.001 where multiplicative does; (iv) agreement with POT to relative tol 1e-4.

**Done when:** all four test groups pass; one diagnostic plot in the notebook showing the cost-versus-ε curve interpolating between LP cost and the upper-bound Sinkhorn-on-uniform-plan cost.

### Sub-phase 1.3 — Sinkhorn divergence. ~12–16 hours.

Implement `S_ε(α, β) = OT_ε(α, β) − ½ OT_ε(α, α) − ½ OT_ε(β, β)`, where each `OT_ε` is the entropic-regularized transport cost. Read Feydy et al. carefully — the divergence is *not* just `OT_ε(α, β)` minus a constant. The symmetrization is what makes `S_ε(α, α) = 0` and makes the divergence a sensible interpolation between OT and MMD.

Tests: (i) `S_ε(α, α) ≤ 1e-8` on a battery of random α; (ii) `S_ε ≥ 0` on random pairs; (iii) `S_ε → W_p^p` as ε → 0 (verify trend, not exact equality); (iv) agreement with POT's `ot.bregman.empirical_sinkhorn_divergence` to relative tol 1e-4.

**Done when:** all four test groups pass and I can articulate in two sentences why the symmetrization gives `S(α, α) = 0`.

### Sub-phase 1.4 — Bures formula and the headline benchmark. ~10–14 hours.

Derive the Bures–Wasserstein formula for `W_2^2` between two Gaussians by hand (`docs/derivations.md`). Implement `bures_wasserstein(mu1, cov1, mu2, cov2)` using `scipy.linalg.sqrtm`. Then the headline experiment: sample n points from each of two known 2D Gaussians, compute (i) the Bures formula on the parameters, (ii) `exact_ot` on the samples, (iii) `sinkhorn` on the samples for ε ∈ {0.01, 0.1, 1.0}, (iv) `sinkhorn_divergence` on the samples for the same ε. Plot cost vs. n at each ε. Sample-based estimates should converge to the Bures answer as n grows; Sinkhorn divergence should converge faster than plain entropic OT (this is the practical content of Genevay et al. 2018).

Tests: Bures gives 0 for identical Gaussians, gives `‖µ1 − µ2‖²` for matched-covariance Gaussians.

**Done when:** the convergence plot is in the notebook and looks the way the theory predicts. If it doesn't, *don't normalize the plot until it does* — debug.

### Sub-phase 1.5 — Polish, document, tag. ~8–12 hours.

Fill out the README with what the package does and how to use it. Strip notebook outputs via `nbstripout`. Run the full test suite from a freshly cloned repo on a different machine (or a fresh conda env on the same machine — the cleanroom check). Tag `v0.1.0-phase1`. Push.

**Done when:** a stranger could clone the repo, run `make smoke` (or whatever we name the entry point), and see the headline notebook regenerate cleanly.

## 7. Technology choices

| Choice | Decision | Why |
|---|---|---|
| Language | Python 3.11 | Mature; matches the rest of the OT/ML ecosystem; no JAX/PyTorch dependency at this phase |
| Array backend | NumPy | Simplest; everything we need fits in memory; avoid premature optimization with JAX |
| LP solver | `scipy.optimize.linprog` (HiGHS) | Solid, free, no licensing |
| Reference impl | POT (`pot` package) | The de facto OT library; what reviewers will compare against |
| Testing | pytest + coverage | Standard; supports parametrized tests cleanly |
| Linting | ruff | Faster than flake8, single-binary; black-compatible formatter |
| Type checking | mypy in CI (lenient) | Catch the obvious mistakes; not a religious commitment to total annotation |
| Notebooks | Jupyter + nbstripout | nbstripout keeps notebook diffs sane in git |
| Env | Conda (Miniforge) + pip | What you already have configured |

Things I'm explicitly *not* using in Phase 1: JAX (no autograd needed yet), PyTorch (same), Hydra (no experiments to configure), Weights & Biases (nothing to log), Docker (overkill for a numpy library). They show up in Phase 5+ when the shape of the work changes.

## 8. Dependencies

Pinned for reproducibility. Versions chosen for "current stable as of late 2025."

| Package | Version | Why |
|---|---|---|
| python | 3.11 | Stable |
| numpy | ≥1.26 | Core arrays |
| scipy | ≥1.11 | linprog, sqrtm, logsumexp |
| pot | ≥0.9.4 | Reference implementation |
| matplotlib | ≥3.8 | Visualization |
| jupyterlab | latest | Notebook environment |
| pytest | ≥8.0 | Test runner |
| pytest-cov | ≥5.0 | Coverage |
| ruff | latest | Linter / formatter |
| mypy | latest | Type checking (lenient mode) |
| nbstripout | latest | Notebook hygiene |
| pre-commit | latest | Git hooks |

Locked in `environment.yml` (conda-side) and `requirements.txt` (pip-side, mirrored). Both committed. Versions can move forward in later phases; for Phase 1 we freeze.

## 9. Reproducibility plan

What "reproducible" means in this phase: someone clones the repo on a fresh Ubuntu 24.04 box, runs three commands, and gets the same test pass / same notebook outputs to reasonable numerical precision.

- **Seeding.** All randomized tests use `numpy.random.default_rng(seed)` with the seed declared in a fixture. No global RNG state. The headline notebook has its seed set at the top.
- **Environment lockfile.** `environment.yml` and `requirements.txt` both committed, both with version pins. Mismatches between them are a CI failure.
- **CI.** GitHub Actions: on push, install env, run `pytest`, run `ruff check`, run `nbstripout --check` on the notebook. Phase 1 CI is fast (full suite < 30 seconds) so we can afford to be strict.
- **Notebook hygiene.** `nbstripout` configured as a git filter so committed notebooks have no embedded outputs. Outputs regenerate on demand.
- **No data dependencies.** Phase 1 uses only synthetic data generated in-script. Nothing to download, nothing to lose.

## 10. Compute budget

Phase 1 runs entirely on CPU on the local machine. Zero dollars of cloud compute. The full test suite is < 30 seconds. The headline notebook is < 2 minutes including all plots. Phase 0–1 has no GPU dependency at all.

The 4GB GTX 1650 is not used in this phase. It re-enters the picture in Phase 3 when we start running models.

## 11. Open questions

Things I don't yet have firm answers on. Some will be settled by reading; some by implementation; some I'll just decide and move on.

- **Float precision.** 64-bit by default for tests (we want tight tolerances), or expose 32-bit as an option for later GPU work? *Lean:* 64-bit only in Phase 1; revisit when arrays get large.
- **Cost matrix abstraction.** Do I want a `Cost` class that knows how to compute itself lazily, or is "you pass in a precomputed `C` matrix" sufficient? *Lean:* precomputed `C` only; lazy costs are a Phase 5+ optimization if matrices get too big to materialize.
- **Sinkhorn convergence criterion.** Marginal-violation L1 norm vs. dual-gap vs. iterate-difference. Schmitzer uses marginal violation; POT uses iterate difference. *Lean:* marginal violation, log it as part of the return value so we can inspect.
- **Whether to expose multiplicative Sinkhorn at all.** It's pedagogically useful to write but dangerous to expose. *Lean:* keep it in the codebase under a `_legacy` submodule, document that production code uses log-domain only.
- **Whether to write derivations in LaTeX or markdown-with-math.** *Lean:* markdown-with-math (KaTeX-style) in `docs/derivations.md` to keep everything in one toolchain.

These are tracked in `docs/open_questions.md` so I don't lose them. Each one gets resolved or explicitly deferred by end of Phase 1.

---

**Approval gate.** Once you sign off on this plan, the next message gives Phase 0 setup as a single block: directory structure, `environment.yml`, `requirements.txt`, `.gitignore`, `pyproject.toml`, `pre-commit` config, smoke test, and the `git init` + initial-commit + remote-add + push commands, with a reminder to create the GitHub repo on github.com first.
