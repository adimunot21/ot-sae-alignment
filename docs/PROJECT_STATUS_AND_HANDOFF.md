# OT-SAE Alignment Project — Status & Handoff

This document is two things at once: a record of what we've built (so a new collaborator or LLM can pick up the project where we left off) and an explainer of the underlying material (so the work makes sense to someone who hasn't been here for the journey).

The intended reader: a hardware-engineering undergrad with comfort in linear algebra, calculus, and basic probability, but no prior OT or mech interp background.

---

## 1. The project in one paragraph

We're building toward a workshop paper at the **Optimal Transport in Machine Learning workshop at NeurIPS 2026**. The contribution is a method for matching features across two **Sparse Autoencoders** (SAEs — explained in §6) using **Gromov–Wasserstein** optimal transport (explained in §3 and §5). The standard tool for this matching is the Hungarian algorithm on cosine similarity, which fails when the two SAE dictionaries have different sizes or live in different model bases. GW handles both cases naturally because it matches based on *relational structure* rather than positional alignment. The phases are: (1) build OT fluency; (2) learn GW; (3) learn SAEs; (4) the actual matching experiments; (5) writeup.

We are at the end of **Phase 2**, ready to start **Phase 3**.

---

## 2. What is Optimal Transport?

### 2.1 The motivating analogy

You have a hundred piles of sand on a beach and a hundred holes elsewhere on the same beach. Each pile holds some amount of sand; each hole has a specific capacity. Moving sand costs effort proportional to distance. **What's the cheapest way to move all the sand into all the holes?**

That's optimal transport. The piles and holes are "measures" — distributions of mass over locations. The plan that says how much sand goes from each pile to each hole is the "transport plan." The cheapest total cost is the **Wasserstein distance** between the source and target distributions.

### 2.2 Why machine learning cares

Two probability distributions are exactly two piles of sand — abstract mass distributed over abstract locations. So OT gives us a principled way to measure how *different* two distributions are, not by comparing them at each point (that's KL divergence and friends), but by asking how much rearranging it would take to turn one into the other.

This matters because point-by-point comparisons break down when distributions don't overlap. Two Gaussians far apart on the real line have effectively zero overlap; KL says infinity (useless), but OT correctly says "they're far apart, and here's how far."

### 2.3 The two formulations: Monge and Kantorovich

**Monge (1781).** The original. Each pile of sand goes entirely to exactly one hole. Find a function `T(x)` that says "sand at location x goes to location T(x)." Clean but sometimes impossible: if you have one big pile and two small holes, Monge can't split the pile.

**Kantorovich (1942).** The relaxation that made the problem tractable. Mass can be split. Instead of a function, look for a *coupling* `π(x, y)` that says "this much mass flows from x to y." The coupling is a joint distribution whose marginals are the source and target.

The Kantorovich problem is a **linear program** — minimize a linear cost subject to linear constraints. Linear programs are the sweet spot of optimization: any standard solver handles them. That's why nearly every OT computation in ML uses Kantorovich's formulation.

Concretely, for finite measures `α = Σᵢ aᵢ δ(xᵢ)` (mass `aᵢ` at point `xᵢ`) and `β = Σⱼ bⱼ δ(yⱼ)`:

```
minimize    Σᵢⱼ Pᵢⱼ Cᵢⱼ                    (total transport cost)
subject to  Σⱼ Pᵢⱼ = aᵢ for all i           (rows sum to source mass)
            Σᵢ Pᵢⱼ = bⱼ for all j           (columns sum to target mass)
            Pᵢⱼ ≥ 0
```

Here `Cᵢⱼ` is the cost of moving one unit from `xᵢ` to `yⱼ` (typically squared Euclidean distance), and `P` is the unknown plan.

### 2.4 The 1D special case

In 1D, the optimal transport plan has a beautiful closed form: **sort both sets of points and match them in order.** No optimization needed. The reason is *cyclical monotonicity*: if any optimal plan tried to "cross" (sending a smaller source to a larger target while a larger source goes to a smaller target), uncrossing them would strictly reduce the cost. So no optimal plan crosses, which means the plan must be the sorted-to-sorted match.

For `n` source points sorted as `x₍₁₎ ≤ ... ≤ x₍ₙ₎` and target points sorted as `y₍₁₎ ≤ ... ≤ y₍ₙ₎`, all with equal weight 1/n:

```
W_p(α, β)^p = (1/n) Σᵢ |x₍ᵢ₎ - y₍ᵢ₎|^p
```

This is our **oracle** for the rest of the OT machinery: any general OT solver, run on a 1D problem, must agree with the closed form.

### 2.5 Sinkhorn: making OT scalable

The Kantorovich LP scales as roughly `O((nm)³)` in the worst case. Fine at n=20, painful at n=200, intractable at n=2000. Real problems need thousands of points.

Cuturi (2013) added a small entropy term to the objective:

```
minimize  Σᵢⱼ Pᵢⱼ Cᵢⱼ - ε · H(P)
```

where `H(P) = -Σ Pᵢⱼ log Pᵢⱼ` is the entropy of the plan. This is called **entropic OT**. The regularization makes the problem strongly convex, and the optimal plan turns out to have a special structure: `P*ᵢⱼ = uᵢ · exp(-Cᵢⱼ/ε) · vⱼ`. You can find the scaling vectors `u, v` by alternating multiplications, an algorithm that converges in milliseconds even at n=10,000. This is **Sinkhorn's algorithm**.

The price: the regularized plan is biased — it's a smoothed-out version of the true Kantorovich plan, and as `ε → ∞` it becomes the uniform plan (no real transport optimization at all). The regularization parameter `ε` is a knob:

- Small `ε` → close to the true LP, sharp plan, slow convergence, numerically dangerous.
- Large `ε` → smoothed plan, fast convergence, biased away from the truth.

### 2.6 Log-domain Sinkhorn

The naive Sinkhorn implementation computes `K = exp(-C/ε)` directly. When `ε` is small, entries of `K` underflow to zero in float64 (anything past `exp(-700)` is zero). The algorithm then divides by zero and produces NaN.

The fix is to do all computations in log-space using the `logsumexp` trick: instead of `log(Σ exp(z))`, compute `max(z) + log(Σ exp(z - max(z)))`. This never underflows, regardless of the magnitudes involved. Every modern OT library uses log-domain Sinkhorn by default.

### 2.7 Sinkhorn divergence

The entropic OT cost has a serious flaw: `OT_ε(α, α) ≠ 0`. The transport cost from a measure to itself is positive because the regularizer prefers a spread-out plan, but the optimal plan from α to α should be the identity. So the "entropic transport cost" is partly measuring α's blurring against itself — useless as a distance.

Genevay, Peyré, Cuturi (2018) fixed this with the **Sinkhorn divergence**:

```
S_ε(α, β) = OT_ε(α, β) - ½ · OT_ε(α, α) - ½ · OT_ε(β, β)
```

Three properties:

1. **Self-zero.** `S_ε(α, α) = 0` exactly.
2. **Non-negative.** `S_ε(α, β) ≥ 0`.
3. **Interpolation.** As `ε → 0`, `S_ε → W₂²` (Wasserstein). As `ε → ∞`, `S_ε → MMD²` (Maximum Mean Discrepancy, a popular kernel-based distance). Two of the most-used distance functions in ML are the endpoints of a single one-parameter family.

This last property is genuinely beautiful. It says Wasserstein and MMD aren't competing tools — they're the same tool at different settings.

### 2.8 An algebraic subtlety we discovered the hard way

There are two ways to compute Sinkhorn divergence, and they give different numbers:

- **Cost form:** subtract transport costs `<P, C>` of the three regularized plans. This is what POT (the standard OT library) computes.
- **Dual form:** subtract regularized objectives `<P, C> - ε · H(P)`, which by Lagrangian duality equals `<f, a> + <g, b>` where (f, g) are the dual potentials. This is what Feydy et al. 2019 formally define.

They differ by `ε` times an entropy gap term. Both are valid, both equal zero on self-self, both approach `W₂²` as `ε → 0`. But they are not the same object. Papers and libraries slide between them without flagging it. (See POT issue #383 for the corresponding confusion in their library.) Our code implements both and documents the relationship.

### 2.9 Bures–Wasserstein: the closed form for Gaussians

For two multivariate Gaussians `N(μ₁, Σ₁)` and `N(μ₂, Σ₂)`:

```
W_2²(α, β) = ‖μ₁ - μ₂‖² + tr(Σ₁ + Σ₂ - 2·(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})
```

The covariance term is the **Bures metric**, a real metric on positive semi-definite matrices. (Same object appears in quantum information theory as the fidelity between density operators — different communities, same math.) This is an oracle: for any pair of Gaussians, we know the exact `W₂²` analytically, so any sample-based OT solver run on samples from those Gaussians must converge to this number as the sample count grows.

---

## 3. What is Gromov–Wasserstein (GW)?

Standard OT requires the source and target to live in the same ambient space. You can't compute Wasserstein between a point cloud in 3D and a point cloud in 50D — the cost function `‖x - y‖²` doesn't make sense across mismatched dimensions.

GW (Mémoli 2011) generalizes OT to incomparable spaces. Each space is described by its *internal* pairwise distance matrix. The two distance matrices `C₁` (within space 1) and `C₂` (within space 2) need not have the same size or live in the same coordinate system. GW finds a coupling `T` that minimizes:

```
GW(α, β) = min_T  Σᵢⱼₖₗ |C₁ᵢₖ - C₂ⱼₗ|² · Tᵢⱼ · Tₖₗ
```

In words: for each pair of points (i, k) in space 1 and each pair (j, l) in space 2, look at how *similar* the pair (i, k)'s within-space distance is to (j, l)'s within-space distance. The optimal GW coupling matches points such that distances are *preserved as much as possible*.

GW is what you reach for when:
- Your two objects don't share a coordinate system.
- You care about *structure* (pairwise relationships) rather than absolute positions.
- You're matching graphs, shapes, or embeddings from different models.

GW is harder than standard OT — it's a quadratic program rather than a linear one, non-convex, with multiple local minima. POT's `entropic_gromov_wasserstein` solves an entropy-regularized version via a Sinkhorn-like inner loop.

---

## 4. The repository — what we built

The repo is at `https://github.com/adimunot21/ot-sae-alignment`. Tag `v0.1.0-phase1` marks the end of Phase 1.

### 4.1 Directory structure

```
ot-sae-alignment/
├── ot_primitives/             # didactic OT library (Phase 1)
│   ├── __init__.py
│   ├── _legacy.py             # multiplicative Sinkhorn (deprecated, deliberately broken at small ε)
│   ├── _utils.py              # marginal validators, shape checks
│   ├── closed_form.py         # bures_wasserstein
│   ├── costs.py               # squared_euclidean_cost
│   ├── divergence.py          # sinkhorn_divergence (cost and dual forms)
│   ├── exact.py               # exact_ot (LP), wasserstein_1d (sort-based)
│   └── sinkhorn.py            # log-domain Sinkhorn — the production implementation
├── tests/                     # 90 tests, 96% coverage
│   ├── conftest.py
│   ├── test_closed_form.py
│   ├── test_costs.py
│   ├── test_divergence.py
│   ├── test_exact_ot.py
│   ├── test_phase0_smoke.py
│   ├── test_sinkhorn.py
│   └── test_wasserstein_1d.py
├── notebooks/
│   └── 02_gromov_wasserstein.ipynb   # Phase 2 — GW on toy graphs
├── scripts/
│   ├── cost_vs_epsilon.py     # bias-variance trade-off visualization
│   └── smoke_test.py          # Phase 0 environment check
├── docs/
│   ├── PROJECT_PLAN_PHASE_1.md
│   ├── PHASE_1_NOTES.md       # honest framing: ot_primitives is didactic
│   ├── derivations.md         # by-hand derivations
│   └── open_questions.md
├── results/                   # experiment outputs (gitignored except where committed)
├── environment.yml
├── requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
├── Makefile
└── README.md
```

### 4.2 Phase 1 — OT primitives from scratch (DIDACTIC)

**Important framing.** The `ot_primitives` package is preserved as a learning artifact. **Production code from Phase 2 onward uses POT directly**, not our implementations. The from-scratch work served its teaching purpose and will eventually feed an appendix or tutorial in the writeup. None of `ot_primitives` is imported by Phase 2+ code.

What Phase 1 produced:
- `exact_ot(a, b, C)` — the Kantorovich LP via `scipy.optimize.linprog` with HiGHS.
- `wasserstein_1d(x, y, p)` — the sort-based closed form for 1D.
- `sinkhorn(...)` — log-domain entropic OT with a `SinkhornResult` dataclass.
- `sinkhorn_divergence(...)` — both cost-form and dual-form, with documented relationship.
- `bures_wasserstein(μ₁, Σ₁, μ₂, Σ₂)` — the Gaussian closed form.
- `squared_euclidean_cost(X, Y)` — efficient cost-matrix computation.
- A test suite that cross-validates against three independent oracles: hand-computed answers, the 1D closed form, and POT. Plus structural property tests (symmetry, non-negativity, marginal satisfaction, self-zero for divergence, etc.).
- A bias-variance visualization (`scripts/cost_vs_epsilon.py`) showing Sinkhorn cost interpolating between LP cost (at small `ε`) and uniform-plan cost (at large `ε`).

What Phase 1 taught (real, transferable understanding):
- The bias of entropic OT and why naive `OT_ε` is unusable as a distance.
- Why log-domain stabilization matters (we deliberately watched the multiplicative form fail at small `ε`).
- The cost-vs-dual-form distinction in Sinkhorn divergence and why papers/libraries are casual about it.
- Numerical floors and tolerances appropriate for OT computations.

### 4.3 Phase 2 — Gromov-Wasserstein on toy graphs

A single notebook (`notebooks/02_gromov_wasserstein.ipynb`) demonstrating GW end-to-end via POT.

**Experiment 1: permuted graph.** Build a 15-node stochastic block model graph G. Make a relabeled copy G'. Run `ot.gromov.entropic_gromov_wasserstein` on the two shortest-path-distance matrices (no node identities, no edge information, no positions provided). Recover the matching from structure alone.

Result: 100% accuracy at recovering the relabeling permutation. GW loss at machine zero.

**Experiment 2: robustness to edge noise.** Same setup but corrupt G' by adding/removing random edges. Multi-seed averaging.

Result:
| edits | mean accuracy | std |
|---|---|---|
| 0 | 100% | 0% |
| 2 | 89% | 10% |
| 5 | 45% | 32% |
| 10 | 32% | 5% |
| 20 | 15% | 11% |

Interpretation: GW recovers exact correspondence on isomorphic graphs, tolerates a couple of edits with mild loss of accuracy, becomes seed-dependent past ~5 edits, collapses toward chance (1/15 ≈ 7%) at heavy noise.

This validates the core mechanism we'll lift into Phase 4 for SAEs: build pairwise structure within each side, run entropic GW, decode the soft coupling via `argmax(T, axis=1)` to a hard matching.

---

## 5. What is a Sparse Autoencoder (SAE)?

This is what Phase 3 covers. Pre-explained here for the handoff.

### 5.1 The mechanistic interpretability problem

Modern language models (GPT, Claude, Llama, etc.) are big neural networks. Inside, intermediate activations form vectors of size hundreds-to-thousands at each layer. We'd like to know what each component of these vectors *represents* — does dimension 437 of layer 8 in GPT-2 mean "the current token is a verb"? Does it mean "we are in a paragraph about cooking"? Does it mean nothing in isolation?

The empirical answer is: usually, individual neurons don't correspond to clean concepts. Concepts are stored as *directions* in the activation space, and many concepts are stored in superposition (multiple concepts share components). This is called the **superposition hypothesis** (Elhage et al. 2022).

### 5.2 What an SAE does

An SAE is a small neural network that we train alongside (but separately from) the main model. It takes the model's activation vector `h ∈ ℝᵈ` (where `d` might be 768) and:

1. Encodes it into a *much wider* sparse representation `f = ReLU(W_enc · h + b_enc)`, where `f ∈ ℝᴺ` with `N >> d`. Typical: d=768, N=24576.
2. Decodes back: `h' = W_dec · f`. Trained so that `h' ≈ h`.
3. The training objective penalizes `f` for being non-sparse (typically with an L1 penalty, or by hard-capping the number of non-zero entries).

The result: each vector `h` is reconstructed using only a handful of the N "features" (typically 50–200 active out of 24,000). Each feature, defined by its decoder direction `W_dec[:, k]`, corresponds to *something* — and the hope is that "something" is more interpretable than a single neuron.

### 5.3 Why this works

If the model genuinely stores concepts as directions in superposition, and there are *more* concepts than the activation space has dimensions, then:
- A linear projection to a lower-dimensional space (the activation itself) compresses the concepts.
- A wider sparse expansion (the SAE's feature space) *decompresses* them.
- With enough sparsity, each feature corresponds to a single concept.

This is the central thesis of mechanistic interpretability circa 2023–2026. It's well-supported empirically: trained SAE features often correspond to interpretable patterns ("the current token is a Python keyword," "we are in a paragraph about a specific city," "sentence is in past tense," etc.).

### 5.4 Why the matching problem matters

Train two SAEs with different random seeds on the same model. They will both learn meaningful features, but the indices of the features will be different — feature 437 in SAE A might be the same concept as feature 12,891 in SAE B. To compare them, you need a **matching** between the two feature dictionaries.

Current state of the art: cosine similarity between decoder rows + Hungarian algorithm. This works when:
- Both dictionaries have the same size N.
- Both SAEs are trained on the same model (so feature directions live in the same coordinate system).

It doesn't work when:
- Dictionary sizes differ (Hungarian needs square matrices).
- The two SAEs are trained on *different* models (decoder directions live in incomparable spaces).

GW is the natural tool here: it doesn't need a shared coordinate system, just within-side structure. The within-side structure of an SAE is the pairwise distance matrix between its features. Match the structure, match the features — even when the spaces don't line up.

### 5.5 An analogy for the hardware-engineering reader

Imagine two FPGA designs that compute the same function but use different routing and different LUT assignments. There's no direct correspondence between LUT 437 in chip A and any specific LUT in chip B. But the *graph structure* of which LUTs feed into which is meaningful. Two LUTs that drive each other in chip A correspond to two LUTs that drive each other in chip B, even though their absolute identities differ. That's a graph-isomorphism flavor of matching — and that's exactly what GW does on the activation-correlation matrices of two SAEs.

---

## 6. Where we are, what's next

### 6.1 Done
- Phase 0: environment, repo, CI scaffolding.
- Phase 1: OT primitives from scratch (didactic — preserved but not imported).
- Phase 2: GW notebook on toy graphs (uses POT).

### 6.2 In progress: Phase 3 — SAE crash course (~5 days)

Goal: by the end, learner can load a pretrained SAE on GPT-2-small, extract activations, run them through the SAE, and for any feature index, identify what it represents from its top-k activating tokens.

Reading list:
1. Bricken et al. (2023) "Towards Monosemanticity" — the foundational SAE paper.
2. A Neel Nanda blog post on SAEs for orientation.
3. Lieberum et al. (2024) "Gemma Scope" — modern industrial SAE.
4. `sae_lens` library README.

Notebook (`notebooks/03_sae_basics.ipynb`):
- Load GPT-2-small + a pretrained residual-stream SAE on layer 8 via `sae_lens`.
- Run model on a corpus, extract activations, encode through SAE.
- For sample features, print top-k activating tokens with context.
- Form an opinion about what features represent.

Status: tooling installed (`sae_lens 6.43.0`, `transformer_lens` confirmed installed). Notebook created but no cells run yet. **Phase 3 is the immediate next step.**

### 6.3 Phase 4 — The actual research (~3-4 weeks)

This is where the contribution lives. Three sub-experiments:

**4a — Equal-size SAE matching.** Two SAEs of identical shape trained with different random seeds on the same model. Apply (i) cosine-Hungarian, (ii) entropic GW on decoder directions, (iii) entropic fused-GW combining decoder directions and activation patterns. Evaluate via matched-pair activation correlation on held-out tokens.

Hypothesis: GW does at least as well as Hungarian here. If it doesn't, that's already a meaningful finding.

**4b — Unequal-size SAE matching.** Two SAEs with different dictionary sizes (e.g., 4096 vs 16384 features) on the same model. Apply (i) padded-Hungarian (ii) partial OT (iii) entropic GW. Hungarian fundamentally cannot handle this case well — that's the strong novelty pitch.

Hypothesis: GW substantially outperforms anything Hungarian-based here.

**4c (optional, if time and 4a/4b are clean) — Cross-model SAE matching.** SAE on GPT-2-small vs SAE on GPT-2-medium. Different residual-stream dimensions, different model bases. Can GW recover meaningful correspondences?

Hypothesis: this is the hardest case but also the most exciting if it works. Reviewers will care most about this result.

Evaluation primitives across all three:
- Matched-pair activation correlation on held-out tokens.
- Ablation equivalence: do paired features cause similar downstream effects when ablated?
- Interpretability transfer: do auto-generated descriptions of features transfer between paired SAEs?

### 6.4 Phase 5 — Writeup (~2 weeks)

4–8 page workshop paper. Sections: introduction, background (OT and SAEs), method, experiments, results, discussion, limitations, future work. The didactic Phase 1 material may become an appendix or a standalone tutorial chapter.

Target: OTML @ NeurIPS 2026, deadline likely early-October 2026.

---

## 7. How to pick up the project

If you're a new collaborator (or a new LLM session continuing the work), here's the runway:

1. Clone the repo. `git clone git@github.com:adimunot21/ot-sae-alignment.git`.
2. Install: `conda env create -f environment.yml && conda activate ot-sae-alignment && pip install -e . && pip install sae-lens transformer-lens datasets`.
3. Verify Phase 1 still passes: `make test`. Should print `90 passed`.
4. Open `notebooks/02_gromov_wasserstein.ipynb` and run all cells. Should reproduce the matching-accuracy table from §4.3.
5. Read `docs/PROJECT_PLAN_PHASE_1.md` and `docs/PHASE_1_NOTES.md` for the framing.
6. Read this document.
7. The immediate next task is Phase 3: create `notebooks/03_sae_basics.ipynb`, work through the cells described in §6.2, write the "what I learned" deliverable.

### 7.1 Working principles to preserve

- **POT is trusted; we don't reimplement it.** The didactic phase is closed.
- **Tests for any non-trivial numerical primitive,** with at least one closed-form or independent-library oracle.
- **Reproducibility.** Seeds set explicitly; configs saved alongside outputs; notebooks have outputs stripped via `nbstripout`.
- **Honest assessment of results.** Single-seed numbers are not results. A null finding cleanly reported is publishable.
- **The contribution isn't the OT machinery.** It's the application to SAE matching where existing tools (Hungarian) fundamentally fail.

### 7.2 Open questions to revisit

- Within-SAE distance metric for GW: cosine on decoder rows, or activation-pattern correlation, or a combination via fused-GW?
- Choice of `ε` in the SAE setting: needs empirical tuning, no good a priori value.
- For unequal-size matching: partial OT vs entropic GW with non-uniform marginals vs rectangular GW formulations — which one fits cleanest?
- Cross-model setup requires aligning two different residual-stream coordinate systems first; GW should handle this implicitly, but verifying that empirically is the most-uncertain part of the project.

### 7.3 Compute budget

Hard ceiling: $50 of Runpod. So far: $0 spent. Phase 3 runs entirely on the local GTX 1650 (4GB). Phase 4 will need real GPU time at the largest dictionary sizes; budget ~$15–25 there. Phase 4c (cross-model) is the only piece that might push the budget; descope if it's clearly going to bust.

---

## 8. References

OT theory:
- Peyré & Cuturi (2019), *Computational Optimal Transport*. Free PDF at https://arxiv.org/abs/1803.00567. The practitioner reference.
- Mémoli (2011), "Gromov–Wasserstein Distances and the Metric Approach to Object Matching." Foundational.
- Peyré, Cuturi, Solomon (2016), "Gromov-Wasserstein Averaging of Kernel and Distance Matrices." The practical GW algorithm.
- Cuturi (2013), "Sinkhorn Distances." Made entropic OT mainstream in ML.
- Genevay, Peyré, Cuturi (2018), "Learning Generative Models with Sinkhorn Divergences." Defines the divergence.
- Feydy et al. (2019), "Interpolating between OT and MMD using Sinkhorn Divergences." The dual-form analysis.

SAE / mech interp:
- Bricken et al. (2023), "Towards Monosemanticity." Anthropic's foundational SAE paper.
- Templeton et al. (2024), "Scaling Monosemanticity." What SAE scaling buys you.
- Lieberum et al. (2024), "Gemma Scope." Industrial-scale SAEs from DeepMind.
- Elhage et al. (2022), "Toy Models of Superposition." Why SAEs work in principle.

Tools:
- POT (Python Optimal Transport): https://pythonot.github.io/
- sae_lens: https://github.com/jbloomAus/SAELens
- transformer_lens: https://github.com/TransformerLensOrg/TransformerLens

---

*Last updated at the end of Phase 2. Next session begins Phase 3.*
