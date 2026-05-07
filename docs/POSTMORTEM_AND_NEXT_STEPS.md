# OT-SAE Alignment Project — Comprehensive Status & Post-Mortem

This document is the canonical reference for the project as of the end of Phase 4a (initial attempt). It covers everything: the math, the code, what we tried, what worked, what didn't, why, and what we're doing next. It is written so that a new collaborator (or new LLM session) can pick up the project mid-flight without losing context, and so a hardware-engineering-undergrad-level reader can follow the technical content without prior OT or interpretability background.

If you only read one section, read §10 (the post-mortem of Phase 4a) and §11 (the path forward). Everything before §10 is context.

---

## Table of contents

1. The project in one paragraph
2. What is Optimal Transport (OT)?
3. What is Gromov-Wasserstein (GW)?
4. What is a Sparse Autoencoder (SAE)?
5. The original research thesis
6. Phase-by-phase work log
7. The repository — what's in it
8. Phase 4a — what we tried
9. Phase 4a — what we measured
10. Phase 4a — what went wrong and why (the post-mortem)
11. The path forward: fused Gromov-Wasserstein
12. What success looks like from here
13. Glossary
14. References and tooling

---

## 1. The project in one paragraph

We are working toward a workshop paper at the Optimal Transport in Machine Learning workshop at NeurIPS 2026. The original contribution was a method for matching features across two **Sparse Autoencoders** (SAEs — explained in §4) using **Gromov–Wasserstein** optimal transport (GW — explained in §3). The standard tool for this matching is the Hungarian algorithm on cosine similarity, which fails when two SAE dictionaries differ in size or live in different model bases. GW was supposed to handle both cases by matching *relational structure* rather than positional alignment. Phase 4a (the first real experiment) showed that **naive GW on within-SAE distance matrices performs at chance** on real SAE data — a clear negative result. We are now pivoting to **fused GW**, which combines within-side structure with cross-side similarity, before deciding whether to commit to a "negative result" paper or pivot the project entirely.

We are at the start of **Phase 4a-bis** (fused-GW retry of Phase 4a).

---

## 2. What is Optimal Transport (OT)?

### 2.1 The motivating analogy

You have a hundred piles of sand on a beach, and a hundred holes elsewhere. Each pile holds some amount of sand; each hole has a fixed capacity. Moving sand costs effort proportional to distance. **What's the cheapest plan to move all the sand into all the holes?**

That's optimal transport. The piles and holes are *measures* — distributions of mass over locations. The cheapest plan is the *transport plan*. The total cost of the cheapest plan is the **Wasserstein distance** between the source and target distributions.

### 2.2 Why ML cares

Two probability distributions are exactly two piles of sand — abstract mass over abstract locations. So OT gives us a principled way to measure how *different* two distributions are: not by comparing them at each point (KL divergence and friends), but by asking how much rearranging it would take to turn one into the other.

This matters because point-by-point comparisons break when distributions don't overlap. Two Gaussians far apart on the number line have effectively zero overlap; KL says infinity (useless), but OT correctly says "they're far apart, by this much."

### 2.3 The two formulations

**Monge (1781).** Each pile of sand goes entirely to exactly one hole. Find a function `T(x)` that says "sand at x goes to T(x)." Clean but sometimes impossible: if you have one big pile and two small holes, Monge can't split.

**Kantorovich (1942).** Mass can be split. Find a *coupling* `π(x, y)` that says "this much mass flows from x to y." A coupling is a joint distribution whose marginals are the source and target. Mathematically, it's a linear program — minimize a linear cost subject to linear constraints — which makes it tractable.

Concretely, for finite measures `α = Σᵢ aᵢ δ(xᵢ)` (mass `aᵢ` at point `xᵢ`) and `β = Σⱼ bⱼ δ(yⱼ)`:

```
minimize    Σᵢⱼ Pᵢⱼ Cᵢⱼ                    (total transport cost)
subject to  Σⱼ Pᵢⱼ = aᵢ for all i           (rows sum to source mass)
            Σᵢ Pᵢⱼ = bⱼ for all j           (columns sum to target mass)
            Pᵢⱼ ≥ 0
```

Here `Cᵢⱼ` is the cost of moving one unit from `xᵢ` to `yⱼ` (typically squared Euclidean distance), and `P` is the unknown plan — the coupling matrix.

### 2.4 1D special case

In 1D, the optimal plan has a closed form: **sort both sets of points and match them in order.** No optimization needed. The reason is *cyclical monotonicity*: if any optimal plan tried to "cross" (sending a smaller source to a larger target while a larger source goes to a smaller target), uncrossing them would strictly reduce the cost. So no optimal plan crosses.

For `n` source points sorted as `x₍₁₎ ≤ ... ≤ x₍ₙ₎` and target points sorted as `y₍₁₎ ≤ ... ≤ y₍ₙ₎`, all with equal weight 1/n:

```
W_p(α, β)^p = (1/n) Σᵢ |x₍ᵢ₎ - y₍ᵢ₎|^p
```

This is our **oracle** for general OT solvers: any solver run on a 1D problem must agree with this closed form.

### 2.5 Sinkhorn: making OT scalable

The Kantorovich LP scales as roughly `O((nm)³)` worst-case. Fine at n=20, painful at n=200, intractable at n=2000.

Cuturi (2013) added a small entropy term to the objective:

```
minimize  Σᵢⱼ Pᵢⱼ Cᵢⱼ - ε · H(P)
```

where `H(P) = -Σ Pᵢⱼ log Pᵢⱼ`. This is **entropic OT**. The regularization makes the problem strongly convex, and the optimal plan turns out to have the form `P*ᵢⱼ = uᵢ · exp(-Cᵢⱼ/ε) · vⱼ`. You find `u, v` by alternating multiplications — **Sinkhorn's algorithm** — which converges in milliseconds even at n=10,000.

The price: the regularized plan is biased — it's a smoothed-out version of the true Kantorovich plan. As `ε → ∞` the plan becomes uniform (no real transport optimization at all). As `ε → 0` the plan approaches the true LP solution but the algorithm becomes numerically dangerous.

The regularization parameter `ε` is a knob:
- Small `ε` → close to the true LP, sharp plan, slow convergence, numerical instability.
- Large `ε` → smoothed plan, fast convergence, biased away from the truth.

### 2.6 Log-domain Sinkhorn

Naive Sinkhorn computes `K = exp(-C/ε)` directly. When `ε` is small, entries of `K` underflow to zero in float64 (anything past `exp(-700)` is zero). The fix: do all computations in log-space using the `logsumexp` trick. Every modern OT library uses log-domain Sinkhorn by default.

### 2.7 Sinkhorn divergence

The entropic OT cost has a flaw: `OT_ε(α, α) ≠ 0`. The transport cost from a measure to itself is positive because the regularizer prefers a spread-out plan, but the optimal plan from α to α should be the identity. So the "entropic transport cost" is partly measuring α's blurring against itself — useless as a distance.

Genevay, Peyré, Cuturi (2018) fixed this with the **Sinkhorn divergence**:

```
S_ε(α, β) = OT_ε(α, β) - ½ · OT_ε(α, α) - ½ · OT_ε(β, β)
```

Three properties:
1. **Self-zero.** `S_ε(α, α) = 0` exactly.
2. **Non-negative.** `S_ε(α, β) ≥ 0`.
3. **Interpolation.** As `ε → 0`, `S_ε → W_2²` (Wasserstein). As `ε → ∞`, `S_ε → MMD²` (a kernel-based distance).

The third property is genuinely beautiful. Wasserstein and MMD aren't competing tools — they're the same tool at different settings, with `ε` as the dial.

### 2.8 An algebraic subtlety

There are two ways to compute Sinkhorn divergence, and they give different numbers:
- **Cost form:** subtract transport costs `<P, C>` of the three regularized plans (what POT computes).
- **Dual form:** subtract regularized objectives `<P, C> - ε · H(P)`, which by Lagrangian duality equals `<f, a> + <g, b>` where (f, g) are the dual potentials (what Feydy et al. 2019 formally define).

They differ by `ε` times an entropy gap term. Both are valid, both equal zero on self-self, both approach `W_2²` as `ε → 0`. Papers and libraries slide between them without flagging it.

### 2.9 Bures-Wasserstein for Gaussians

For two multivariate Gaussians `N(μ₁, Σ₁)` and `N(μ₂, Σ₂)`:

```
W_2²(α, β) = ‖μ₁ - μ₂‖² + tr(Σ₁ + Σ₂ - 2·(Σ₁^{1/2} Σ₂ Σ₁^{1/2})^{1/2})
```

This is an oracle for sample-based OT solvers: for any pair of Gaussians, the exact `W_2²` is known analytically.

---

## 3. What is Gromov-Wasserstein (GW)?

Standard OT requires the source and target to live in the same ambient space. You can't compute Wasserstein between a point cloud in 3D and a point cloud in 50D — the cost function `‖x - y‖²` doesn't make sense across mismatched dimensions.

GW (Mémoli 2011) generalizes OT to incomparable spaces. Each space is described by its *internal* pairwise distance matrix. The two distance matrices `C₁` (within space 1) and `C₂` (within space 2) need not have the same size or live in the same coordinate system. GW finds a coupling `T` that minimizes:

```
GW(α, β) = min_T  Σᵢⱼₖₗ |C₁ᵢₖ - C₂ⱼₗ|² · Tᵢⱼ · Tₖₗ
```

In words: for each pair of points (i, k) in space 1 and each pair (j, l) in space 2, look at how *similar* the within-side distance from (i, k) is to the within-side distance from (j, l). The optimal coupling matches points so distances are preserved as much as possible.

GW is the right tool when:
- The two objects don't share a coordinate system.
- You care about *structure* (pairwise relationships) rather than absolute positions.
- You're matching graphs, shapes, or embeddings from different models.

GW is harder than standard OT — quadratic instead of linear, non-convex, multiple local minima. POT's `entropic_gromov_wasserstein` solves an entropy-regularized version with a Sinkhorn-style inner loop.

**Crucial property of GW that turned out to matter for our project:** GW only sees within-side structure. It receives `C₁` and `C₂` separately and must align them based on internal patterns. If both `C₁` and `C₂` are nearly featureless — say, all entries roughly equal — GW has nothing to lock onto and its plan collapses to uniform.

---

## 4. What is a Sparse Autoencoder (SAE)?

### 4.1 The mechanistic interpretability problem

Modern language models (GPT, Claude, Llama) are big neural networks. Inside, intermediate activations form vectors of size hundreds to thousands at each layer. We'd like to know what each component represents — does dimension 437 of layer 8 in GPT-2 mean "the current token is a verb"? "We're in a paragraph about cooking"? Nothing in particular?

The empirical answer: usually, individual neurons don't correspond to clean concepts. Concepts are stored as *directions* in the activation space, and many concepts share components. This is called **superposition** (Elhage et al. 2022).

### 4.2 What an SAE does

An SAE is a small neural network we train alongside (but separately from) the main model. It takes the model's activation vector `h ∈ ℝᵈ` (where `d` might be 768) and:

1. Encodes it into a *much wider* sparse representation `f = ReLU(W_enc · h + b_enc)`, where `f ∈ ℝᴺ` with `N >> d`. Typical: d=768, N=24,576.
2. Decodes back: `h' = W_dec · f`. Trained so that `h' ≈ h`.
3. The training objective penalizes `f` for being non-sparse (typically with an L1 penalty).

The result: each vector `h` is reconstructed using only a handful of the N "features" — typically 50–200 active out of 24,000. Each feature, defined by its decoder direction `W_dec[:, k]`, corresponds to *something* — and the hope is that "something" is more interpretable than a single neuron.

### 4.3 Why this works (in theory)

If the model genuinely stores concepts as directions in superposition, and there are *more* concepts than the activation space has dimensions, then a wider sparse expansion (the SAE's feature space) decompresses them. With enough sparsity, each feature corresponds to a single concept.

Empirically this mostly works: trained SAE features often correspond to interpretable patterns ("Python keyword," "we're discussing a specific city," "past tense"). They're not perfect, but they're better than raw neurons.

### 4.4 Hardware-engineering analogy

Imagine two FPGA designs that compute the same function but use different routing and different LUT assignments. There's no direct correspondence between LUT 437 in chip A and any specific LUT in chip B. But the *graph structure* of which LUTs feed into which is meaningful — two LUTs that drive each other in chip A correspond to two LUTs that drive each other in chip B. That's a graph-isomorphism flavor of matching.

In our project: train two SAEs with different random seeds (or on different layers, or on different models). Both learn meaningful features, but the indices differ. To compare them, you need a **matching** between the two feature dictionaries.

---

## 5. The original research thesis

The proposed contribution: GW for matching SAE feature dictionaries.

### 5.1 The current standard (Hungarian)

Today's tool: cosine similarity between decoder rows + Hungarian algorithm (`scipy.optimize.linear_sum_assignment`). This works when:
- Both dictionaries have the same size N.
- Both SAEs are trained on the same model (so feature directions live in the same coordinate system).

### 5.2 Where Hungarian fails

- **Unequal dictionary sizes** — Hungarian requires square matrices; padding doesn't work cleanly.
- **Cross-model SAEs** — decoder directions live in incomparable spaces; cosine similarity isn't meaningful.

### 5.3 The original GW pitch

GW doesn't need a shared coordinate system, just within-side structure. Compute pairwise distances *inside* each SAE; run GW. The within-side structure of an SAE is the pairwise distance matrix between its features (e.g., cosine distance between decoder rows). Match the structure, match the features — even when the spaces don't line up.

This is the thesis I sold you in the original proposal. Phase 4a tested it. It did not survive contact with reality (see §10).

---

## 6. Phase-by-phase work log

| Phase | Status | Description |
|---|---|---|
| 0 | Done | Environment setup, repo, CI scaffolding. |
| 1 | Done (didactic, mothballed) | OT primitives from scratch — exact LP, log-domain Sinkhorn, Sinkhorn divergence, Bures formula. 90 tests pass. Preserved in `ot_primitives/` but not used in Phase 2+. |
| 2 | Done | Toy graph experiment with GW via POT. Recovered isomorphism at 100% accuracy on clean inputs; degraded gracefully under edge noise. Pipeline pattern lifted into Phase 4. |
| 3 | Done | SAE crash course. Loaded GPT-2-small + pretrained SAE, found "document-boundary detector" features (high-magnitude features on `<\|endoftext\|>`), formed an opinion about feature splitting. |
| 4a | **Done — negative result** | Cross-layer SAE matching (layer 7 vs layer 8 SAEs). Tested cosine-Hungarian, activation-Hungarian, entropic GW. **GW failed catastrophically.** See §10. |
| 4a-bis | Next | Fused-GW retry. Combine within-side GW structure with cross-side direct similarity. |
| 4b | Pending | Unequal-size SAE matching (was original main contribution; viable only if 4a-bis works). |
| 4c | Pending | Cross-model SAE matching (highest-risk; only attempted if 4a-bis and 4b succeed). |
| 5 | Pending | Paper writeup. Includes Phase 1 didactic material as appendix or tutorial chapter. |

---

## 7. The repository — what's in it

The repo is at `https://github.com/adimunot21/ot-sae-alignment`. Tag `v0.1.0-phase1` marks the end of Phase 1.

```
ot-sae-alignment/
├── ot_primitives/             # didactic OT library (Phase 1, mothballed)
│   ├── __init__.py
│   ├── _legacy.py             # multiplicative Sinkhorn (deliberately broken at small ε)
│   ├── _utils.py
│   ├── closed_form.py         # bures_wasserstein
│   ├── costs.py
│   ├── divergence.py          # sinkhorn_divergence (cost & dual forms)
│   ├── exact.py               # exact_ot, wasserstein_1d
│   └── sinkhorn.py            # log-domain Sinkhorn
├── ot_sae/                    # PRODUCTION CODE for project (POT-based)
│   ├── __init__.py
│   ├── activations.py         # collect_sae_features, top_active_features
│   ├── matching.py            # cosine_hungarian, activation_hungarian, gw_matching
│   └── evaluation.py          # evaluate_matching (held-out correlation metrics)
├── tests/                     # 95 tests, ~96% coverage on ot_primitives
│   ├── conftest.py
│   ├── test_closed_form.py
│   ├── test_costs.py
│   ├── test_divergence.py
│   ├── test_exact_ot.py
│   ├── test_matching.py       # synthetic-data sanity tests for ot_sae.matching
│   ├── test_phase0_smoke.py
│   ├── test_sinkhorn.py
│   └── test_wasserstein_1d.py
├── notebooks/
│   ├── 02_gromov_wasserstein.ipynb   # Phase 2
│   ├── 03_sae_basics.ipynb           # Phase 3
│   └── 04_cross_layer_matching.ipynb # Phase 4a (current)
├── scripts/
│   ├── cost_vs_epsilon.py
│   └── smoke_test.py
├── docs/
│   ├── PROJECT_PLAN_PHASE_1.md
│   ├── PHASE_1_NOTES.md
│   ├── PROJECT_STATUS_AND_HANDOFF.md  # earlier (now superseded by this document)
│   ├── derivations.md
│   └── open_questions.md
├── results/                   # experiment outputs (gitignored except final figures)
├── environment.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 7.1 Key working principle preserved across phases

**`ot_primitives` is didactic.** Production code from Phase 2 onward uses POT directly. The from-scratch implementations served their teaching purpose and may eventually feed an appendix or tutorial in the writeup. None of `ot_primitives` is imported by Phase 2+ code. Don't re-derive what POT does well.

---

## 8. Phase 4a — what we tried

### 8.1 Setup

- Two SAEs from `gpt2-small-res-jb`: layer 7 and layer 8 of GPT-2-small. Both 24,576 features over 768-dim residual stream. Same training data, same architecture, just different layers.
- Question: do features at layer 7 have layer-8 counterparts?
- Restricted matching to top-2000 most-frequently-firing features per layer (24576 × 24576 was too large for CPU memory).
- 100 documents × 128 tokens ≈ 12,800 tokens; 70/30 train/eval split.
- Three matching methods compared.

### 8.2 The three methods

**Cosine-Hungarian.** L2-normalize each SAE's decoder rows. Compute pairwise cosine similarity matrix between layer-7 and layer-8 decoder rows (cross-side). Solve linear assignment via `scipy.optimize.linear_sum_assignment`.

**Activation-Hungarian.** Run both SAEs on the train tokens. For each (i, j) pair, compute Pearson correlation between feature-i's activation pattern in SAE A and feature-j's activation pattern in SAE B. Solve linear assignment on the negative correlation matrix.

**Entropic GW.** Compute pairwise *cosine distance* matrix *within* each SAE (so two 2000×2000 matrices). Run `ot.gromov.entropic_gromov_wasserstein` with ε=5e-3. Decode the soft coupling via `argmax(T, axis=1)`.

### 8.3 Evaluation metric

For each matching, on the held-out 30% of tokens: for each pair (i, π(i)), compute Pearson correlation between feature i's activation and feature π(i)'s activation. Report mean, median, quartiles, and fraction of pairs above various correlation thresholds (0.3, 0.5, 0.7).

---

## 9. Phase 4a — what we measured

### 9.1 Headline numbers

| Method | mean_corr | median_corr | frac > 0.3 | frac > 0.5 | frac > 0.7 |
|---|---|---|---|---|---|
| Cosine-Hungarian | 0.329 | 0.260 | 45.5% | 27.5% | 13.8% |
| Activation-Hungarian | 0.314 | 0.234 | 43.9% | 27.4% | 14.0% |
| **Entropic GW (cosine dist)** | **0.043** | **0.001** | **3.3%** | **1.7%** | **1.0%** |

GW is at chance. Hungarian methods find real signal — about 28% of feature pairs have held-out activation correlation > 0.5, which is real, structurally meaningful matching.

### 9.2 Diagnostics on what GW was doing

Within-SAE distance matrices (cosine):
- Layer 7: mean=0.991, std=0.126, range [0, 1.78]
- Layer 8: mean=0.992, std=0.112, range [0, 1.79]

Both are nearly featureless. Almost all pairs of decoder vectors are nearly orthogonal (cosine distance ~1).

GW transport plan diagnostics:
- max entry = 2.94e-04 (vs uniform baseline 1/n² = 2.5e-07; so plan max is 1000× uniform but tiny in absolute terms)
- mean of top-5 per row = 3.15e-06
- fraction of mass on argmax = 0.79%

The plan is essentially uniform. argmax extracts noise.

Matching agreement:
- cosine vs activation: 49% (the two Hungarian methods only half-agree)
- GW vs cosine: 2.3%
- GW vs activation: 2.0%

### 9.3 Diagnostic retry: co-firing distance

Tried a different within-SAE distance: `1 - corr(activation_i, activation_j)` over training tokens. Two features that always fire together get distance 0; two that never co-fire get ~1.

- Layer 7 co-fire: mean=0.999, std=0.021
- Layer 8 co-fire: mean=0.999, std=0.022

*Even less* variance than cosine. GW plan collapsed completely (max=2.6e-07, exactly uniform). All correlations near zero or slightly negative.

---

## 10. Phase 4a — what went wrong and why (the post-mortem)

This is the critical section. I owe you an honest account.

### 10.1 The proximate cause

GW's transport plan never developed peaks. The plan was essentially uniform — every feature in SAE A was matched to every feature in SAE B with roughly equal probability. argmax on a uniform plan extracts random noise.

### 10.2 Why the plan was uniform

GW only sees within-side structure. Both within-SAE distance matrices we tried were nearly featureless:
- Cosine distances: std=0.12 around mean=0.99 (decoder vectors nearly orthogonal)
- Co-firing distances: std=0.02 around mean=0.999 (features almost never co-fire substantially)

The signal-to-noise ratio of within-side structure is too low. The entropic regularizer (ε=5e-3) dominates the signal and smooths everything to uniform.

### 10.3 Why this isn't a tuning issue

I considered shrinking ε. With std=0.02 on co-firing distances, you'd need ε on the order of 1e-5 or smaller for the regularizer not to dominate. At that ε, POT's entropic GW becomes numerically unstable and very slow. Even if you could push it through, you'd be on the edge of where the algorithm is doing anything meaningful.

The deeper issue: even an *unregularized* GW would struggle because the within-side structure is genuinely close to uniform. There just isn't much variation in within-SAE distances to encode meaningful structural information.

### 10.4 Why Hungarian works

Hungarian uses *cross-side* information directly. Cosine-Hungarian compares decoder direction in layer-7 to decoder direction in layer-8 — both live in the same 768-dim residual stream, so this comparison is meaningful. Activation-Hungarian compares firing patterns over the same tokens, which is direct evidence of what each feature responds to.

Both methods see direct evidence of feature correspondence. GW only sees within-side structure, which doesn't carry enough information.

### 10.5 The deeper conceptual issue with the original thesis

I sold you "GW handles incomparable spaces, so it'll work where Hungarian fails." This is *partly* right — GW genuinely handles incomparable spaces — but I conflated "handles" with "works well on." GW needs *meaningful* within-side structure. SAE features don't naturally have that structure under either of the obvious metrics.

The Phase 2 graph experiment misled me here. Graphs have rich within-side structure (triangles, paths, communities, degree distributions) that GW can lock onto. SAE feature dictionaries do not — they're closer to a cloud of nearly-orthogonal vectors with little geometric organization beyond the high-level "features that co-fire share some semantic space."

### 10.6 What's salvageable

Three things, in increasing order of how much they matter:

1. **Infrastructure works.** Activation collection, evaluation, three matching baselines, the experimental framework — all functional and reusable.

2. **The negative result is real and publishable.** "We tested the natural GW formulation for SAE feature matching across two distance metrics; both perform at chance because within-SAE structure is too uniform to carry signal" is a careful, OT-grounded result that closes off a popular intuition. OTML reviewers would respect this.

3. **Fused-GW combines both signals.** Hungarian uses cross-side; GW uses within-side. Fused-GW (Vayer et al. 2018) uses both. If fused-GW works, the project's contribution is "the right OT formulation for SAE feature matching is fused-GW; here are the regimes where it beats each baseline." That's a real positive result.

---

## 11. The path forward: fused Gromov-Wasserstein

### 11.1 What fused-GW is

Fused Gromov-Wasserstein (Vayer et al. 2018) is a convex combination of standard OT (cross-side similarity) and standard GW (within-side structure):

```
FGW(α, β) = min_T  α · Σᵢⱼ Mᵢⱼ Tᵢⱼ  +  (1-α) · Σᵢⱼₖₗ |C₁ᵢₖ - C₂ⱼₗ|² Tᵢⱼ Tₖₗ
```

where:
- `M` is the *cross-side* cost matrix (e.g., 1 - cosine_similarity between layer-7 feature i and layer-8 feature j).
- `C₁, C₂` are the within-side distance matrices.
- `α ∈ [0, 1]` is the fusion parameter. α=1 is pure Wasserstein (cosine-Hungarian's logic); α=0 is pure GW.

### 11.2 Why this should help

The cross-side cost `M` provides *direct* matching signal — exactly what makes Hungarian work. The within-side terms provide *structural* refinement — features that fire on the same things in both SAEs should be matched, even if their exact decoder directions don't align perfectly. Together, they should give us at least Hungarian-level performance, with the structural term providing a tiebreaker when cross-side similarity is ambiguous.

### 11.3 Why this is also the right tool for the unequal-size case

This is the key strategic point. In Phase 4b, we'll match a 24,576-feature SAE to (say) a 4,096-feature SAE. Hungarian fundamentally can't run on rectangular matrices. Fused-GW can, because GW handles unequal sizes natively, and the cross-side `M` matrix is just rectangular — also fine.

So fused-GW is the unified tool for *all three* cases:
- Equal-size, same model (Phase 4a-bis): should match Hungarian's performance.
- Unequal-size, same model (Phase 4b): Hungarian fails; fused-GW works.
- Cross-model (Phase 4c): cross-side similarity is meaningless (different bases), so the fused-GW degenerates toward pure GW — we'd hope the within-side structure is enough.

### 11.4 What's coming next

Add `fused_gw_matching()` to `ot_sae/matching.py`. Add a synthetic-data test to `tests/test_matching.py`. Add a cell to the notebook running fused-GW alongside the existing three methods. Sweep `α` in {0.1, 0.3, 0.5, 0.7, 0.9} to map the interpolation.

If fused-GW at any α matches or beats Hungarian, we proceed to Phase 4b. If fused-GW also fails (which would surprise me — it has Hungarian's signal as α→1), we have a more decisive negative result and need to seriously consider Path 2 (project pivot).

---

## 12. What success looks like from here

To be honest about what an OTML workshop paper would need:

**Minimum acceptable outcome:** fused-GW matches Hungarian on Phase 4a (within noise), and *clearly beats* padded-Hungarian baselines on Phase 4b (unequal-size). Cross-model (Phase 4c) is bonus.

**Strong positive outcome:** fused-GW beats Hungarian on Phase 4a too, at some α value, and the optimal α reveals something interpretable (e.g., "α=0.7 is best, suggesting cross-side similarity dominates but structural information helps").

**Honest negative outcome:** Hungarian wins everywhere it can run, fused-GW only wins where Hungarian fails, the structural term in fused-GW doesn't add much. Paper becomes "we systematically test OT formulations for SAE feature matching; here's the regime decomposition."

All three outcomes are publishable at OTML. The first two are stronger but the third is honest research.

---

## 13. Glossary

- **Activation pattern**: how a feature fires across many tokens — a vector of length n_tokens.
- **Coupling / transport plan**: a joint distribution `P` whose row sums equal source marginal and column sums equal target marginal. The unknown of the OT problem.
- **Decoder direction**: column of `W_dec` in an SAE; the vector that gets added to the residual stream when a feature is active.
- **Dictionary size**: the wide dimension N of an SAE's feature space (e.g., 24,576).
- **Entropic regularization**: adding `-ε · H(P)` to the OT objective, which makes Sinkhorn applicable but biases the solution.
- **Fused-GW**: convex combination of OT and GW objectives; uses both cross-side and within-side cost.
- **Hungarian algorithm**: classical algorithm for solving linear assignment (matching n items to n items minimizing total cost). `scipy.optimize.linear_sum_assignment`.
- **Mech interp**: mechanistic interpretability. The subfield of ML that tries to reverse-engineer what neural networks do internally.
- **Residual stream**: the running activation vector that gets added to as a transformer processes a token; has the same dimensionality at every layer (768 for GPT-2-small).
- **SAE**: sparse autoencoder. The model that maps `h → f → h'` with f sparse and high-dim.
- **Sinkhorn algorithm**: alternating-projection method for entropic OT.
- **Sinkhorn divergence**: debiased version of entropic OT cost; `S_ε(α, α) = 0`.
- **Superposition**: hypothesis that neural networks store more concepts than they have neurons by overlapping concept directions.
- **Wasserstein distance / OT cost**: the optimal-transport-derived distance between probability measures.
- **Within-side / cross-side**: within-side = inside one of the two distributions being matched; cross-side = between the two.

---

## 14. References and tooling

### 14.1 OT theory and methods
- Peyré & Cuturi (2019), *Computational Optimal Transport*. Free PDF at https://arxiv.org/abs/1803.00567.
- Mémoli (2011), "Gromov-Wasserstein Distances and the Metric Approach to Object Matching."
- Peyré, Cuturi, Solomon (2016), "Gromov-Wasserstein Averaging of Kernel and Distance Matrices."
- **Vayer, Chapel, Flamary, Tavenard, Courty (2018), "Optimal Transport for structured data with application on graphs."** — the Fused-GW paper.
- Cuturi (2013), "Sinkhorn Distances."
- Genevay, Peyré, Cuturi (2018), "Learning Generative Models with Sinkhorn Divergences."
- Feydy et al. (2019), "Interpolating between OT and MMD using Sinkhorn Divergences."

### 14.2 SAE and mech interp
- Bricken et al. (2023), "Towards Monosemanticity." Anthropic's foundational SAE paper.
- Templeton et al. (2024), "Scaling Monosemanticity."
- Lieberum et al. (2024), "Gemma Scope." Industrial-scale SAEs from DeepMind.
- Elhage et al. (2022), "Toy Models of Superposition."

### 14.3 Tools
- POT (Python Optimal Transport): https://pythonot.github.io/
- sae_lens: https://github.com/jbloomAus/SAELens
- transformer_lens: https://github.com/TransformerLensOrg/TransformerLens

### 14.4 Hardware and resources
- Local: GTX 1650, 4GB VRAM (often shared with other workloads, so we mostly run on CPU).
- Remote: Runpod, $50 hard ceiling. Phase 4 may use up to ~$25.

---

*Last updated at the end of Phase 4a (initial attempt). Next session begins Phase 4a-bis with fused-GW.*
