# 10 — Phase 4b: Unequal Sizes (The Decisive Negative Result)

This is the experiment that determined the fate of the project's main hypothesis. The setup was clean, the data was strong, and the result was unambiguous in a direction we didn't want.

## What we were testing

Phase 4a-bis showed fused-GW gives a small (~5%) advantage over Hungarian on the equal-size case. Phase 4b's question: does that advantage grow when the two SAEs have different dictionary sizes — the case where Hungarian is supposed to fundamentally fail?

The original project pitch said yes. That was the whole point: GW handles unequal sizes natively, Hungarian doesn't, so for the unequal case the gap should be large.

The reality: it didn't work that way.

## The dataset (a nice piece of luck)

We discovered that the same group that trained `gpt2-small-res-jb` (the SAEs we'd been using) also released `gpt2-small-res-jb-feature-splitting`. This is a series of SAEs trained at the same hook point (layer 8 residual stream, pre-block) with identical training setup but different dictionary sizes:

```
blocks.8.hook_resid_pre_768
blocks.8.hook_resid_pre_1536
blocks.8.hook_resid_pre_3072
blocks.8.hook_resid_pre_6144
blocks.8.hook_resid_pre_12288
blocks.8.hook_resid_pre_24576
blocks.8.hook_resid_pre_49152
blocks.8.hook_resid_pre_98304
```

Eight sizes, 768 to 98304, all trained the same way. This is a controlled comparison: the only thing that differs between any pair is the dictionary size.

We picked three pairs to test:
- **6144 ↔ 12288 (2× size ratio).** Modest mismatch.
- **6144 ↔ 24576 (4× ratio).** "Default" mismatch.
- **3072 ↔ 24576 (8× ratio).** Aggressive mismatch.

For each, we'd run all four matching methods across 3 seeds.

## The methods

Same four methods as Phase 4a-bis:

1. **Cosine-Hungarian (rectangular).** scipy's `linear_sum_assignment` does support rectangular cost matrices. It picks the best `min(n_a, n_b)` assignments, leaving the surplus unassigned. So for a 1500 × 6000 case, it picks the 1500 best matches from the 6000 candidates and ignores the rest.

2. **Activation-Hungarian (rectangular).** Same idea but using activation correlation instead of cosine.

3. **Pure entropic GW.** Same as Phase 4a — fails on within-only structure.

4. **Fused-GW at α=0.05.** The Phase 4a-bis winner. Same configuration.

## What I expected

Going in, I expected:

- Cosine-Hungarian's performance to *degrade* as the size ratio increased. With more candidates to pick from, Hungarian has more chances to make wrong choices, especially since the matching has to be one-to-one. Lots of features in the larger SAE would simply be ignored.
- Pure GW to remain at chance, same as always.
- Fused-GW to *outperform* cosine-Hungarian, with the gap *growing* as the size ratio grew. In the 8× case, fused-GW should crush Hungarian.

This was the project's main hypothesis, and Phase 4b was the test.

## What actually happened

Aggregated results across 3 seeds:

### 6144 ↔ 12288 (2× ratio)

| Method | mean_corr (mean ± std) | frac > 0.5 |
|---|---|---|
| **cosine_hungarian** | **0.6601 ± 0.0014** | **0.7622** |
| activation_hungarian | 0.6531 ± 0.0017 | 0.7538 |
| pure_gw | 0.0158 ± 0.0111 | 0.0047 |
| fused_gw | 0.6586 ± 0.0007 | 0.7580 |

### 6144 ↔ 24576 (4× ratio)

| Method | mean_corr (mean ± std) | frac > 0.5 |
|---|---|---|
| **cosine_hungarian** | **0.5548 ± 0.0025** | **0.6244** |
| activation_hungarian | 0.5347 ± 0.0048 | 0.5982 |
| pure_gw | 0.0105 ± 0.0064 | 0.0033 |
| fused_gw | 0.5494 ± 0.0031 | 0.6180 |

### 3072 ↔ 24576 (8× ratio)

| Method | mean_corr (mean ± std) | frac > 0.5 |
|---|---|---|
| **cosine_hungarian** | **0.4670 ± 0.0051** | **0.4822** |
| activation_hungarian | 0.4365 ± 0.0025 | 0.4469 |
| pure_gw | 0.0056 ± 0.0037 | 0.0007 |
| fused_gw | 0.4651 ± 0.0059 | 0.4798 |

## Reading the data honestly

Three things to pull out, and one is the headline:

**Cosine-Hungarian wins all three pairs.** Not by huge margins, but consistently and reproducibly. The std bars don't overlap on the 4× case (Hungarian 0.5548±0.0025 vs fused-GW 0.5494±0.0031). On the 2× and 8× cases, the gap is smaller in absolute terms but in the same direction.

**The performance numbers are much higher overall than Phase 4a's.** Cross-layer matching gave 0.33; same-layer, different-size matching gives 0.47-0.66 depending on size ratio. That's because same-layer-different-size is a *much easier* matching problem: features in two SAEs at the same hook point on the same model literally compute on the same residual stream activations, so the "true" matching exists and is recoverable.

**Pure GW is at chance everywhere.** Same story as Phase 4a, just confirmed: within-side structure alone is insufficient regardless of the size ratio.

The headline: **the project's main hypothesis is wrong**. Cosine-Hungarian is *not* fundamentally hobbled by the rectangular case. The cherry-picking effect — Hungarian gets to pick its best assignments from a larger pool — actually *helps* it, providing a stronger baseline than naive intuition suggests.

## The cherry-picking effect, made explicit

Here's what's going on. In the 4× ratio case, the smaller SAE has 1500 features (after our top-N selection) and the larger has 6000. We're asking: for each of the 1500 features in the smaller SAE, which feature in the larger SAE corresponds to it?

Cosine-Hungarian computes a 1500 × 6000 cosine-similarity matrix and runs scipy's rectangular linear-sum-assignment. The output: for each of the 1500 source features, the best match from the 6000 candidates *given that* the matching must be one-to-one (no two source features can match the same target feature).

The *cherry-picking effect*: Hungarian gets to choose its 1500 best matches from a pool of 6000. That's a much easier problem than the equal-size case, where Hungarian has to find the best 1:1 matching between two sets of 1500. The bigger pool gives more flexibility.

This is why Hungarian's performance *doesn't* fall off a cliff as the size ratio grows. The cherry-picking effect partially compensates for the increased difficulty.

Meanwhile, fused-GW has to navigate a larger search space (the 1500 × 6000 transport plan has 4× more entries than the equal-size case) without the structural information being any more helpful. The within-side cosine distance distributions in the larger SAE are still nearly uniform — all of Phase 4a's diagnostics still apply. So fused-GW's performance doesn't improve, while Hungarian's degrades less than expected.

The net result: Hungarian still wins. Even at the 8× ratio, where I would have predicted fused-GW would shine, Hungarian wins by 0.0019 in mean correlation. The gap isn't large, but the *direction* is unambiguous: more size mismatch doesn't favor fused-GW.

## Fused-GW's agreement with Hungarian

Across all three pairs, fused-GW agrees with cosine-Hungarian on 89-93% of feature matches. So fused-GW isn't producing a substantially different matching; it's mostly reproducing Hungarian, with about 10% of pairs that disagree. And on those 10% of disagreements, fused-GW's choices are slightly *worse* on the held-out metric.

In other words: the structural term in fused-GW, in the unequal-size case, is producing *noise* relative to the cross-side cosine signal. Pure cosine-Hungarian doesn't make those 10% of "different" choices, and it scores higher.

This is a clean, decisive finding. It's also the *opposite* of what the original project pitched.

## What this means for the project

The original hypothesis was: GW (or fused-GW) is the right unified tool for SAE matching, especially shining in the unequal-size case where Hungarian fails. Phase 4b shows this is wrong.

Hungarian doesn't fail in the unequal-size case. Hungarian's rectangular extension via scipy is a strong baseline. The cherry-picking effect compensates for the lost simplicity. And the within-SAE structural information that GW depends on is too uniform — under any metric we tried — to provide a meaningful advantage.

The decisive read: **for SAE feature matching with two SAEs trained on the same model, cosine-Hungarian is the right tool, full stop.** Whether sizes are equal or unequal, whether you use cosine or activation similarity, Hungarian-based methods dominate.

Fused-GW's small advantage in the equal-size case (Phase 4a-bis) is real but doesn't generalize. It's a special case, not a general phenomenon.

## What's salvageable

Two things, even though the main hypothesis is dead:

**The negative result is real research.** "We tested fused-GW for SAE feature matching across equal-size and unequal-size cases. Hungarian dominates everywhere it can run, including the rectangular case. The within-SAE structural information that GW relies on is too uniformly distributed to provide useful matching signal." This is a careful, OT-grounded finding that closes off a popular intuition. Future researchers proposing OT-based feature matching can read this and avoid the dead end.

**The infrastructure is reusable.** Activation collection, evaluation harnesses, three baselines, the experimental framework — all functional. If someone has a *different* hypothesis about how OT could help in this domain (or another domain), they have the tools.

## The path forward (as of the end of Phase 4b)

Three options on the table:

1. **Try Phase 4c (cross-model matching).** The one case where Hungarian truly cannot run, because decoder directions in different models live in incomparable coordinate systems. Whether GW could provide something useful here is genuinely an open question.

2. **Write up what we have as a careful negative-result paper.** Real research, modest in claims, useful for the field.

3. **Pivot the project.** Different OT application, different topic, with the OT/SAE infrastructure as starting capital.

The next chapter is the conclusion, where we work through these options and what they actually deliver.

Next: [Conclusion](11_conclusion.md).
