"""Phase 4b: matching across SAEs with unequal dictionary sizes.

Same model (GPT-2-small), same hook point (blocks.8.hook_resid_pre),
same training (`gpt2-small-res-jb-feature-splitting` release), different
dictionary sizes. Three size pairs, three seeds each.

Compares:
1. Cosine-Hungarian (rectangular). The "incumbent" extended to unequal sizes
   via scipy's rectangular linear_sum_assignment.
2. Activation-Hungarian (rectangular).
3. Pure entropic GW.
4. Fused-GW at the alpha that won Phase 4a (alpha=0.05).

Saves results/phase4b_unequal_size.json.

Run from repo root:
    python scripts/phase4b_unequal_size.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer

from ot_sae.evaluation import evaluate_matching
from ot_sae.matching import (
    activation_hungarian,
    cosine_hungarian,
    fused_gw_matching,
    gw_matching,
)

OUT_DIR = Path(__file__).parent.parent / "results"
DEVICE = "cpu"

# Sweep config. Keep TOP_N small enough that the larger SAE's all-features case
# is tractable, AND so that the smaller SAE's TOP_N doesn't exceed its dict size.
SEEDS = [0, 1, 2]

# (smaller_size, larger_size, top_n_smaller, top_n_larger)
# top_n is "use these many most-active features per side". When top_n is the
# size of the smaller SAE, we use ALL its features.
SIZE_PAIRS = [
    (6144, 12288, 1500, 3000),  # 2x ratio
    (6144, 24576, 1500, 6000),  # 4x ratio
    (3072, 24576, 1500, 12000),  # 8x ratio
]
N_DOCS = 100
MAX_TOKENS_PER_DOC = 128
EPSILON = 5e-3
MAX_ITER = 500
TOL = 1e-7
ALPHA_FUSED = 0.05  # winner from Phase 4a-bis
RELEASE = "gpt2-small-res-jb-feature-splitting"
HOOK_BASE = "blocks.8.hook_resid_pre"


def collect_dual_features(model, sae_a, sae_b, hook_name: str, seed: int) -> tuple:
    """Run model once, encode through both SAEs."""
    skip = seed * N_DOCS

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    acts: list[torch.Tensor] = []
    consumed = 0
    for item in dataset:
        if consumed < skip:
            consumed += 1
            continue
        if len(acts) >= N_DOCS:
            break
        text = item["text"][:1500]
        tok = model.to_tokens(text)[:, :MAX_TOKENS_PER_DOC]
        with torch.no_grad():
            _, cache = model.run_with_cache(tok, names_filter=[hook_name])
        acts.append(cache[hook_name][0])
        del cache
        if len(acts) % 25 == 0:
            print(f"      {len(acts)}/{N_DOCS} docs")

    flat_acts = torch.cat(acts)
    with torch.no_grad():
        f_a = sae_a.encode(flat_acts).cpu()
        f_b = sae_b.encode(flat_acts).cpu()
    return f_a, f_b


def run_one_pair(
    model, size_smaller: int, size_larger: int, top_n_smaller: int, top_n_larger: int
) -> dict:
    """Run all 4 methods across SEEDS for one size-pair."""
    print(f"\n{'='*70}")
    print(f"PAIR: {size_smaller} (top {top_n_smaller}) <-> {size_larger} (top {top_n_larger})")
    print(f"{'='*70}")

    print("Loading SAEs...")
    sae_smaller = SAE.from_pretrained(
        release=RELEASE,
        sae_id=f"{HOOK_BASE}_{size_smaller}",
        device=DEVICE,
    )
    sae_larger = SAE.from_pretrained(
        release=RELEASE,
        sae_id=f"{HOOK_BASE}_{size_larger}",
        device=DEVICE,
    )

    seed_results = []
    for seed in SEEDS:
        print(f"\n  seed {seed}: collecting...")
        t0 = time.time()
        F_smaller_full, F_larger_full = collect_dual_features(
            model, sae_smaller, sae_larger, HOOK_BASE, seed
        )
        print(f"    done in {time.time() - t0:.1f}s")

        # Top-N most-active features in each.
        counts_s = (F_smaller_full > 0).float().sum(dim=0)
        counts_l = (F_larger_full > 0).float().sum(dim=0)
        idx_s = torch.topk(counts_s, k=top_n_smaller).indices.sort().values
        idx_l = torch.topk(counts_l, k=top_n_larger).indices.sort().values

        W_dec_s = sae_smaller.W_dec[idx_s].cpu()
        W_dec_l = sae_larger.W_dec[idx_l].cpu()
        F_s = F_smaller_full[:, idx_s]
        F_l = F_larger_full[:, idx_l]

        n_tokens = F_s.shape[0]
        n_train = int(0.7 * n_tokens)
        F_s_train = F_s[:n_train]
        F_l_train = F_l[:n_train]
        F_s_eval = F_s[n_train:]
        F_l_eval = F_l[n_train:]

        seed_data = {"seed": seed}

        print("    cosine-Hungarian (rectangular)...")
        t0 = time.time()
        m_cos = cosine_hungarian(W_dec_s, W_dec_l)
        seed_data["cosine_hungarian"] = evaluate_matching(m_cos, F_s_eval, F_l_eval)
        seed_data["cosine_hungarian"]["time_s"] = time.time() - t0
        print(f"      mean_corr={seed_data['cosine_hungarian']['mean_corr']:.4f}")

        print("    activation-Hungarian (rectangular)...")
        t0 = time.time()
        m_act = activation_hungarian(F_s_train, F_l_train)
        seed_data["activation_hungarian"] = evaluate_matching(m_act, F_s_eval, F_l_eval)
        seed_data["activation_hungarian"]["time_s"] = time.time() - t0
        print(f"      mean_corr={seed_data['activation_hungarian']['mean_corr']:.4f}")

        print("    pure GW...")
        t0 = time.time()
        m_gw = gw_matching(W_dec_s, W_dec_l, epsilon=EPSILON, max_iter=MAX_ITER, tol=TOL)
        seed_data["pure_gw"] = evaluate_matching(m_gw, F_s_eval, F_l_eval)
        seed_data["pure_gw"]["time_s"] = time.time() - t0
        print(f"      mean_corr={seed_data['pure_gw']['mean_corr']:.4f}")

        print(f"    fused-GW (alpha={ALPHA_FUSED})...")
        t0 = time.time()
        m_fgw = fused_gw_matching(
            W_dec_s,
            W_dec_l,
            alpha=ALPHA_FUSED,
            epsilon=EPSILON,
            max_iter=MAX_ITER,
            tol=TOL,
        )
        seed_data["fused_gw"] = evaluate_matching(m_fgw, F_s_eval, F_l_eval)
        seed_data["fused_gw"]["time_s"] = time.time() - t0
        seed_data["fused_gw"]["agree_with_cosine"] = float((m_fgw == m_cos).mean())
        print(
            f"      mean_corr={seed_data['fused_gw']['mean_corr']:.4f}  "
            f"agree_w_cos={seed_data['fused_gw']['agree_with_cosine']:.4f}"
        )

        seed_results.append(seed_data)

    # Aggregate.
    aggregated = {}
    for method in ["cosine_hungarian", "activation_hungarian", "pure_gw", "fused_gw"]:
        means = [r[method]["mean_corr"] for r in seed_results]
        fracs = [r[method]["frac_above_0p5"] for r in seed_results]
        aggregated[method] = {
            "mean_corr_mean": float(np.mean(means)),
            "mean_corr_std": float(np.std(means)),
            "frac_above_0p5_mean": float(np.mean(fracs)),
            "frac_above_0p5_std": float(np.std(fracs)),
            "per_seed": means,
        }

    return {
        "size_smaller": size_smaller,
        "size_larger": size_larger,
        "ratio": size_larger / size_smaller,
        "top_n_smaller": top_n_smaller,
        "top_n_larger": top_n_larger,
        "per_seed": seed_results,
        "aggregated": aggregated,
    }


def print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS — Phase 4b (mean ± std across 3 seeds)")
    print("=" * 80)
    for pair_result in all_results:
        s, n_l = pair_result["size_smaller"], pair_result["size_larger"]
        ratio = pair_result["ratio"]
        agg = pair_result["aggregated"]
        print(f"\n{s} <-> {n_l}  (ratio {ratio:.0f}x):")
        print(f"  {'method':>22s}  {'mean_corr':>14s}  {'frac>0.5':>14s}")
        for method in ["cosine_hungarian", "activation_hungarian", "pure_gw", "fused_gw"]:
            m = agg[method]
            print(
                f"  {method:>22s}  "
                f"{m['mean_corr_mean']:.4f} ± {m['mean_corr_std']:.4f}  "
                f"{m['frac_above_0p5_mean']:.4f} ± {m['frac_above_0p5_std']:.4f}"
            )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model...")
    model = HookedSAETransformer.from_pretrained("gpt2", device=DEVICE)

    all_results = []
    overall_t0 = time.time()
    for size_s, size_l, top_s, top_l in SIZE_PAIRS:
        all_results.append(run_one_pair(model, size_s, size_l, top_s, top_l))
        elapsed_min = (time.time() - overall_t0) / 60.0
        print(f"\n[Pair complete; {elapsed_min:.1f} min total]")

    print_summary(all_results)

    payload = {
        "config": {
            "release": RELEASE,
            "hook_base": HOOK_BASE,
            "seeds": SEEDS,
            "size_pairs": SIZE_PAIRS,
            "n_docs": N_DOCS,
            "max_tokens_per_doc": MAX_TOKENS_PER_DOC,
            "epsilon": EPSILON,
            "max_iter": MAX_ITER,
            "tol": TOL,
            "alpha_fused": ALPHA_FUSED,
        },
        "results": all_results,
    }
    out_path = OUT_DIR / "phase4b_unequal_size.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved {out_path}")
    print(f"total time: {(time.time() - overall_t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
