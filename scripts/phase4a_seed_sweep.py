"""Multi-seed alpha sweep for fused-GW vs Hungarian on cross-layer SAE matching.

Reproduces Phase 4a-bis with three seeds of activation collection and a
finer alpha grid near the apparent optimum at alpha=0.1.

Saves results/phase4a_seed_sweep.json with per-seed and aggregated metrics.

Run from repo root:
    python scripts/phase4a_seed_sweep.py
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

# Sweep config.
SEEDS = [0, 1, 2]
ALPHA_VALUES = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
N_DOCS = 100
MAX_TOKENS_PER_DOC = 128
TOP_N = 2000
EPSILON = 5e-3
MAX_ITER = 500
TOL = 1e-7
HOOK_7 = "blocks.7.hook_resid_pre"
HOOK_8 = "blocks.8.hook_resid_pre"


def collect_activations_with_seed(
    model, sae_7, sae_8, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect SAE features on a corpus with a deterministic skip pattern.

    To get truly different feature matrices across seeds, we skip a different
    number of documents at the start of the streaming dataset. Cheap and
    reproducible.
    """
    skip = seed * N_DOCS

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    tokens_7: list[torch.Tensor] = []
    acts_7: list[torch.Tensor] = []
    tokens_8: list[torch.Tensor] = []
    acts_8: list[torch.Tensor] = []

    consumed = 0
    for item in dataset:
        if consumed < skip:
            consumed += 1
            continue
        if len(tokens_7) >= N_DOCS:
            break
        text = item["text"][:1500]
        tok = model.to_tokens(text)[:, :MAX_TOKENS_PER_DOC]
        with torch.no_grad():
            _, cache = model.run_with_cache(tok, names_filter=[HOOK_7, HOOK_8])
        tokens_7.append(tok[0])
        tokens_8.append(tok[0])
        acts_7.append(cache[HOOK_7][0])
        acts_8.append(cache[HOOK_8][0])
        del cache
        if (len(tokens_7)) % 25 == 0:
            print(f"    collected {len(tokens_7)}/{N_DOCS} docs")

    flat_acts_7 = torch.cat(acts_7)
    flat_acts_8 = torch.cat(acts_8)
    flat_tokens = torch.cat(tokens_7)

    with torch.no_grad():
        f_7 = sae_7.encode(flat_acts_7)
        f_8 = sae_8.encode(flat_acts_8)

    return f_7.cpu(), f_8.cpu(), flat_tokens.cpu(), flat_tokens.cpu()


def run_one_seed(model, sae_7, sae_8, seed: int) -> dict:
    """Run all matching methods for one seed."""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    print("Collecting activations...")
    t0 = time.time()
    F_7_full, F_8_full, _, _ = collect_activations_with_seed(model, sae_7, sae_8, seed)
    print(f"  done in {time.time() - t0:.1f}s")
    print(f"  F_7: {F_7_full.shape}, F_8: {F_8_full.shape}")

    # Top-N most-active features per SAE on this seed's corpus.
    counts_7 = (F_7_full > 0).float().sum(dim=0)
    counts_8 = (F_8_full > 0).float().sum(dim=0)
    idx_7 = torch.topk(counts_7, k=TOP_N).indices.sort().values
    idx_8 = torch.topk(counts_8, k=TOP_N).indices.sort().values

    W_dec_7 = sae_7.W_dec[idx_7].cpu()
    W_dec_8 = sae_8.W_dec[idx_8].cpu()

    F_7 = F_7_full[:, idx_7]
    F_8 = F_8_full[:, idx_8]

    n_tokens = F_7.shape[0]
    n_train = int(0.7 * n_tokens)
    F_7_train = F_7[:n_train]
    F_8_train = F_8[:n_train]
    F_7_eval = F_7[n_train:]
    F_8_eval = F_8[n_train:]

    seed_results: dict = {"seed": seed, "n_tokens": n_tokens, "n_eval": n_tokens - n_train}

    # Three baselines.
    print("Cosine-Hungarian...")
    t0 = time.time()
    m_cos = cosine_hungarian(W_dec_7, W_dec_8)
    seed_results["cosine_hungarian"] = evaluate_matching(m_cos, F_7_eval, F_8_eval)
    seed_results["cosine_hungarian"]["time_s"] = time.time() - t0
    print(f"  mean_corr={seed_results['cosine_hungarian']['mean_corr']:.4f}")

    print("Activation-Hungarian...")
    t0 = time.time()
    m_act = activation_hungarian(F_7_train, F_8_train)
    seed_results["activation_hungarian"] = evaluate_matching(m_act, F_7_eval, F_8_eval)
    seed_results["activation_hungarian"]["time_s"] = time.time() - t0
    print(f"  mean_corr={seed_results['activation_hungarian']['mean_corr']:.4f}")

    print("Pure GW...")
    t0 = time.time()
    m_gw = gw_matching(W_dec_7, W_dec_8, epsilon=EPSILON, max_iter=MAX_ITER, tol=TOL)
    seed_results["pure_gw"] = evaluate_matching(m_gw, F_7_eval, F_8_eval)
    seed_results["pure_gw"]["time_s"] = time.time() - t0
    print(f"  mean_corr={seed_results['pure_gw']['mean_corr']:.4f}")

    # Fused-GW sweep.
    print("\nFused-GW sweep...")
    fgw: dict = {}
    for alpha in ALPHA_VALUES:
        t0 = time.time()
        m_fgw = fused_gw_matching(
            W_dec_7,
            W_dec_8,
            alpha=alpha,
            epsilon=EPSILON,
            max_iter=MAX_ITER,
            tol=TOL,
        )
        elapsed = time.time() - t0
        res = evaluate_matching(m_fgw, F_7_eval, F_8_eval)
        res["time_s"] = elapsed
        res["agreement_with_cosine_hungarian"] = float((m_fgw == m_cos).mean())
        res["agreement_with_activation_hungarian"] = float((m_fgw == m_act).mean())
        fgw[str(alpha)] = res
        print(
            f"  alpha={alpha:5.2f}  mean_corr={res['mean_corr']:.4f}  "
            f"frac>0.5={res['frac_above_0p5']:.4f}  time={elapsed:.1f}s"
        )

    seed_results["fused_gw"] = fgw
    return seed_results


def aggregate_across_seeds(all_results: list[dict]) -> dict:
    """Compute mean/std of mean_corr across seeds for each method/alpha."""
    agg: dict = {"baselines": {}, "fused_gw": {}}

    for method in ["cosine_hungarian", "activation_hungarian", "pure_gw"]:
        means = [r[method]["mean_corr"] for r in all_results]
        fracs = [r[method]["frac_above_0p5"] for r in all_results]
        agg["baselines"][method] = {
            "mean_corr_mean": float(np.mean(means)),
            "mean_corr_std": float(np.std(means)),
            "frac_above_0p5_mean": float(np.mean(fracs)),
            "frac_above_0p5_std": float(np.std(fracs)),
            "per_seed": means,
        }

    for alpha in ALPHA_VALUES:
        key = str(alpha)
        means = [r["fused_gw"][key]["mean_corr"] for r in all_results]
        fracs = [r["fused_gw"][key]["frac_above_0p5"] for r in all_results]
        agg["fused_gw"][key] = {
            "mean_corr_mean": float(np.mean(means)),
            "mean_corr_std": float(np.std(means)),
            "frac_above_0p5_mean": float(np.mean(fracs)),
            "frac_above_0p5_std": float(np.std(fracs)),
            "per_seed": means,
        }

    return agg


def print_summary(agg: dict) -> None:
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS (mean ± std across 3 seeds)")
    print("=" * 80)
    print(f"{'method':>30s}  {'mean_corr':>14s}  {'frac>0.5':>14s}")
    print("-" * 80)

    for method in ["cosine_hungarian", "activation_hungarian", "pure_gw"]:
        m = agg["baselines"][method]
        print(
            f"{method:>30s}  "
            f"{m['mean_corr_mean']:.4f} ± {m['mean_corr_std']:.4f}  "
            f"{m['frac_above_0p5_mean']:.4f} ± {m['frac_above_0p5_std']:.4f}"
        )

    print("-" * 80)
    for alpha in ALPHA_VALUES:
        m = agg["fused_gw"][str(alpha)]
        label = f"fused_gw_alpha={alpha}"
        print(
            f"{label:>30s}  "
            f"{m['mean_corr_mean']:.4f} ± {m['mean_corr_std']:.4f}  "
            f"{m['frac_above_0p5_mean']:.4f} ± {m['frac_above_0p5_std']:.4f}"
        )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print("Loading model + SAEs...")
    model = HookedSAETransformer.from_pretrained("gpt2", device=DEVICE)
    sae_7 = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=HOOK_7, device=DEVICE)
    sae_8 = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=HOOK_8, device=DEVICE)

    all_results = []
    overall_t0 = time.time()
    for seed in SEEDS:
        all_results.append(run_one_seed(model, sae_7, sae_8, seed))
        elapsed_min = (time.time() - overall_t0) / 60.0
        print(f"\n[Seed {seed} complete; {elapsed_min:.1f} min total elapsed]")

    agg = aggregate_across_seeds(all_results)
    print_summary(agg)

    payload = {
        "config": {
            "seeds": SEEDS,
            "alpha_values": ALPHA_VALUES,
            "n_docs": N_DOCS,
            "max_tokens_per_doc": MAX_TOKENS_PER_DOC,
            "top_n": TOP_N,
            "epsilon": EPSILON,
            "max_iter": MAX_ITER,
            "tol": TOL,
        },
        "per_seed": all_results,
        "aggregated": agg,
    }

    out_path = OUT_DIR / "phase4a_seed_sweep.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nsaved {out_path}")
    print(f"total time: {(time.time() - overall_t0) / 60.0:.1f} min")


if __name__ == "__main__":
    main()
