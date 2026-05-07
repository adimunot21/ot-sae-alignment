"""Activation collection: run a model + SAE on text, return feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from datasets import load_dataset


@dataclass
class FeatureCollection:
    """SAE feature activations on a corpus.

    Attributes
    ----------
    features : torch.Tensor
        Shape (n_tokens, d_sae). Per-token feature activations from the SAE.
    tokens : torch.Tensor
        Shape (n_tokens,). The token IDs that produced these activations.
    hook_name : str
        The hook point (e.g. 'blocks.8.hook_resid_pre').
    d_sae : int
        Dictionary size.
    """

    features: torch.Tensor
    tokens: torch.Tensor
    hook_name: str
    d_sae: int


def collect_sae_features(
    model,
    sae,
    hook_name: str,
    n_docs: int = 100,
    max_tokens_per_doc: int = 128,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    progress_every: int = 25,
) -> FeatureCollection:
    """Run model on a streaming dataset, encode through SAE, collect features.

    Parameters
    ----------
    model
        A HookedSAETransformer or HookedTransformer.
    sae
        A loaded SAE.
    hook_name
        The hook point to extract activations from.
    n_docs
        Number of documents to process.
    max_tokens_per_doc
        Cap per-document tokens to keep memory bounded.
    dataset_name, dataset_config
        Hugging Face dataset to stream.
    progress_every
        Print progress every N documents.
    """
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )

    all_tokens: list[torch.Tensor] = []
    all_acts: list[torch.Tensor] = []

    for i, item in enumerate(dataset):
        if i >= n_docs:
            break
        text = item["text"][:1500]
        tokens = model.to_tokens(text)[:, :max_tokens_per_doc]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        all_tokens.append(tokens[0])
        all_acts.append(cache[hook_name][0])
        del cache
        if (i + 1) % progress_every == 0:
            print(f"  processed {i + 1}/{n_docs} docs")

    flat_tokens = torch.cat(all_tokens)
    flat_acts = torch.cat(all_acts)

    with torch.no_grad():
        sae_features = sae.encode(flat_acts)

    d_sae = sae_features.shape[1]
    return FeatureCollection(
        features=sae_features.cpu(),
        tokens=flat_tokens.cpu(),
        hook_name=hook_name,
        d_sae=d_sae,
    )


def top_active_features(fc: FeatureCollection, top_k: int = 2000) -> torch.Tensor:
    """Return indices of the top_k most-frequently-firing features.

    Restricting to active features keeps matching tractable and excludes
    dead features that never fire on real data.
    """
    firing_counts = (fc.features > 0).float().sum(dim=0)
    return torch.topk(firing_counts, k=top_k).indices.sort().values
