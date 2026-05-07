# 07 — Phase 3: SAE Crash Course

This was the chapter where the project's two halves — the OT side and the interpretability side — met for the first time. We loaded a real SAE, ran it on real text, and looked at what individual features represented.

The technical material was new for me; the goal of Phase 3 was operational fluency. By the end, we needed to be able to load an SAE, get activations, and identify what each feature responded to. That's table stakes for Phase 4, where we'd be matching features across two SAEs.

## What we set up

We used `sae_lens`, the standard library for working with pretrained SAEs. It interfaces with `transformer_lens`, which gives clean access to activations inside transformer models.

Concretely we loaded:
- **GPT-2-small.** The 124M-parameter version of GPT-2. Tiny by modern standards — runs comfortably on CPU.
- **A pretrained SAE on layer 8 of GPT-2-small.** From the `gpt2-small-res-jb` release by Joseph Bloom. Trained on the residual stream activations at layer 8 (mid-network). Dictionary size: 24,576 features over a 768-dim residual stream — about 32× expansion factor.

The SAE is hooked into the model: when GPT-2-small processes a token, the residual-stream activation at layer 8 gets passed through the SAE's encoder, producing a 24576-dim sparse feature vector.

## What we ran

We took the model, ran it on a corpus of 30 text documents (roughly 6,000 tokens total), and at each token we collected:
- The 768-dim residual stream activation at layer 8.
- The 24576-dim SAE feature activation produced by the SAE encoder.

This gave us a feature matrix `F` of shape (n_tokens, 24576). Row i is the feature activation at token i. Column j tells us, across all tokens, how strongly feature j fired and where.

## Sparsity, observed

We measured how sparse the feature activations actually are. On our 6,000-token corpus, the average token activated about 67 features out of 24,576 — about 0.27% sparsity.

This is consistent with what's reported in the SAE literature: sparsities of 0.1% to 1%, depending on training hyperparameters. The L1 penalty during training pushes toward this sparse regime, and what you get out is a representation where *most features are silent for most tokens*, but the small handful that *do* fire are (hopefully) interpretable.

## Looking at individual features — the messy picture

This is where it got interesting.

We picked the 10 features that fired most frequently across the corpus and printed, for each, the top-10 tokens that activated it most strongly along with surrounding context. These are the candidate "interpretable" features.

What we found, honestly:

- Most "top-firing" features were *not* clearly interpretable. They fired on common short words like "and", "the", "to" — function words that don't carry semantic content. The SAE has dedicated some features to encoding "this is a common function word" or "this is a punctuation token" — useful for the model, not useful for our interpretation goals.

- Several features were "single-document features" — they fired heavily on tokens from one particular document in the corpus, and we couldn't characterize what concept they encoded without more data. With only 30 documents, you don't have enough examples.

- Some features had mixed firing patterns: tokens that don't seem related to each other. Either the feature really is polysemantic (encoding multiple unrelated things), or the corpus is too small to see the unifying pattern.

This was a useful lesson on its own: characterizing SAE features requires *much more data than 30 documents*. Real interpretability work uses 50,000+ tokens per feature, sometimes millions, to get clean characterizations.

## The "document boundary" finding (the clean one)

After filtering out the muddiest cases, we found a striking cluster of features that fired *exclusively* on the special token `<|endoftext|>`. This is GPT-2's document-separator token, automatically inserted between training documents.

These features had several distinctive properties:

- **Activation magnitudes 50-100× larger than typical concept features.** Where a normal feature might fire at strength 4-7, these fired at strength 100-500 on `<|endoftext|>` tokens.
- **Never fired on anything else.** Every example of high activation was on a `<|endoftext|>` token.
- **Multiple features in this cluster.** We found 8 different feature indices that all behaved this way, and they all co-activated on the same tokens at similar strengths.

The interpretation is clean: the SAE has dedicated multiple features to detecting "we are at a document boundary." The model needs to handle these tokens specially (it's about to process an unrelated new document, so any "context" it's been carrying needs to reset), and the SAE has captured this with multiple high-magnitude features.

The fact that *multiple* features encode the same concept is a real phenomenon called **feature splitting** — described in Bricken et al. 2023 ("Towards Monosemanticity"). When the SAE dictionary is much larger than the number of distinct concepts in the training data, the SAE has "extra capacity" and uses it by encoding common concepts redundantly across multiple features.

## The Phase 3 deliverable, in one sentence

After Phase 3, I could:

- Load a pretrained SAE on top of GPT-2-small.
- Run it on text and extract per-token feature activations.
- For any feature index, look at top-activating tokens and form an opinion about what concept (if any) the feature encodes.
- Recognize feature splitting when I see it.

That's enough operational fluency to take into Phase 4, where we'd be doing matching between *pairs* of SAEs and trying to evaluate whether the matchings are "good."

## What I underestimated about characterization

The key thing I learned, that turned out to matter for Phase 4: characterizing SAE features cleanly is *expensive*. You need a lot of data per feature to be confident about what it encodes, you need to filter out the trivial features, and you need patience.

This shaped Phase 4's evaluation strategy. Instead of asking "are the matched feature pairs *interpretable* in similar ways" — which would require running auto-interpretation on each feature, expensive — we used a cheaper proxy: *do matched features fire on similar tokens on a held-out corpus?* Specifically, we measured the Pearson correlation between matched features' activation patterns over evaluation tokens.

This is a cheaper, more automated evaluation. It says: if we matched feature i in SAE A to feature j in SAE B, and on the held-out corpus they fire on similar tokens with similar strengths, that's evidence the matching is meaningful. If they fire on completely different tokens, the matching is probably random.

This is the metric that drove every subsequent phase. The math is simple: for each (i, π(i)) pair in the matching, compute the Pearson correlation of (F_A column i) with (F_B column π(i)) on the held-out token set. Report the mean and quartiles over all matched pairs.

A correlation of 1.0 means the two features fire identically on the same tokens. A correlation of 0 means they're unrelated. Correlations above 0.5 indicate "real, meaningful matching." Around 0.3 means "some signal, mostly noise." Below 0.1 is essentially chance.

The whole project's experimental story is told in this metric. Coming up: [Phase 4a](08_phase_4a.md), where we got our first real numbers.

Next: [Phase 4a](08_phase_4a.md).
