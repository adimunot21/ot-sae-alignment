# 03 — Background: What is a Sparse Autoencoder?

This chapter is the ML / interpretability side. If you're a hardware engineer, the analogies should help; if you've used PyTorch but never read mech interp papers, the technical content is basic enough to skim.

## The motivating question

Modern language models — GPT-2, GPT-3, Claude, Llama — are big neural networks. Inside one of these models, intermediate computations produce *activation vectors* — vectors of numbers (size 768 for GPT-2-small, 4096 for Llama-3-8B, etc.) that flow through the layers.

If we want to understand what the model is doing, the natural question is: what does each of those numbers mean?

If component 437 of the layer-8 activation vector lights up, what does that correspond to? "The current token is a Python keyword"? "We're in a paragraph about cooking"? "The next word is likely capitalized"? Something else entirely?

For a long time, the field assumed individual neurons in neural nets carried interpretable meaning. That is mostly false. When people poked at individual neurons in trained models, they found neurons that fire on *multiple unrelated things* — the same neuron lighting up for "Python keyword" and "the current paragraph mentions a famous person." This is called **polysemanticity**, and it makes interpreting models from raw neurons very hard.

## The superposition hypothesis

Around 2022, Anthropic published "Toy Models of Superposition" (Elhage et al.) which proposed an explanation: neural networks store more concepts than they have neurons. The way they pull this off is by representing concepts as *directions* in the activation vector space, and many concepts share components.

If I have a 768-dimensional activation vector, naively I can store 768 orthogonal concepts (one per neuron). But if concepts can share components, I can fit far more. The cost is that any individual neuron is "polysemantic" — it carries pieces of multiple concepts at once.

The superposition picture predicts that *individual neurons* won't be interpretable, but *linear directions* in activation space might be.

## Hardware-engineering analogy

This is structurally identical to what you'd see in an ASIC where engineers had to fit a function into limited gates. They'd merge logic, share LUTs, and the resulting netlist wouldn't have any one LUT cleanly mapping to one function — but the *function itself* still exists, distributed across the design. To understand what the chip does, you don't read individual LUT outputs; you'd reverse-engineer the dataflow and find which combinations of LUTs implement which subfunctions.

Same problem here. The "function" (concept) exists; it's just spread across activation components.

## What an SAE does

A sparse autoencoder is a tool for *un-mixing* the superposed representation.

It's a small neural network you train alongside (but separately from) the main model. It takes the model's activation vector `h ∈ ℝᵈ` and:

1. **Encodes** it into a much wider, *sparse* representation:
   ```
   f = ReLU(W_enc · h + b_enc)
   ```
   where `f ∈ ℝᴺ` with `N >> d`. Typical sizes: d=768, N=24,576 (32× expansion).

2. **Decodes** back:
   ```
   h' = W_dec · f
   ```
   trained so that `h' ≈ h`.

3. The training penalizes `f` for having many non-zero entries — typically with an L1 penalty. So at inference time, only a small handful of the N entries of f are non-zero (typically 50-200 out of 24,000).

The columns of W_dec — the *decoder directions* — are vectors in the original 768-dim space. Each column corresponds to one "feature." When that feature's entry of f is nonzero, the decoder adds a scaled version of that direction to reconstruct h.

## Why this works (when it works)

If the model genuinely stores concepts as directions in superposition, and there are more concepts than dimensions, then a wider sparse expansion *decompresses* them. Each feature in the SAE corresponds to one of the concepts the model has stored — at least, that's the hope.

Empirically this mostly works on language models. Trained SAE features often correspond to interpretable patterns:
- "Python keyword token"
- "We're in a paragraph about a specific city"
- "Past tense verb"
- "Current token is at a sentence boundary"
- and so on, including features for very specific things like "Disney character names" or "this is a code comment"

It's not perfect. Some features don't correspond to anything humans recognize. Some "concepts" get split into multiple sub-features. Some never fire. But the overall picture works.

## What we did in Phase 3

We loaded a real SAE on top of GPT-2-small using the `sae_lens` library, ran it on some text, and looked at what each feature responded to. We found:

- The SAE has 24,576 features for GPT-2-small layer 8.
- Most tokens activate ~50-100 features (out of 24,576). Confirms the sparsity.
- Some features have a clear interpretation. We found a "document boundary detector" family — a cluster of features that all fire only on the special token `<|endoftext|>` that GPT-2 puts between training documents. These features had activation magnitudes 50-100× larger than typical concept features, and they all co-activated on the same tokens, illustrating *feature splitting* (where the SAE redundantly encodes a single concept across multiple features when the dictionary is bigger than needed).
- Most features didn't have an obvious interpretation from a 30-document corpus. To really characterize features you need 50,000+ tokens; we didn't bother because the project's main contribution wasn't about interpretability per se.

This was the operational fluency baseline: by end of Phase 3, we could load a real SAE, extract activations, and find what individual features represent. Enough to support Phase 4.

## Why match SAE features at all?

This is the question Phase 4 addresses. Several reasons:

1. **Universality.** If you train two SAEs with different random seeds on the same model, do they learn the same features? Researchers want to know. To check, you need to *match* features between the two SAEs.

2. **Cross-model transfer.** If you painstakingly characterize features in GPT-2-small ("feature 437 is the Python keyword detector"), can you find the equivalent feature in GPT-2-medium without re-doing all the work? Matching across models would let you transfer the labels.

3. **Multi-size studies.** As you scale up SAE dictionary size, do features just split (the same concepts spread across more sub-features) or do new concepts appear? You need to match across sizes to answer.

4. **Comparison and ablation.** Lots of ML research wants to compare two trained models and see what's different. SAEs give you a structured way to compare; matching gives you the alignment.

The current state of the art for matching is **the Hungarian algorithm on cosine similarity**. You compute the cosine similarity between each pair of decoder directions (one from each SAE), and use the Hungarian algorithm to find the optimal assignment. This is `scipy.optimize.linear_sum_assignment`, two lines of code.

This works when both SAEs have the same dictionary size and are trained on the same model. The decoder directions live in the same residual-stream coordinate system, so cosine similarity is meaningful.

It fails (or doesn't even apply) when:
- **Sizes differ.** Hungarian wants square cost matrices.
- **Models differ.** GPT-2-small (768-dim residual stream) vs GPT-2-medium (1024-dim) — cosine similarity isn't defined across different dimensions.

The original project hypothesis was: GW is the right tool for the cases where Hungarian fails. GW handles incomparable spaces natively. It should give us a unified matching method that works in all settings.

This is what we're going to test.

Next chapter: the precise research question and what we hoped to show.
