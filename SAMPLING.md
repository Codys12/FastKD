# Sampling Teacher Logits

This repository contains utilities for extracting a sparse subset of logits from a large language model. The sampling procedure is implemented in `generator.py` and summarized below.

## Overview

The goal is to approximate the full softmax distribution of the teacher model while only storing a few tokens per position. We perform **importance sampling** over the vocabulary for each token in the sequence and store the sampled token ids together with their normalized probabilities.

## Steps

1. **Compute Distributions**
   - Convert logits to `float32` to avoid overflow and underflow during the softmax computation.
   - Given the raw logits for a sequence, compute the teacher distribution `p` via `softmax(logits)`.
   - Compute a proposal distribution `q` by applying temperature to the logits and then softmax: `q = softmax(logits / temperature)`.
   - Sampling temperatures in the range **0.8**–**1.2** generally give the lowest variance.
   - Replace any `NaN` or infinite values in `q` with zero and clamp negative entries to `0`.  If the resulting distribution becomes degenerate (all zeros or non‑finite) fall back to a uniform distribution before normalising.  Otherwise add a small `1e-6` to avoid zero probabilities before normalising.

2. **Draw Samples**
   - For every position in the sequence, draw `rounds` samples from `q` with replacement using `torch.multinomial`.
   - The same token may appear multiple times in the drawn set. We aggregate duplicates using `torch.unique(..., return_counts=True)`.

3. **Importance Weights**
   - For each unique sampled token, compute a weight

     ```python
     weight = count * (p[token_id] / q[token_id])
     ```
     where `count` is the number of times the token was drawn.
   - Convert the weights to `float32`, replace any non‑finite values with zero and normalise them.  If the sum of weights is zero or not finite, fall back to a uniform distribution across the sampled tokens.

4. **Store Results**
   - For every position `s` in every sequence `b`, store two parallel lists:
     - `sampled_ids[b][s]` – the token ids that were sampled.
     - `sampled_probs[b][s]` – the normalized probability of each corresponding id.
   - The number of unique tokens grows sub‑linearly with the number of rounds.  Fifty rounds typically yield around twelve unique tokens per position.

The function `sample_distribution` in `generator.py` implements the process above and returns the `sampled_ids` and `sampled_probs` structures. These results can then be serialized and used for knowledge distillation.

## Example

```python
from generator import sample_distribution

logits = model(input_ids)                  # [batch, seq, vocab]
ids, probs = sample_distribution(logits, rounds=50, temperature=1.0)
```

Each element of `ids[b][s]` matches a probability in `probs[b][s]`. This pair represents an unbiased approximation of the teacher distribution for token `s` in sequence `b`.

