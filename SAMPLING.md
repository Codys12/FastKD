# Sampling Teacher Logits

This repository contains utilities for extracting a sparse subset of logits from a large language model. The sampling procedure is implemented in `generator.py` and summarized below.

## Overview

The goal is to approximate the full softmax distribution of the teacher model while only storing a few tokens per position. We perform **importance sampling** over the vocabulary for each token in the sequence and store the sampled token ids together with their normalized probabilities.

## Steps

1. **Compute Distributions**
   - Given the raw logits for a sequence, compute the teacher distribution `p` via `softmax(logits)`.
   - Compute a proposal distribution `q` by applying temperature to the logits and then softmax: `q = softmax(logits / temperature)`.

2. **Draw Samples**
   - For every position in the sequence, draw `rounds` samples from `q` with replacement using `torch.multinomial`.
   - The same token may appear multiple times in the drawn set. We aggregate duplicates using `torch.unique(..., return_counts=True)`.

3. **Importance Weights**
   - For each unique sampled token, compute a weight
     
     ```python
     weight = count * (p[token_id] / q[token_id])
     ```
     where `count` is the number of times the token was drawn.
   - Normalize all weights so that they sum to `1.0`. These normalized weights become the probabilities associated with the sampled tokens.

4. **Store Results**
   - For every position `s` in every sequence `b`, store two parallel lists:
     - `sampled_ids[b][s]` – the token ids that were sampled.
     - `sampled_probs[b][s]` – the normalized probability of each corresponding id.
   - On average, using 50 sampling rounds yields roughly twelve unique tokens per position.

The function `sample_distribution` in `generator.py` implements the process above and returns the `sampled_ids` and `sampled_probs` structures. These results can then be serialized and used for knowledge distillation.

## Example

```python
from generator import sample_distribution

logits = model(input_ids)                  # [batch, seq, vocab]
ids, probs = sample_distribution(logits, rounds=50, temperature=1.0)
```

Each element of `ids[b][s]` matches a probability in `probs[b][s]`. This pair represents an unbiased approximation of the teacher distribution for token `s` in sequence `b`.

