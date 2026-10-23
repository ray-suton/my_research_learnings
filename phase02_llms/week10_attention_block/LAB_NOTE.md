# Week 10 Lab Note - A Hand-Written Transformer Block

**Date:** 2026-10-05
**Scope:** Week 10 Sessions 1-5. Correctness and cost only; no training was run.
**Status:** Closed. Both Week 10 deliverables - the tested block and the prepared
LM dataset - are implemented and their fixture suites are green.

## One-Paragraph Summary

Week 10 built the core Transformer block from scratch in PyTorch: numerically
stable softmax, additive causal and padding masks, scaled dot-product attention,
multi-head attention with a manual per-head reference, and finally a pre-norm
residual block containing hand-written layer normalization and a GELU MLP. The
week also closed the prepared-LM-dataset carry-over: per-split token streams with
one EOS per nonempty record, `T + 1` windows cut at stride `T`, one-shifted
targets, and separate attention and loss masks. Every claim below is backed by a
fixture suite that was re-run on 2026-10-05; nothing here is a model-quality
result, because no model was trained.

## What The Block Does

```text
x  (B, T, d_model)
  |
  +--> LayerNorm ----------------------------------------+
  |                                                        |
  |    MultiHeadAttention  (hand-written, causal + padding)|
  |                                                        v
  +<-------------------------- add <-----------------------+
  |
  +--> LayerNorm ----------------------------------------+
  |                                                        |
  |    FeedForward: Linear -> GELU -> Linear               |
  |                                                        v
  +<-------------------------- add <-----------------------+
  |
  v
out (B, T, d_model)
```

- **Layer norm** normalizes over the **last** axis, so each of the `B * T`
  positions gets its own mean-0, variance-1 normalization over `d_model`
  features. No position sees another position's statistics and no batch row sees
  another's. `gamma` and `beta` are registered as learnable parameters.
- **Attention** is the only place the `T` axis is mixed. The causal mask blocks
  every position above the diagonal; the padding mask blocks padded keys; the two
  are **summed additively**.
- **The residual stream** is the identity path around each sublayer. Pre-norm
  keeps that path clean, which is what makes a deep stack trainable.
- **The MLP** is position-wise: it mixes features, never positions.

## Reproduction

Run from the repository root. Python 3.13, PyTorch 2.11.0, CPU, seed 0,
`b08601e0` WikiText-2 revision under `AI_ROADMAP_DATA`.

```bash
export AI_ROADMAP_DATA="$HOME/data/ai_research_roadmap"

.venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_multihead.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py
.venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
.venv/bin/python phase02_llms/week10_attention_block/block.py
.venv/bin/python phase02_llms/week09_tokenization_text_data/lm_data.py
```

Expected results:

| Command | Expected |
| --- | --- |
| `test_attention.py` | 15/15 PASS, exit 0 |
| `test_multihead.py` | 9/9 PASS, exit 0 |
| `test_block.py` | 11/11 PASS, exit 0 |
| `test_session4_lm_data.py` | 9/9 PASS, exit 0 |
| `block.py` | `(2, 8, 32)` shapes, 12,704 parameters |
| `lm_data.py` | per-split stream/window counts, one worked example |

The prepared-dataset counts on the pinned corpus:

| Split | Rows | Records | Stream tokens | Windows |
| --- | --- | --- | --- | --- |
| train | 36,718 | 23,767 | 3,682,072 | 28,767 |
| validation | 3,760 | 2,461 | 385,377 | 3,011 |
| test | 4,358 | 2,891 | 436,253 | 3,409 |

## What The Checks Establish

- **Shapes and dtypes** are preserved end to end for `(B, T, d_model)`.
- **Causality is real, not just a mask value.** Perturbing a future token leaves
  every earlier position's output bit-identical - the direct test, not an
  inspection of the mask matrix.
- **Attention rows are distributions** over unmasked positions, and masked
  positions receive exactly zero weight.
- **Multi-head equals a manual per-head loop**, including with the combined
  causal and padding mask. This is what catches a missing backward transpose: at
  `H = 2` the error changes the output by about 0.4, while at `H = 1` it is
  invisible.
- **Pre-norm composition is exact** - the block's output matches a reference
  rebuilt from its own submodules in the frozen pre-norm order.
- **Every parameter receives a non-`None` gradient** after one backward pass.
- **Numerical stability holds**: softmax survives large logits, layer norm
  survives a near-constant position, and a fully masked attention row does not
  produce `NaN`.
- **The data layer is self-consistent**: consecutive windows share exactly one
  token, targets are shifted exactly once, and the stream appends exactly one EOS
  per nonempty record. The validation stream (385,377) cross-checks against the
  Session 3 token count (382,916) plus 2,461 EOS tokens.

## What The Checks Do Not Establish

- **No training, no quality.** There is no perplexity, accuracy, loss curve, or
  generation sample in Week 10, because no model was trained. The 12,704
  parameters were never optimized.
- **No scale evidence.** All fixtures are small (`T = 8`). Correctness at `T = 8`
  is not evidence of numerical behavior at `T = 128` or at depth `N > 1`.
- **No performance claim.** The only timing recorded this week is a micro-
  benchmark of the mask builder (loop vs vectorized, up to ~4,500x at
  `(128, 129)`); it is not a model-throughput measurement.
- **No positional behavior.** The block is position-agnostic by design, so
  none of these checks say anything about sequence order.
- **No library-free guarantee by construction.** Compliance with the
  "hand-written only" rule was checked by reading the sources and scanning for
  forbidden calls, not by a runtime trap.
- **No real-corpus mask exercise.** The prepared dataset's masks are verified on
  fixtures; the first real batch through a model happens in Week 11.

## Limitations And Known Issues

1. **A literal `<eos>` inside a source row** would be tokenized to id 1 and split
   one record into two. The pinned WikiText-2 splits contain zero such rows, so
   this is a documented contract edge case, not an observed defect. It is worth a
   guard if the corpus is ever swapped.
2. **`lone_trailing_token` has no real-corpus coverage** - all three splits end
   on a full window, so only the hand-written fixture exercises that branch.
3. **The test split of WikiText-2 was never encoded** in the Session 3
   tokenizer comparison; the split boundary is declared but only train and
   validation have measured token statistics.
4. **`attn.w_k.bias` can show a zero-norm gradient** because a shared key-bias
   shift cancels inside softmax. The fixture confirms `grad is not None`; a
   zero-valued gradient is not the same thing as a disconnected parameter.

## What Week 11 Inherits

- `TransformerBlock(d_model, n_heads, d_ff)` on `(B, T, d_model)` with a
  broadcastable additive mask, to be stacked `N` times.
- The frozen mask conventions: additive, summed causal + padding, masked weight
  exactly zero, no `NaN` on a fully masked row.
- The frozen data contract: tokenizer artifact and vocab 2048, PAD 0, EOS 1,
  `IGNORE_INDEX -100`, `T = STRIDE = 128`, and the four batch tensors.
- Two traps stated so they are not rediscovered: targets are already shifted, and
  `attention_mask` is not `loss_mask`.

## References

- Vaswani et al., *Attention Is All You Need* - https://arxiv.org/abs/1706.03762
- Harvard NLP, *The Annotated Transformer* -
  https://nlp.seas.harvard.edu/annotated-transformer/
- Ba, Kiros, Hinton, *Layer Normalization* - https://arxiv.org/abs/1607.06450
- Hendrycks, Gimpel, *Gaussian Error Linear Units (GELUs)* -
  https://arxiv.org/abs/1606.08415
- Xiong et al., *On Layer Normalization in the Transformer Architecture* -
  https://arxiv.org/abs/2002.04745
