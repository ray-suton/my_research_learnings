# Experiment Plan

## Status And Learning Question

Proposed on **2026-09-28**; nothing has been executed yet.

Can a hand-written Transformer block reproduce the reference behavior of scaled
dot-product attention, causal masking, multi-head projection, residual
connections, layer normalization, and a position-wise MLP - with shapes, masks,
and gradients that Week 11's training loop can consume unchanged?

Expected outcome: the hand-written block matches a slow, obviously-correct
reference on small tensors; masked positions receive exactly zero attention
weight; changing a future token does not change an earlier position's output;
every parameter receives a gradient.

This is an implementation week, not an experiment week. "Measured" here means
test outcomes, parameter counts, and runtimes - not model quality.

## Working Defaults To Freeze Before Execution

| Item | Proposed choice |
| --- | --- |
| Framework | PyTorch, CPU, hand-written |
| Forbidden shortcuts | `torch.nn.MultiheadAttention`, `torch.nn.TransformerEncoderLayer`, `torch.nn.functional.scaled_dot_product_attention`, and `torch.nn.LayerNorm` for the layer-norm exercise |
| Fixture sizes | `B = 2`, `T = 8`, `d_model = 32`, `n_heads = 4`, `d_head = 8`, `d_ff = 128` |
| Real sizes | `T = 128`, matching the frozen LM block length; `d_model` is chosen in Week 11 |
| Norm placement | Pre-norm: layer norm before each sublayer, residual added around it |
| Attention scaling | Divide logits by `sqrt(d_head)` |
| Softmax stability | Subtract the row maximum before exponentiating |
| Mask arithmetic | Additive: `-inf` where masked; the causal and padding masks are summed |
| Mask dtype | One convention only, recorded in the run - boolean or additive float |
| Dtype | `float32` |
| Seed | `0` for every recorded run |
| Compute | CPU; the fixture forward and backward pass should finish in seconds |

References: the [attention paper](https://arxiv.org/abs/1706.03762), the
[annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/), and
the [PyTorch attention documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).
Reading the documentation is expected; calling the function is not.

## The Block Contract

This is the interface Week 11 consumes. Freeze it before implementing.

```text
x:              (B, T, d_model)   float
attention_mask: (B, T)            1 = real token, 0 = padding   [produced by Week 9's LM data]
causal:         bool              True for a decoder block
```

| Component | Input | Output | Rule |
| --- | --- | --- | --- |
| `scaled_dot_product_attention(q, k, v, mask)` | `q, k, v` `(B, H, T, d_head)`; additive `mask` broadcastable to `(B, H, T, T)` | `out` `(B, H, T, d_head)`, `weights` `(B, H, T, T)` | `softmax(q @ kᵀ / sqrt(d_head) + mask) @ v`; weights sum to 1 over unmasked positions |
| `build_causal_mask(T)` | - | additive `(1, 1, T, T)` | 0 on and below the diagonal, `-inf` above |
| `padding_to_mask(attention_mask)` | `(B, T)` of 0/1 | additive `(B, 1, 1, T)` | 0 for real tokens, `-inf` for padding |
| `MultiHeadAttention(d_model, n_heads)` | `x` `(B, T, d_model)` | `(B, T, d_model)` | one `q`/`k`/`v` projection each; split into heads; attend; merge; output projection |
| `FeedForward(d_model, d_ff)` | `(B, T, d_model)` | `(B, T, d_model)` | linear, activation, linear, applied per position |
| `TransformerBlock(d_model, n_heads, d_ff)` | `x` `(B, T, d_model)` and an additive mask | `(B, T, d_model)` | pre-norm: `x = x + attn(ln(x))`, then `x = x + ffn(ln(x))` |

Rules the tests enforce:

- The causal and padding masks are **summed additively**. Never multiply them,
  and never let one overwrite the other.
- Every masked position contributes exactly zero attention weight.
- A row that is entirely masked must not produce `NaN`. Handle it explicitly and
  test it; `-inf` through a softmax gives `NaN` otherwise.
- `d_model` must be divisible by `n_heads`; validate and raise a clear error.
- The block is position-agnostic. No positional encoding goes inside it.

## Measurement Contract

This week measures correctness and cost, not quality.

- Pass/fail per behavioral check, with any failing fixture retained.
- Parameter count per component and for the assembled block.
- Forward and backward wall time on CPU at the fixture sizes, with seed and
  package versions recorded.
- Gradient presence: every parameter has a non-`None` gradient after one
  backward pass on the fixture.
- Masking statistics on a saved attention weight matrix: rows sum to 1 over
  unmasked positions, and masked entries are exactly 0.
- Determinism: two runs with the same seed and input agree exactly.

An optional small PNG of a causal attention weight matrix may be saved under
`results/`. It is illustrative, not a measurement.

## Failure Conditions And Completion

Stop before declaring success if any of these hold:

- A library shortcut (`nn.MultiheadAttention`, `nn.TransformerEncoderLayer`,
  `F.scaled_dot_product_attention`) is doing work meant to be hand-written.
- The causal mask is applied on the wrong axis, or transposed, so causality
  silently fails while the shapes still match.
- Attention weights do not sum to 1 over unmasked positions.
- The padding mask and the causal mask are conflated into one.
- A fully masked row produces `NaN`.
- Softmax overflows on large logits because max-subtraction was skipped.
- A future-token perturbation changes an earlier position's output.

Completion requires implemented and passing behavioral checks, a recorded run
note with the actual commands, the prepared LM dataset from the carry-over track,
and a lab note. See [tests/README.md](tests/README.md) for the planned checks.
