# Behavioral Checks

**Implemented and green as of 2026-10-05.** All three suites pass:
`test_attention.py` (15/15), `test_multihead.py` (9/9), `test_block.py` (11/11).
The criteria below are retained as the contract they were written to be, not as
a wish list.

## Attention Core

- `scaled_dot_product_attention` returns the declared output and weight shapes
  for the fixture sizes.
- Attention weights are non-negative and sum to 1 along the last axis over
  unmasked positions; masked positions receive exactly 0 weight.
- A hand-computed attention example on a tiny fixture matches the implementation.
- The scale is `1 / sqrt(d_head)`: at fixed logits, changing `d_head` moves the
  distribution in the expected direction.

## Masking

- The causal mask blocks every position `s > t` and allows `s <= t`.
- A future-token perturbation does not change an earlier position's output - the
  direct causality test, not just a mask-value check.
- The padding mask zeroes attention to padding positions.
- The causal and padding masks combine additively; using only one of them fails a
  test.
- A fully masked row does not produce `NaN`.
- Large logits do not overflow.

## Multi-Head And The Block

- Splitting into heads and merging back round-trips exactly.
- Multi-head attention matches a manual per-head reference loop.
- The block preserves shape and dtype for `(B, T, d_model)`.
- Every parameter receives a non-`None` gradient after one backward pass.
- With the causal mask disabled, permuting the input positions permutes the
  outputs identically. With the causal mask enabled that no longer holds, and the
  test must not assert it.
- `MultiHeadAttention` raises a clear error when `d_model` is not divisible by
  `n_heads`.

## Execution

Use small local fixtures; no network access and no corpus data are needed for the
block checks. Record the eventual commands and complete outputs in
[RUN_LOG.md](../RUN_LOG.md), including skips. No training job is required to
validate the block contract.
