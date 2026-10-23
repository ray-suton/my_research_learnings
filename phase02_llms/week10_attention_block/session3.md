# Session 3 - Multi-Head Attention

## Status

**Closed 2026-09-30.** `MultiHeadAttention` is implemented in
[multihead.py](multihead.py) and **9/9 fixture checks pass** in
[tests/test_multihead.py](tests/test_multihead.py), including equivalence against
an independently written per-head reference. `attention.py` from Session 2 was
not touched and still passes 15/15. The run is in [RUN_LOG.md](RUN_LOG.md); the
understanding checks are [below](#understanding-checks---2026-09-30).

[Session 2](session2.md) closed on 2026-09-29 with 15/15 checks passing.

## Where This Sits

```text
x (B, T, d_model) -> [SESSION 3: project, split, attend, merge] -> (B, T, d_model)
                                            |
                              scaled_dot_product_attention (Session 2)
```

Session 2 built one head's attention. Session 3 wraps it in the projections and
the head split so it can be called on the block's actual input. Session 4 adds
residuals, layer norm, and the MLP around this.

## What Session 2 Left

Reused unchanged:

- `scaled_dot_product_attention(q, k, v, mask=None)` at `(B, H, T, d_head)`.
- `build_causal_mask(T)` and `padding_to_mask(attention_mask)`, additive floats
  that combine by addition.

What `sdpa` deliberately does **not** do, and this session must:

- It never sees `x`. It takes `q`, `k`, `v` that someone else already projected
  and split.
- It has no learned parameters. Every projection is new here.

## The Contract To Implement

`EXPERIMENT_PLAN.md` remains the authority if these diverge.

| Component | Input | Output |
| --- | --- | --- |
| `MultiHeadAttention(d_model, n_heads)` | `x` `(B, T, d_model)`, optional additive mask | `(B, T, d_model)` |

Steps inside, in order:

1. Project `x` three ways with one `d_model x d_model` linear layer each:
   `q`, `k`, `v`, all `(B, T, d_model)`.
2. Split the last axis into heads: `(B, T, d_model)` -> `(B, T, H, d_head)`,
   where `d_head = d_model // n_heads`.
3. Transpose to `(B, H, T, d_head)` and call `scaled_dot_product_attention`.
4. Transpose back to `(B, T, H, d_head)` and merge to `(B, T, d_model)`.
5. Apply the output projection `W_O`, also `d_model x d_model`.

Rules:

- `d_model % n_heads == 0`, validated with a clear error. `d_head = d_model // n_heads`.
- One projection matrix per role, split by head. The split must be **identical**
  across `q`, `k`, and `v`, or the heads pair the wrong slices.
- The mask passes through unchanged. `MultiHeadAttention` does not build masks.
- No `torch.nn.MultiheadAttention`. It is the thing being written.

## Shape Walkthrough

| stage | shape |
| --- | --- |
| input `x` | `(B, T, d_model)` |
| after `W_Q`, `W_K`, `W_V` | `(B, T, d_model)` each |
| split into heads | `(B, T, H, d_head)` |
| transposed for attention | `(B, H, T, d_head)` |
| `sdpa` output | `(B, H, T, d_head)`, weights `(B, H, T, T)` |
| transposed back | `(B, T, H, d_head)` |
| merged | `(B, T, d_model)` |
| after `W_O` | `(B, T, d_model)` |

The `(B, T, H, d_head)` <-> `(B, H, T, d_head)` transposes are the classic silent
bug in this session: get them wrong and the shapes still work while every head
mixes the wrong positions.

## Task Checklist

- [x] Understanding check before writing code (2026-09-30, six questions).
- [x] Scaffold: placeholder bodies in a new module, leaving `attention.py` untouched (2026-09-30).
- [x] Implement the three input projections and the output projection (2026-09-30).
- [x] Implement the head split and merge, consistent across `q`, `k`, `v` (2026-09-30).
- [x] Validate `d_model % n_heads` with a clear error (2026-09-30).
- [x] Wire in `scaled_dot_product_attention` and pass the mask through (2026-09-30).
- [x] Pass the fixture checks, including equivalence against a manual per-head loop - **9/9** (2026-09-30).
- [x] Record the run in `RUN_LOG.md` and `RESULTS.md` (2026-09-30).
- [x] Understanding check after the code (2026-09-30, three questions).

## Finish Condition

1. All fixture checks pass, run with the recorded command.
2. Multi-head output matches a manual per-head reference loop exactly - this is
   the check that proves the split and merge are consistent, not merely
   shape-correct.
3. The head split and merge round-trip exactly under a permuted head order.
4. `d_model % n_heads != 0` raises a clear error rather than corrupting shapes.
5. The learner can explain what multi-head buys over one head of the same total
   width, and what the mask does and does not affect in this layer.

## Common Traps

- **Inconsistent head splits.** `q` split one way, `k` another. Shapes pass,
  attention is nonsense.
- **The transpose pair.** `(B, T, H, d_head)` vs `(B, H, T, d_head)`. Swapping
  them keeps every shape valid.
- **Applying the mask per head.** The mask is `(1, 1, T, T)` or `(B, 1, 1, T)`
  precisely so it broadcasts across heads. Anything head-specific is a bug.
- **A new head count that does not divide `d_model`.** Validate and raise.
- **Reaching for `nn.MultiheadAttention`.** It is the exercise.
- **Changing `attention.py`.** Session 2 is closed; this session adds a module.

## Session 4 Handoff

Session 4 needs `MultiHeadAttention(d_model, n_heads)` with an optional additive
mask, on `(B, T, d_model)`. It then adds, around it and around a position-wise
MLP: layer normalization, residual connections, and the pre-norm block layout
frozen in the experiment plan.

## Understanding Checks - 2026-09-30

### Before the code - six questions

| Topic | Learner evidence | Guidance and assessment limit |
| --- | --- | --- |
| What multi-head buys over one wide head | "Coverage", and each head having its own projections - direction right | Precision supplied: H *independent* attention distributions instead of one, at the same parameter count. "Different qkv per token" corrected to "different projections per head". |
| Compute vs memory across head counts | Not correct initially | Supplied: matmul work is unchanged by H; the attention-weight tensor grows H-fold. |
| Does the mask change for multi-head | Correct - no change | Supplied what it would mean if it did: heads with different visibility. |
| Building a reference check | "It is the same mathematical function" - true but not sufficient | Supplied the independence requirement: a reference sharing the split/merge code cannot catch a bug in it, and must slice heads by hand. |
| Why `W_O` is needed | Correct after a prompt - "combine information from the different heads" | Supplied the block-structure reason: without it, output features are partitioned by head and nothing mixes them. |
| Parameter scaling | Said doubling `d_model` roughly doubles parameters - wrong | Supplied: projections are `d_model x d_model`, so parameters scale with `d_model^2`; H does not move the parameter count at all, only the attention memory. |

### After the code - three questions

| Topic | Learner evidence | Guidance and assessment limit |
| --- | --- | --- |
| Why the output must be `d_model` wide | That `d_head` is narrower - right in spirit; attributed the break to information loss and named the residual add as the component that restores the space | Supplied: the operative reason is the shape mismatch in the elementwise add, and the component in *this* layer that returns to the residual space is `W_O`, not the Session 4 add. |
| Parameter count with `H` changed | Declined to compute, but concluded correctly that nothing changes because H only subdivides the same weights | Correct conclusion, correct reason. Count supplied for the record: 4 x (32^2 + 32) = 4224 at `d_model = 32`, unchanged by H. |
| Does `H = 1` test the split or the transpose | Correct that the split is untested; claimed H=1 would still catch a missing transpose | Disproved by measurement: with H=1 the missing backward transpose gives an **identical** output, because swapping a size-1 axis against a size-T axis reorders nothing. At H=2 the same bug moves the output by 0.4. |

**Pattern.** Same as Sessions 1 and 2: mechanisms are solid once walked
through, and the thin spots are where two known facts have to be combined (the
shape mismatch behind the width requirement) or where a claim needs checking
rather than reasoning (the `H = 1` transpose case, which measurement settled the
other way). The reference-check reasoning was the strongest answer of the
session once supplied, and the learner applied it correctly by the end.
