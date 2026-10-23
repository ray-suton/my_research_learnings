# Session 4 - Residuals, Layer Norm, The MLP, And The Assembled Block

## Status

**Closed 2026-10-01.** The understanding check, learner implementation, review,
fixture run, smoke run, and post-code understanding check are recorded below.
`attention.py` and `multihead.py` stayed untouched.

[Session 3](session3.md) closed on 2026-09-30 with 9/9 checks passing.

## Where This Sits

```text
x -> [ SESSION 4 ] -> x
      ln1 -> attention -> add
      ln2 -> MLP       -> add
```

`attention` is Session 3's `MultiHeadAttention`. The MLP is new here, and so are
the two normalisations and the two residual adds.

This is the last piece of the roadmap's "tested Transformer block" deliverable.
Week 11 stacks blocks like this into a mini-GPT.

## What Sessions 2 And 3 Left

Reused unchanged:

- `MultiHeadAttention(d_model, n_heads)` on `(B, T, d_model)`, mask forwarded.
- `scaled_dot_product_attention`, and both additive mask builders, from Session 2.

What attention deliberately does **not** do, and this session must:

- It has no normalisation. Raw activations go in.
- It has no per-position nonlinearity. Attention routes information; something
  else has to compute with it.
- It is not wrapped in a residual, so nothing protects the gradient path through
  a stack of them.

## The Contract To Implement

`EXPERIMENT_PLAN.md` remains the authority if these diverge.

| Component | Input | Output |
| --- | --- | --- |
| `LayerNorm(d_model, eps=1e-5)` | `(..., d_model)` | same shape |
| `FeedForward(d_model, d_ff)` | `(B, T, d_model)` | `(B, T, d_model)` |
| `TransformerBlock(d_model, n_heads, d_ff)` | `(B, T, d_model)`, optional mask | `(B, T, d_model)` |

Frozen decisions, from the experiment plan:

| Item | Choice |
| --- | --- |
| Norm placement | **Pre-norm**: the norm is inside the branch, the residual path is a clean identity |
| Block form | `x = x + attn(ln1(x))`, then `x = x + mlp(ln2(x))` |
| Layer norm axis | The **last** axis - per position, per sample, over features |
| `eps` | `1e-5`, added inside the square root |
| `gamma`, `beta` | Learnable, initialised to ones and zeros |
| MLP shape | `d_model -> d_ff -> d_model`, activation in between |
| Activation | `nn.GELU()` |
| Forbidden | `torch.nn.LayerNorm` - writing it by hand is the exercise |

## Shape Walkthrough

| stage | shape |
| --- | --- |
| input `x` | `(B, T, d_model)` |
| `ln1(x)` | `(B, T, d_model)` |
| `attn(ln1(x), mask)` | `(B, T, d_model)` |
| first residual add | `(B, T, d_model)` |
| `ln2(...)` | `(B, T, d_model)` |
| `mlp(...)` | `(B, T, d_model)` |
| second residual add | `(B, T, d_model)` |

Layer norm normalises over the **last** axis. For a `(B, T, d_model)` input,
that is `B * T` independent normalisations, each over `d_model` numbers. No
position sees another position's statistics, and no batch row sees another's.

## Task Checklist

- [x] Understanding check before writing code (2026-10-01).
- [x] Scaffold: placeholder bodies in a new module, leaving `attention.py` and `multihead.py` untouched.
- [x] Implement `LayerNorm` by hand, over the last axis, with `gamma` and `beta`.
- [x] Implement `FeedForward`: `d_model -> d_ff -> d_model` with a nonlinearity.
- [x] Implement `TransformerBlock` as the two pre-norm lines.
- [x] Propagate the mask through the block to attention.
- [x] Pass the fixture checks, including the pre-norm composition check (11/11, 2026-10-01).
- [x] Confirm every parameter receives a gradient.
- [x] Implement `main()` and run it (2026-10-01).
- [x] Record the run in `RUN_LOG.md` and `RESULTS.md`.
- [x] Understanding check after the code (2026-10-01).

## Finish Condition

1. All fixture checks pass, run with the recorded command.
2. The block's forward matches a reference rebuilt from its own submodules in
   the frozen pre-norm order - this is what catches a missing residual or a
   norm on the wrong side.
3. Every parameter has a non-`None` gradient after one backward pass.
4. Layer norm's output is mean-zero and unit-variance per position, and stays
   finite on a constant input.
5. The learner can explain what the residual path buys, and why layer norm
   normalises per position rather than per batch.

## Common Traps

- **Norm on the wrong side of the residual.** `ln(x + sublayer(x))` is post-norm.
  It still runs, still has the right shape, and changes what the block learns.
- **Normalising over the wrong axis.** Over `T` instead of over features would
  mix positions and is a different operation entirely.
- **Dropping a residual add.** Then the block becomes `sublayer(ln(x))` and the
  identity path is gone - and the gradient check is where it shows up.
- **A missing `eps`**, or `eps` outside the square root, on a near-constant
  position.
- **`gamma`/`beta` as plain tensors.** They would never train. Same failure as
  Session 3's unregistered projection.
- **Reaching for `nn.LayerNorm`.**
- **Touching `attention.py` or `multihead.py`.** Both are closed and verified.

## Implementation And Verification - 2026-10-01

The learner implemented `LayerNorm`, `FeedForward`, and the two pre-norm
residual lines in `TransformerBlock`. The instructor supplied only mechanical
review feedback during debugging and added the requested `main()` smoke runner;
`attention.py` and `multihead.py` were not changed.

Fixture command:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py
```

Result: **11/11 fixture checks passed**, including feature-axis normalization,
constant-input stability, parameter registration, MLP nonlinearity, pre-norm
composition, residual preservation, gradient presence, and mask propagation.

Smoke command:

```text
.venv/bin/python phase02_llms/week10_attention_block/block.py
```

Output: input and output shapes `(2, 8, 32)`, **12,704 parameters**, and mean
absolute residual movement `0.259858`. The unchanged Session 2 and Session 3
fixtures were re-run with **15/15** and **9/9** checks passing.

## Understanding Checks - 2026-10-01

Before implementation, the learner explained the identity-path purpose of
pre-norm, the per-token feature axis of layer norm, the separation between
attention's position mixing and the MLP's position-wise transformation, and
why the second norm consumes the state after the first residual add. During
review, the learner identified the distinction between a missing gradient and
an exactly zero gradient. The `attn.w_k.bias` zero gradient was explained by
softmax's invariance to a row-wise constant shift, while the gradient-presence
fixture confirmed that all registered parameters remain connected.

After implementation, the learner identified that attention supplies contextual
information before the MLP, and that a zero sublayer update leaves the residual
representation unchanged, allowing the block to learn corrections from an
identity starting path.

## Session 5 Handoff

Session 5 reviews the block, writes the Week 10 lab note, and hands off to
Week 11's mini-GPT, which stacks this block, adds token and positional
embeddings, and trains with the cross-entropy loss.
