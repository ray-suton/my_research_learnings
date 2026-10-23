# Session 1 - Attention, Masking, and the Shape Contract

## Status

Prepared on **2026-09-28**. The conceptual discussion took place the same day and
is recorded below.

**Session 1 is complete as a guided conceptual checkpoint.** The
[discussion record](#discussion-record) separates the learner's reasoning from
instructor explanation. This does not establish independent mastery of every
subtopic, and no implementation or measurement has been checked.

No code exists for Week 10. Nothing has been implemented, executed, or measured.
[Session 2](#session-2-handoff) is next.

## From Week 9 To Week 10

Week 5's LSTM read a sequence one step at a time and carried a hidden state
forward. Week 9 produced the integer token streams and (in progress) next-token
examples that a model would consume. Attention replaces the sequential
recurrence with a direct, content-based lookup:

```text
Week 5:  ... -> [ recurrence: h_t = f(h_{t-1}, x_t) ] -> ...
Week 10: ... -> [ attention: every position reads every visible position ] -> ...
```

## Attention As A Soft Dictionary

A normal dictionary maps one key to one value. Attention maps a query to a
weighted combination of *all* values, where the weights come from query-key
similarity. Nothing is hard-selected, so the lookup is differentiable.

| Term | Role | Shape here |
| --- | --- | --- |
| Query `q` | what this position is looking for | `(B, H, T, d_head)` |
| Key `k` | what each position offers to be matched | `(B, H, T, d_head)` |
| Value `v` | what is passed on when matched | `(B, H, T, d_head)` |
| Score | `q · k` similarity | `(B, H, T, T)` |
| Weight | softmax over the last axis | `(B, H, T, T)` |

## Scaled Dot-Product Attention

```text
scores  = q @ kᵀ / sqrt(d_head)     # (B, H, T, T)
weights = softmax(scores + mask)     # over the last axis
out     = weights @ v                # (B, H, T, d_head)
```

Two details are worth understanding rather than copying:

- **Why divide by `sqrt(d_head)`.** If the components of `q` and `k` are roughly
  independent with unit variance, their dot product has variance of about
  `d_head`. Without rescaling, larger widths push logits into the flat region of
  the softmax, where gradients nearly vanish. Dividing by `sqrt(d_head)` keeps
  the logits at a comparable scale.
- **Why subtract the row maximum in softmax.** `exp` of a large positive logit
  overflows. Subtracting the row maximum is mathematically a no-op and
  numerically essential. The first implementation does it explicitly.

## Two Masks, Not One

Week 9's Session 1 introduced this distinction; the block is where it becomes
executable.

| Mask | Question it answers | Shape | Comes from |
| --- | --- | --- | --- |
| Causal mask | may position `t` see position `s > t`? | `(1, 1, T, T)` | the model: decoder self-attention |
| Padding mask | is position `s` a real token or filler? | `(B, 1, 1, T)` | the data: Week 9's LM `attention_mask` |

Both are applied by **adding** `-inf` to the masked logits, so the softmax sends
those positions to zero weight. They are summed, not multiplied, and neither
replaces the other. An all-ones padding mask does not make an attention layer
causal, and a causal mask does not hide padding.

## Multi-Head Attention

One attention pattern per layer forces one notion of relevance. Multi-head
attention runs `H` independent attention computations on `d_head = d_model / H`
slices and concatenates them, so different heads can specialize.

Shapes: `(B, T, d_model)` -> project to `q`, `k`, `v` -> reshape and split to
`(B, H, T, d_head)` -> attend -> merge back to `(B, T, d_model)` -> output
projection.

The transpose between `(B, T, H, d_head)` and `(B, H, T, d_head)` is the most
common silent bug in this week's code: get it wrong and the shapes may still
match while every head mixes the wrong positions.

## Residual Stream, Layer Norm, And The MLP

Attention moves information between positions; the other two pieces process each
position on its own.

- **Residual connection** adds a sublayer's output back to its input. It gives
  gradients a short path and lets the sublayer learn a correction rather than a
  full replacement.
- **Layer normalization** normalizes each position's features to zero mean and
  unit variance, then applies a learned scale and shift. It is computed over the
  last axis, per position and per sample.
- **Position-wise MLP** is the same small two-layer network applied to every
  position independently, usually expanding to `d_ff` and back.

With pre-norm, the block is:

```text
x = x + attention(layernorm(x))
x = x + mlp(layernorm(x))
```

## Positional Encodings (Read Now, Implement In Week 11)

Attention is permutation-equivariant: shuffle the input positions and the outputs
shuffle the same way. That is a problem for language, where order carries
meaning. Positional information is therefore added at the embedding layer, before
the block. This week reads that idea; Week 11 implements it with the mini-GPT's
embedding layer, which is where the roadmap places it.

## Cost

Attention compares every position with every visible position, so scores and
weights are `T x T`. Time and memory grow quadratically in sequence length. At
`T = 128` this is trivial; it is also why long-context work has its own
efficiency phase later in the roadmap.

## Opening Discussion Prompt

Why does attention need a query/key/value split at all - why not one similarity
matrix over the same vectors? Discuss what the separate projections let the layer
express, and what the causal mask removes that the padding mask does not.

Ask one focused question at a time. Explain routine shape and index arithmetic
directly rather than quizzing on it.

## Discussion Record

Recorded **2026-09-28**. Seven questions, asked one at a time, with corrections
between each. "Learner evidence" is what was produced unprompted; "guidance"
marks what had to be supplied, including explanations given after the learner
asked for them directly.

| Topic | Learner evidence | Guidance and assessment limit |
| --- | --- | --- |
| Q/K/V roles | Initially inverted query and value ("query is like the answer"). After correction, supplied an independent asymmetry example: "it" needs "dog"; "dog" does not need "it". | The role mapping was supplied, not derived. The asymmetry example was the learner's own and is the part that counts. |
| Why three projections | Did not derive it. Needed the symmetry argument spelled out: with one shared `W`, `q_t · k_s = x_tᵀ Wᵀ W x_s` equals `q_s · k_t`, forcing a symmetric grid. | Instructor-supplied. The learner recognized it afterward but had not inferred it. |
| Mask failure mode | First answered what the masks *mean* rather than what breaks. On a narrower re-ask, correctly identified that dropping the padding mask lets real positions attend to `PAD` slots. | Guidance supplied why both a shape assertion and a row-sum check still pass, which is why the project needs behavioral checks. |
| Where padding bites | Correctly located padding to the short final window and recognized that full windows make the mask idle. | Batch-level nuance supplied: one short row pads the entire batch, and the mask becomes row-varying. Inference-time padding deferred to Week 12. |
| Scaling | Correct direction (logits shrink) and correct framing (trading one failure for another), unprompted. | Mechanism was imprecise: attributed the vanishing to the softmax, where near-uniform weights actually give the *largest* Jacobian. Corrected: the attenuation is the scale factor multiplying the gradient back to `W_Q` and `W_K`. Also clarified that the no-division case is saturation, not classic gradient explosion. |
| Multi-head | Correctly identified that every head reads the full `d_model` input - the common misconception avoided. | Attributed head divergence to initialization rather than to separate parameter tensors. Supplied the tied-weights argument, the consistent head-split trap, and the emergent-not-supervised caveat. The learner then asked an unprompted follow-up about how divergence is guaranteed. |
| Permutation and order | Correctly identified that the block is order-blind. | Supplied the precise permutation-equivariance statement, plus the footnote that causal masking leaks weak position information without replacing positional encoding. |
| Block components | MLP role correct (nonlinear transform, expansion to `d_ff`). Residual connection was a complete independent answer: correction rather than replacement, easier gradient path, and the learn-zero case. | Supplied the `I + ∂f/∂x` form, the pre-norm versus post-norm residual-path distinction, and the output-projection init practice. |
| LayerNorm | Correctly identified that generation supplies no batch for batch statistics, which is the decisive practical argument. | Held the misconception that LayerNorm aggregates "all the layer's outputs." Corrected to strictly per-position, per-sample, over the feature axis; parallelism does not change what it reads. |

**Pattern.** Reasoning about mechanism and consequence is the strength: the
residual-stream answer and the Q/K asymmetry example were produced without help.
Two recurring gaps: answering an adjacent question instead of the one asked
(happened twice and needed a narrower re-ask), and reaching for CNN or batch
intuitions - neighborhoods, "all the outputs" - where the operation is strictly
per-token. No repeat quiz is scheduled; the practical check is whether these
distinctions survive contact with the code in Sessions 2-4.

## Clarifications To Carry Forward

- Divergence between heads comes from separate parameter tensors, not from
  initialization itself; tied weights would stay identical forever.
- The head split must be consistent across `q`, `k`, and `v`, or heads pair the
  wrong slices while the shapes still look valid.
- LayerNorm reads one token's own features only; it never aggregates across
  positions or across the batch.
- Residual addition makes each sublayer learn a correction, and a block that adds
  nothing reduces to identity.
- The causal and padding masks are added, not multiplied, and never conflated.
- Masked positions must receive exactly zero attention weight.
- A fully masked row must be handled explicitly, or the softmax returns `NaN`.
- `d_head = d_model / n_heads`; the split and merge order must round-trip.
- The block is position-agnostic; positional information enters at the embeddings.
- The block is checked with fixtures; training and loss live in Week 11.

## Session 2 Handoff

1. Freeze the fixture sizes from the [experiment plan](EXPERIMENT_PLAN.md).
2. Implement scaled dot-product attention and the additive causal mask by hand.
3. Write the first checks: shape, row sums, exactly-zero masked weights, and the
   future-token perturbation test that proves causality.
4. Record the actual commands and outcomes in [RUN_LOG.md](RUN_LOG.md).

Session 2 has not started. The learner owns ordinary lesson implementation; the
instructor explains and reviews. This closure authorizes documentation, not
automatic implementation of the block.
