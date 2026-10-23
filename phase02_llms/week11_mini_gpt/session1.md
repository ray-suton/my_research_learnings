# Session 1 - Embeddings, Positions, The LM Head, And Next-Token Loss

## Status

**Closed 2026-10-05.** Initialized 2026-10-05 as the Week 11 starter. Conceptual
session; no model code was written. All eight decision rows were confirmed and
the frozen contract is in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md). The
discussion record and the learning record are appended below. Session 2 owns
`mini_gpt.py`.

## Where This Sits

```text
Week 9   text -> token ids            (done: tokenizer, streams, prepared dataset)
Week 10  ids  -> hidden states        (done: hand-written TransformerBlock, position-agnostic)
Week 11  + embeddings + positions     <- this session's subject
         + N blocks + LM head
         -> logits -> next-token loss  <- and this session's subject
```

Week 10 ended with a component that maps `(B, T, d_model) -> (B, T, d_model)` and
knows nothing about order or vocabulary. Week 11 wraps it into a model that reads
token ids and produces a probability distribution over the next token at every
position.

## The Five Things To Actually Understand

### 1. Token embedding

A lookup table `nn.Embedding(vocab_size, d_model)`. Token id 1373 becomes a
learned `d_model` vector. It has no notion of similarity between tokens by
itself - any structure is learned. This is the first place the vocabulary axis
disappears and the feature axis appears.

### 2. Positional embedding

A second lookup table `nn.Embedding(T, d_model)`, added to the token embedding.
Without it the model is permutation-invariant: `"dog bites man"` and `"man bites
dog"` produce the same set of position-wise computations, because attention is
just a weighted sum over positions with no built-in order.

Two things worth stating precisely:

- Position `t` is a **slot**, not a token. Position 5 is position 5 whether the
  token there is `the` or `,`.
- The positional embedding is added **before the first block**, so every block
  sees an already-ordered residual stream and stays position-agnostic itself.

### 3. The decoder stack

`N` copies of Week 10's `TransformerBlock`, connected in sequence, with the
causal mask applied at every layer. Each block preserves `(B, T, d_model)`. The
causal mask is what makes this a *decoder*: position `t` can read positions
`<= t` and nothing later, which is exactly what "predict the next token" needs.

Depth buys composition: one block can attend from a token to its neighbours, and
later blocks can attend to what earlier blocks computed. The residual stream is
the shared channel those blocks read from and write to.

### 4. The LM head

A `nn.Linear(d_model, vocab_size)` that turns each position's final hidden state
into `vocab_size` logits - one score per possible next token. Applied to all `T`
positions at once, so one forward pass produces `T` next-token predictions, and
one window of 128 inputs yields 128 training signals.

### 5. Next-token loss

For each position `t`, the target is the token at `t + 1`. Cross-entropy between
the logits and that target, averaged over the positions that count.

Three details decide whether this is correct or subtly wrong:

- **The targets are already shifted.** The data layer produced
  `targets = window[1:T+1]`. A training loop that shifts again predicts two
  tokens ahead.
- **Not every position counts.** Padding targets are `-100`, and positions whose
  next token is padding must not contribute. Use `reduction="none"` and weight by
  `loss_mask`.
- **No softmax before the loss.** `CrossEntropyLoss` applies log-softmax
  internally; applying softmax first is a double normalization.

## Shape Walkthrough

| stage | shape |
| --- | --- |
| `input_ids` | `(B, T)` of ints |
| `embed_tokens(input_ids)` | `(B, T, d_model)` |
| `+ embed_positions(0..T-1)` | `(B, T, d_model)` |
| one `TransformerBlock` | `(B, T, d_model)` |
| `N` blocks | `(B, T, d_model)` |
| final layer norm | `(B, T, d_model)` |
| `lm_head(...)` | `(B, T, vocab_size)` |
| cross-entropy vs `targets (B, T)` masked by `loss_mask (B, T)` | scalar |

## Three Questions To Answer Before Writing Code

1. If the positional embedding is added to the token embedding, why does the
   block still not need to know about positions?
2. The data layer already shifted the targets. If a model API shifted labels
   internally, what would the double shift do to the loss, and how would you
   detect it?
3. The last real input position in a padded window has a real input but a
   padding target. Should `attention_mask` or `loss_mask` exclude it from the
   loss, and why is that the opposite of what `attention_mask` does there?

## Decisions To Freeze

The full table with reasons lives in the [README](README.md#decisions-to-freeze-in-session-1).
Confirm or overturn each row, then copy the confirmed set into
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md):

- Positional information: learned absolute vs sinusoidal.
- Where positions are added: before the first block vs inside each block.
- Weight tying: tied vs untied LM head.
- Final normalization: present vs absent.
- Loss reduction: masked mean vs plain mean.
- Initialization: GPT-2-style scaled vs PyTorch default.
- Softmax: never in the model vs before the loss (recommended: never).
- Dropout: 0.1 for the real run, disabled for the overfit check.

## Task Checklist

- [x] Discuss each of the five concepts and answer the three questions above. (2026-10-05; see the discussion record)
- [x] Confirm or overturn every row of the decision table and record why. (2026-10-05)
- [x] Copy the frozen decisions into `EXPERIMENT_PLAN.md`. (2026-10-05)
- [x] State the model sizes (`d_model`, `n_heads`, `n_layers`, `d_ff`, dropout). (frozen in the plan)
- [x] State the optimizer settings and the training budget. (frozen in the plan)
- [x] State the Session 3 overfit criterion and the Session 4 HPC budget. (recorded below; step bound to be validated in Session 3)
- [x] Hand off to Session 2 to implement `mini_gpt.py`.

## Finish Condition

Session 1 is done when a reader can, from this file and the frozen plan, state
the exact tensor flow from `input_ids` to the scalar loss, name every frozen
decision, and explain why a plain-mean loss and an unshifted target are both
wrong. No code is required here.

## Common Traps

- **Encoding position as data instead of as a signal.** Positional embeddings are
  added, not concatenated, and they are learned parameters.
- **Confusing pad slots with pad tokens.** Pad positions still get an embedding;
  that is fine because padding is masked, but it means position indices must be
  valid for every window, padded or not.
- **Double shift.** `targets` is already `input_ids` shifted by one.
- **Plain-mean loss.** Padding dominates short windows and quietly changes the
  objective.
- **Softmax then cross-entropy.** Double normalization; the gradient is wrong.
- **Untied head by accident.** Tying is an explicit choice, and it is easy to
  build an untied head without noticing.
- **Skipping the final norm** in a pre-norm stack, then blaming the optimizer for
  unstable loss.
- **Stacking depth without residual-scaled init.** The residual stream's variance
  grows with depth and the loss diverges or stalls.

## Discussion Record - 2026-10-05

### Concepts covered

- **Token embedding.** `nn.Embedding(2048, 128)` is a lookup table, not a
  transformation: token id -> one learned 128-dim row. Token ids are arbitrary
  integers with no arithmetic meaning, so the model needs continuous vectors, and
  the one-hot alternative is orthogonal across tokens and carries no geometry.
  Only the rows present in a batch receive gradient, and `padding_idx=0` pins the
  PAD row at zero, which under tying makes the PAD logit exactly `0`.
- **Positional embedding.** `nn.Embedding(128, 128)`, one learned vector per
  slot, added elementwise before the first block. Attention weights are
  `softmax(q . k)` with `q` and `k` from content alone, so a head cannot select a
  key by distance; order has to enter before the mixing layers.
- **Decoder block.** Week 10's `TransformerBlock`, unchanged: multi-head causal
  self-attention (4 projections, `H = 4`, `d_head = 32`) plus a position-wise
  GELU MLP (`128 -> 512 -> 128`), both inside pre-norm residuals. Shape is
  preserved end to end, which is why `N` of them stack, and the weights are
  shared across all `T` positions.
- **Encoder vs decoder.** The difference is the mask. Bidirectional attention is
  more informative per position, but with next-token targets it puts the label
  inside the input, so the loss collapses into a copy operation and the training
  procedure cannot be run at inference. Causal masking is what makes the target
  legitimate. Padding masks and causal masks are different objects combined
  additively. The original Transformer decoder had cross-attention; this
  GPT-style block does not.
- **LM head.** `nn.Linear(128, 2048)`; `logit_v = W[v] . h`; logits only, never
  probabilities; weight-tied, so `logit_v = E[v] . h`. Turning scores into a
  token is decoding, which is Week 12.
- **Next-token loss.** Already-shifted targets, `loss_mask`-weighted mean, and no
  softmax before the loss.

### The three questions

**Q1 - positions and the block.** Learner: the position vector is needed so the
model can represent sequence and dependency, and adding it after the last block
means the layers never use it. Correct in spirit. Corrected: "nothing symbolizes
position" is too strong - the causal mask leaks a *count* (position `t` can see
`t` keys), so the block has a position-adjacent signal but no coordinate it can
attach to a token. The conclusion is unchanged: position must be added before the
first block.

**Q2 - double shift.** Learner: the model is trained on the second-next token,
and the last positions lose supervision. Correct on the first half - the
effective target becomes `input_ids[t+2]`. Corrected on the second: on a full
window, position `T-1` is never scored and `w_1` is never a target; on a padded
window the boundary position's target becomes `-100`, so `ignore_index` returns
`0.0`. The sharper defect is the denominator - `loss_mask.sum()` is unchanged, so
the mean divides by a set one larger than the set that produced gradient.

**Q3 - attention_mask vs loss_mask.** Learner: `k` is a real token, so both masks
should include it. Correct for `attention_mask`, which is indexed by input.
Incorrect for `loss_mask`, which is indexed by target: `targets[:, k] =
input_ids[:, k+1]`, a pad slot stored as `-100`, so nothing is scored there.
Including it would add `0` to the numerator and `1` to the denominator. The last
real token is still predicted, by position `k-1`.

### Evidence gathered

`CrossEntropyLoss` semantics were measured on this venv (`torch 2.11.0+cu128`)
rather than recalled, on a `(2, 4)` target block with 3 of 8 positions padded:

| Pad target stored as | explicit `loss_mask` mean | default `reduction="mean"` | `sum / element_count` |
| --- | --- | --- | --- |
| `-100` | 3.175697 | 3.175697 | 0.992405 |
| token id `0` | 3.175697 | 3.103513 | 1.551756 |

PyTorch's `mean` divides by the count of non-ignored targets, so with `-100`
targets it coincides with the explicit masked mean, and it diverges as soon as
the sentinel is dropped. That measurement is the reason row 5 is written
explicitly.

### Decision table - confirmed

| Decision | Frozen choice |
| --- | --- |
| Positional information | Learned absolute, `nn.Embedding(128, 128)` |
| Where positions are added | To the token embedding, before the first block |
| Weight tying | LM head tied to the token embedding |
| Final normalization | Present, after the last block and before the LM head |
| Loss reduction | `reduction="none"`, then a `loss_mask`-weighted mean |
| Initialization | Normal `(0, 0.02)`; residual outputs scaled by `1/sqrt(2 * n_layers)` |
| Softmax | Nowhere in the model; logits only |
| Dropout | `0.1` for the training run, `0.0` for the overfit check |

Reasons, accepted costs, and overturn criteria are recorded in the
[plan's freeze record](EXPERIMENT_PLAN.md#session-1-freeze-record---2026-10-05).
Rejected alternative for row 1: sinusoidal encodings, which compute a vector for
any index but would return an unvalidated value past 128 instead of failing.

### Two precision corrections to this file

1. **"permutation-invariant"** (in the positional-embedding section) is the
   bidirectional statement. With a causal mask, a model without positional
   embeddings is not literally permutation-invariant, because position 1 always
   sees exactly one token whatever sits there. The practical conclusion - add
   positions before the first block - is unchanged.
2. **"Plain mean over all positions - silently trains on padding"** (Common
   Traps) holds only when the padding target is not the ignore sentinel. With
   `-100` targets, PyTorch's default `mean` already excludes them and matches the
   explicit masked mean. The real argument for the explicit form is that the
   denominator is declared, auditable, and assertable - the count check is what
   catches the double shift.

## Learning Record - 2026-10-05

Independently reasoned by the learner:

- Positional information must reach the layers that mix tokens, so it cannot be
  added only at the read-out.
- A double shift trains the model on the token two ahead.
- `attention_mask[:, k] = 1` at the last real input position.
- Parameter saving is a correct reason for tying, and `reduction="none"` plus a
  `loss_mask` is how positions are selected.

Required instructor correction:

- The causal-mask position leak, and that "no position signal at all"
  understates what the block can see.
- The edge-effect mechanism and the denominator defect in the double shift.
- That `loss_mask` is indexed by target, not by input.
- "More regularized" stated as a label rather than the equality-constraint
  mechanism.
- "Disproportionate logits" is a by-construction property with confounds, so it
  cannot serve as an overturn criterion.
- Elementwise product, not a dot product, in the masked mean.

Supplied by the instructor rather than derived by the learner: the five concept
explanations, the encoder-decoder distinction, the LM-head and unembedding
mechanics, and the `ignore_index` measurement above.

No code was written, and no run was performed, in this session.
