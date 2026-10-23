# Session 2 - Scaled Dot-Product Attention and the Additive Masks

## Status

**Closed on 2026-09-29.** All four functions plus `main()` are implemented in
[attention.py](attention.py), and **15/15 fixture checks pass** in
[tests/test_attention.py](tests/test_attention.py). The run, its settings, and
the two defects found along the way are in [RUN_LOG.md](RUN_LOG.md); the
understanding check is [below](#understanding-check---2026-09-29).

[Session 1](session1.md) closed on 2026-09-28 as a guided conceptual
checkpoint. This session turned the first half of that concept into code.

## Where This Sits

```text
x -> [Session 3: projections, heads] -> q, k, v -> [SESSION 2] -> out -> [Session 4: block]
```

Session 2 builds the core of attention and both masks. Multi-head projection
(Session 3) and the residual/norm/MLP block (Session 4) sit on top of it, so
everything here is written to be reused unchanged.

## What Session 1 Frozen

Carried forward unchanged; do not relitigate these while implementing:

- Divide the logits by `sqrt(d_head)`, not by `d_head`.
- Softmax subtracts the row maximum before exponentiating.
- The causal and padding masks are **additive** and are **summed**, never
  multiplied and never one overwriting the other.
- Masked positions receive exactly zero attention weight.
- A fully masked row must not produce `NaN`.

## The Contract To Implement

[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) remains the authority if these ever
diverge. Four functions, all hand-written: no
`torch.nn.functional.scaled_dot_product_attention`, no
`torch.nn.MultiheadAttention`, no `torch.nn.TransformerEncoderLayer`.

| Function | Input | Output |
| --- | --- | --- |
| `stable_softmax(x, dim=-1)` | any tensor | same shape; non-negative, sums to 1 along `dim` |
| `build_causal_mask(seq_len)` | - | additive `(1, 1, seq_len, seq_len)`: 0 where `s <= t`, `-inf` where `s > t` |
| `padding_to_mask(attention_mask)` | `(B, T)` of 0/1, 1 = real | additive `(B, 1, 1, T)`: 0 real, `-inf` padding |
| `scaled_dot_product_attention(q, k, v, mask=None)` | `q, k, v` `(B, H, T, d_head)`; `mask` additive, broadcastable to `(B, H, T, T)` | `out` `(B, H, T, d_head)`, `weights` `(B, H, T, T)` |

The last one is three lines of arithmetic:

```text
scores  = q @ k.transpose(-2, -1) / sqrt(d_head)
weights = stable_softmax(scores + mask, dim=-1)
out     = weights @ v
```

Run the checks **from the repository root** — the virtual environment does not
resolve from inside the week folder:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py
```

Right now that command stops on `NotImplementedError` at the first check, which
is the expected red state.

## A Worked Illustration

Hand-computed, so you can check the arithmetic before trusting the code. Use
`T = 2`, `d_head = 1` (so `sqrt(d_head) = 1` and the scale does nothing), all
zeros for `q` and `k`, and values `2` and `4` for `v`:

```text
v = [[2], [4]]

no mask
  scores  = [[0, 0], [0, 0]]            q @ kᵀ
  weights = [[0.5, 0.5], [0.5, 0.5]]    softmax of equal logits
  out     = [[3], [3]]                  both positions average the two values

causal mask
  mask    = [[0, -inf], [0, 0]]
  weights = [[1.0, 0.0], [0.5, 0.5]]    position 0 may only see itself
  out     = [[2], [3]]                  position 0 returns v₀, position 1 averages
```

The `[[1.0, 0.0], [0.5, 0.5]]` row is the whole point of the causal mask: a
weight of exactly `0.0`, not a small number.

Two traces of the same idea with padding, since padding behaves differently:

- **Trailing padding, real first position.** `attention_mask = [1, 0]`. For row
  0, both masks block column 1, so the combined mask equals what the causal mask
  already did. Padding changes nothing here — which is why full windows make the
  padding mask idle.
- **Padding in the first position.** `attention_mask = [0, 1]`. Row 0 is now
  blocked by the causal mask (only `s <= 0` is allowed) *and* by the padding mask
  (column 0 is filler). Every entry is `-inf`. This is the NaN case, and it is
  the one the fixture checks by name.

## Task Checklist

- [x] Implement `stable_softmax` with max-subtraction (2026-09-29).
- [x] Implement `build_causal_mask` as an additive `(1, 1, T, T)` tensor (2026-09-29).
- [x] Implement `padding_to_mask` as an additive `(B, 1, 1, T)` tensor (2026-09-29).
- [x] Implement `scaled_dot_product_attention`, including the fully masked row (2026-09-29).
- [x] Implement `main()` and run it for a printed smoke summary (2026-09-29).
- [x] Pass all fixture checks in [tests/test_attention.py](tests/test_attention.py) - **15/15** (2026-09-29).
- [x] Record the actual command and outcome in [RUN_LOG.md](RUN_LOG.md), and the
  measured shapes/row-sums in [RESULTS.md](RESULTS.md) (2026-09-29).
- [ ] Write the short code walkthrough: what the scale does, what each mask
  blocks, and how the fully masked row is handled. **Still open** - the
  mechanism was demonstrated in the understanding check, but no written
  walkthrough exists yet.

## Finish Condition

All of these, in order:

1. Every fixture check passes, run with the recorded command.
2. The future-token perturbation check passes — this is the causality proof, not
   a mask-value inspection.
3. The fully masked row is handled explicitly, with the choice described in a
   comment, and no `NaN` appears.
4. The two masks are combined by addition in at least one exercised path.
5. The learner can explain, in their own words, why the causal and padding masks
   cannot be substituted for one another.

Passing the fixture file is necessary but not sufficient: item 5 is a spoken
check, and items 1-4 are only meaningful if the checks actually run.

## Common Traps

- **Scaling by `d_head` instead of `sqrt(d_head)`.** No crash, valid shapes, and
  a `test_scaling_prevents_saturation_at_large_d_head` failure. The head is
  over-shrunk toward a uniform average.
- **Considering the mask on the wrong axis**, or transposing it, so causality
  silently fails while the shapes still broadcast.
- **Copying the padding mask and calling it causal** (or the reverse). Both
  broadcast to `(B, H, T, T)`, so nothing structural complains.
- **Forgetting the fully masked row.** `exp(-inf)` is 0 everywhere, the row sums
  to 0, and `0/0` gives `NaN` that propagates through the whole tensor.
- **Skipping max-subtraction** because the fixture logits look small. The
  large-logit check will catch it.
- **Returning only `out`.** The fixture spec and every later debug need the
  weights too.

## First Learning Step

Start with `build_causal_mask` — it is the smallest piece, has no dependencies,
and the fixture checks it directly. Confirm `(1, 1, T, T)` with `0.0` on and
below the diagonal and `-inf` above, then move to `stable_softmax`, then the
padding adapter, and only then `scaled_dot_product_attention`.

The learner owns this implementation. The instructor explains, reviews, and
records; requests to work through the code together do not authorize the
instructor to write it. Ask for a hint rather than a solution.

## Session 3 Handoff

Session 3 needs exactly two things from here, unchanged:

- `scaled_dot_product_attention(q, k, v, mask)` operating on
  `(B, H, T, d_head)` and returning `(out, weights)`.
- Both mask builders, combinable by addition, so a `(B, T)` `attention_mask` and
  a `(T, T)` causal mask can be mixed in one call.

Session 3 adds the `W_Q`/`W_K`/`W_V`/`W_O` projections, the head split and
merge, and the check that multi-head matches a manual per-head loop.

## Understanding Check - 2026-09-29

Seven questions after the checks went green. Recorded honestly, in the Week 9
style: what was produced unprompted, and what had to be supplied.

| Topic | Learner evidence | Guidance and assessment limit |
| --- | --- | --- |
| `sdpa(x, x, x)` on raw block input | Caught that there is no causal mask, so later positions are visible. Sensed the axes were wrong | Did not reach the point: with `q` and `k` the same tensor, `score(i,j) = x_i·x_j = score(j,i)`, so the weight matrix is symmetric and relevance is forced to be mutual - the exact limitation Session 1 established. Also did not name that axis 1 is read as the head axis. |
| Why max-subtraction is a no-op | Correct: the same factor divides every term, so it cancels. `exp(x - m) = exp(x)/exp(m)`. | Wording fix supplied: subtracting is a *shift* in logits; multiplying is a *scaling*, and that one does change the softmax. Also answered the follow-up correctly - only extreme-magnitude inputs expose a missing max-subtraction. |
| A padding position as query | Recognised it never carries content. | Attributed safety to "padding never appears at inference", which is wrong for batched generation. Missed all three containment mechanisms: the padding mask blocks its column in every layer, and `loss_mask` excludes its prediction. |
| Is the padding mask needed in layer 2 | Correct conclusion - every layer. | Mechanism supplied: layer 2 builds its keys and values from layer 1's output at that position, which is a plausible-looking average of real tokens, not noise. |
| Missing causal mask, trained decoder | Correct mechanism: the model reads the token it must predict, so the loss falls. | Missed the decisive half - the loss is *lower* than a correct model's, so the curve cannot flag it. It shows up at generation. |
| Left padding | Partly: kept the padding mask blocking padding. | Missed that `sdpa` needs no change at all, and that the causal mask never referenced padding. Also missed that with left padding the fully blocked row becomes routine rather than theoretical. |
| Can a fully blocked row occur here | Not answered; after narrowing to two facts ("every row may see column 0" and "column 0 is always real under right padding"), answered `idk`. | Answer supplied in the session: no, it cannot occur in this pipeline - but the contract requires the repair for any valid input, and left padding or batched inference makes it live. |

**Pattern.** Consistent with Session 1: strong on "what does this do" and on
mechanisms already walked through; thinner on inferences that require combining
two facts already known, and on questions about what the *evidence* licenses
rather than what the code does. The mask row-versus-column argument was produced
unprompted, which is the Session 2 finish-condition item 5. No repeat quiz is
scheduled; Session 3's head-split and multi-head checks are the natural place to
retest the reasoning.
