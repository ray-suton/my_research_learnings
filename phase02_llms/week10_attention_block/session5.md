# Session 5 - Block Review, Lab Note, And Week 11 Handoff

## Status

**Closed 2026-10-05.** Instructor-led review and verification pass, with the
learner's carry-over data-contract discussion recorded below. Every command in
this session was re-run on 2026-10-05; no earlier result is reused as if it were
new, and no independent block-level mastery is claimed.

[Session 4](session4.md) closed on 2026-10-01 with **11/11** block checks
passing. [Session 3](session3.md) closed on 2026-09-30 with **9/9** multi-head
checks, and [Session 2](session2.md) on 2026-09-29 with **15/15** attention
checks.

## Where This Sits

```text
Week 9  tokenizer artifact + per-split rows
Week 10 [ THIS WEEK ]   x -> TransformerBlock -> x        (position-agnostic)
Week 11  token embedding + positions -> N x block -> LM head -> next-token loss
```

Week 10's block takes `(B, T, d_model)` and returns `(B, T, d_model)`. It is
deliberately position-agnostic, so Week 11 is the component that has to inject
position information and then read the residual stream out to vocabulary logits.

Two deliverables close here, and Week 11 cannot start training until both do:

1. The tested Transformer block (Sessions 2-4).
2. The prepared LM dataset - the carry-over half of
   [Week 9 Session 4](../week09_tokenization_text_data/session4.md) - whose
   `T + 1` windows, one-shifted targets, `attention_mask`, and `loss_mask` are
   now implemented and green.

## What Sessions 1-4 Left

| Piece | Where | State at Session 5 entry |
| --- | --- | --- |
| Attention concept checkpoint | [session1.md](session1.md) | Closed 2026-09-28; guided, no independent mastery claimed |
| `stable_softmax`, both masks, scaled dot-product attention | [attention.py](attention.py) | Implemented; 15/15 |
| `MultiHeadAttention` | [multihead.py](multihead.py) | Implemented; 9/9 |
| `LayerNorm`, `FeedForward`, `TransformerBlock` | [block.py](block.py) | Implemented; 11/11 |
| Prepared LM dataset | [lm_data.py](../week09_tokenization_text_data/lm_data.py) | Implemented; 9/9 |

## What Session 5 Did

1. **Re-ran all four fixture suites** from the repository root and recorded the
   exit status of each.
2. **Re-ran the carry-over data pipeline on the real corpus** and recorded the
   per-split counts, rather than trusting the earlier run.
3. **Reviewed the block sources against the week's "hand-written only" rule** -
   a read-through plus a forbidden-call scan, not a proof.
4. **Verified the Week 11 inheritance contract** - the shapes, the mask
   conventions, and the frozen constants that the training loop will consume.
5. **Wrote this session record and [LAB_NOTE.md](LAB_NOTE.md).**

The learner contribution this session was conceptual, on the carry-over data
contract: what one training example is, why windows may cross row boundaries
inside a split, why no attention exists across windows or batch rows, and what
the single EOS token per record is for. Those positions are recorded under
"Understanding Check" below; the block-internal review was instructor-led.

## Verification Performed - 2026-10-05

All four commands run from the repository root, Python 3.13, PyTorch 2.11.0,
CPU, seed 0.

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_multihead.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py
.venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
```

| Suite | Result | Exit |
| --- | --- | --- |
| Attention | 15/15 | 0 |
| Multi-head | 9/9 | 0 |
| Block | 11/11 | 0 |
| Prepared LM dataset | 9/9 | 0 |

Block smoke run, `.venv/bin/python phase02_llms/week10_attention_block/block.py`:

```text
x               (2, 8, 32)
out             (2, 8, 32)
parameters      12704
mean |out - x|  0.259858
```

Carry-over data run, with `AI_ROADMAP_DATA` pointed at the pinned corpus:

```text
train       rows= 36718 records= 23767 stream= 3682072 windows= 28767
validation  rows=  3760 records=  2461 stream=  385377 windows=  3011
test        rows=  4358 records=  2891 stream=  436253 windows=  3409
```

The validation stream length (385,377) equals the Session 3 measured token count
(382,916) plus one EOS per nonempty record (2,461) - an independent cross-check
that the stream builder adds exactly one EOS per record and nothing else.

Forbidden-call scan over `attention.py`, `multihead.py`, and `block.py`: the only
matches are docstring warnings and the learner's own function name
`scaled_dot_product_attention`. No `torch.nn.MultiheadAttention`,
`torch.nn.TransformerEncoderLayer`,
`torch.nn.functional.scaled_dot_product_attention`, or `torch.nn.LayerNorm` call
appears in the module sources.

## Understanding Check

Recorded as it actually happened, with the boundary marked.

Settled with the learner during this thread (carry-over data contract):

- One training example is one `T + 1` window, not one row and not one batch. A
  window may contain the tail of one record and the head of the next.
- Windows are cut with stride `T`, so window `k` and window `k + 1` share exactly
  one token - the token that is window `k`'s last target and window `k + 1`'s
  first input.
- Attention contracts only over `T` inside a single window. There is no
  cross-window and no cross-row attention, because a window is the unit the model
  sees; the window boundary is invisible to the model.
- Batch rows are independent examples packed for throughput. Row `b` never
  attends to row `b'`.
- The one EOS per nonempty record is a **record/formatting** marker, not a
  sentence marker. Heading rows prove it: a heading ends with EOS and has no
  terminal punctuation.
- The `attention_mask` and the `loss_mask` must disagree at exactly one position
  per padded window: the last real input, whose target is padding.

Open items carried into Week 11, to be answered by the learner before the
training loop is written:

- Why the block must not include positional encoding, and what breaks if it does.
- Why `targets` are pre-shifted here, and what a model API that shifts labels
  internally would do to them.
- Why the final real input position must not carry a loss when its target is
  padding.

No independent block-internal mastery is claimed; the block review was
instructor-led.

## Limitations

- **Nothing was trained.** Every number recorded this week is a correctness
  check, a shape, a parameter count, or a token count. No perplexity, accuracy,
  or generation-quality claim exists for Week 10.
- **Fixture scale.** The block checks run at `B=2`, `T=8`, `d_model=32`,
  `n_heads=4`, `d_ff=128`. Correctness at that size does not prove numerical
  behavior at `T=128` or at depth.
- **The block is not a model.** There is no embedding, no positional encoding,
  no LM head, and no loss in this week's code.
- **Prepared-LM-dataset limitation (documented, not live).** A row containing a
  literal `<eos>` token would produce an internal EOS id and split one record
  into two. The pinned WikiText-2 splits contain zero such rows, so this is a
  contract edge case rather than an observed defect.
- **`lone_trailing_token` is unexercised on the real corpus.** All three splits
  happen to end with a full window, so the one-token-leftover branch only has
  fixture coverage.
- **The attention-mask path into the block is only checked at fixture scale**,
  with an additive mask - the padding-convention interaction with real data
  starts in Week 11.

## Week 11 Handoff

What Week 11 inherits unchanged:

```text
tokenizer artifact   AI_ROADMAP_DATA/processed/week09_tokenization_text_data/session3_wikitext_trainonly/tokenizer.json
vocab size           2048
PAD / EOS / IGNORE   0 / 1 / -100
T = STRIDE           128
window payload       129 tokens  (T inputs + 1 shifted target)
batch payload        input_ids, targets, attention_mask, loss_mask - all (B, T)
models consume       (B, T, d_model) at T=128
```

What Week 11 must add on top: token embedding, positional embedding, `N`
copies of `TransformerBlock`, an output normalization if chosen, an LM head, and
cross-entropy next-token loss driven by `loss_mask`.

Two traps carried forward explicitly:

- The targets are **already shifted** by the data layer. A training loop that
  shifts them again is off by one token.
- `attention_mask` and `loss_mask` are different tensors. Feeding `loss_mask`
  to attention, or `attention_mask` to the loss, silently changes what the model
  can see or what it is graded on.

Exit criteria for Week 10: both deliverables implemented, all four suites green,
this record and the lab note written, and the EOS/window contract above
re-verifiable from the commands in this file.
