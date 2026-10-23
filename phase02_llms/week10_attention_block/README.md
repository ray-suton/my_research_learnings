# Week 10 - Transformer from Scratch I: Attention Block

## Purpose

Build the core Transformer block by hand and understand every piece of it:
scaled dot-product attention, causal masking, multi-head attention, residual
connections, layer normalization, and a position-wise MLP. Week 11 assembles
these pieces into a mini-GPT, so the shapes and mask conventions frozen here are
what the training loop inherits.

## Current Checkpoint

Initialized on **2026-09-28** as a documentation scaffold. The scaffold and
frozen contract remain in this folder; **all five sessions have recorded
evidence and the week is closed (2026-10-05).**

**Session 1 closed on 2026-09-28** as a guided conceptual checkpoint. The
[discussion record](session1.md#discussion-record) separates learner reasoning
from instructor explanation; no independent mastery or implementation is claimed.

**Session 2 started on 2026-09-29.** [attention.py](attention.py) is the learner
scaffold carrying the four frozen signatures, and
[tests/test_attention.py](tests/test_attention.py) is the fixture spec - 14
checks, red by design because the bodies are unimplemented.

**Session 2 closed on 2026-09-29.** All four functions plus `main()` are
implemented and **15/15 fixture checks pass**; the run, the smoke summary, and
two defects found along the way are recorded in [RUN_LOG.md](RUN_LOG.md).

**Session 3 opened on 2026-09-30.** [session3.md](session3.md) carries the
multi-head contract, the shape walkthrough, the checklist, and the finish
condition. **Session 3 closed the same day:** `MultiHeadAttention` is
implemented in [multihead.py](multihead.py) and **9/9 fixture checks pass**,
including equivalence against an independently written per-head reference.
`attention.py` was not touched and still passes 15/15.

**Session 4 closed on 2026-10-01.** [session4.md](session4.md) records the
understanding checks, learner implementation, review, and verification.
`block.py` now contains the hand-written layer norm, GELU MLP, pre-norm
residual block, and smoke runner. **11/11 block checks pass**, and the smoke run
reports `(2, 8, 32)` input/output shapes and 12,704 parameters. `attention.py`
and `multihead.py` stayed untouched.

**Session 5 closed on 2026-10-05.** Every suite was re-run rather than trusted:
15/15 attention, 9/9 multi-head, 11/11 block, and 9/9 on the prepared LM
dataset. [session5.md](session5.md) records the review and the Week 11 handoff;
[LAB_NOTE.md](LAB_NOTE.md) records reproduction, what the checks establish and
do not establish, and the limitations. The unprepared paths are now the Week 11
training loop, not this block.

Two workstreams run in this week:

| Track | Where it lives | Status |
| --- | --- | --- |
| Attention block | Sessions 1-5 in this folder | **Closed 2026-10-05**; 15/15 attention, 9/9 multi-head, 11/11 block |
| Prepared LM dataset | [Week 9 Session 4](../week09_tokenization_text_data/session4.md) carry-over, reassigned to Week 10 on 2026-09-28 | **Implemented 2026-10-05**; 9/9 fixture checks, real-corpus run recorded |

**Week 10's deliverable is the tested Transformer block and the prepared LM
dataset** ([roadmap](../../ai_research_roadmap_codex.md)). Week 11's mini-GPT
training cannot run until both are done.

## Five-Session Plan

Treat these as roughly 90-minute learning sessions; adapt the pacing to the
learner's progress rather than forcing a calendar deadline.

| Session | Work | Evidence to retain |
| --- | --- | --- |
| 1 | Attention concepts: queries, keys, and values; scaled dot-product; softmax; why the `1/sqrt(d_head)` scale; causal versus padding masks; positional encodings, the residual stream, layer norm, and the MLP at reading level | Closed 2026-09-28: [discussion record](session1.md#discussion-record) separating learner reasoning from instructor explanation |
| 2 | Implement `stable_softmax`, both additive masks, and scaled dot-product attention by hand | Closed 2026-09-29: **15/15 fixture checks pass**, including the future-token perturbation causality test; run and smoke summary in [RUN_LOG.md](RUN_LOG.md) |
| 3 | Multi-head attention: split, per-head math, merge, output projection | Closed 2026-09-30: **9/9 fixture checks pass**, including equivalence against a hand-written per-head reference; run and smoke summary in [RUN_LOG.md](RUN_LOG.md) |
| 4 | Residual connections, layer normalization, and the position-wise MLP; assemble the block | Closed 2026-10-01: **11/11 fixture checks pass**, with smoke output recorded in [RUN_LOG.md](RUN_LOG.md) |
| 5 | Review the block, write the lab note, and hand off to Week 11 | Closed 2026-10-05: [session5.md](session5.md) review record and [LAB_NOTE.md](LAB_NOTE.md) with reproduction steps, limitations, and the Week 11 handoff |

## Planned Scope

Implement every piece by hand in PyTorch: no `torch.nn.MultiheadAttention`, no
`torch.nn.TransformerEncoderLayer`, and no
`torch.nn.functional.scaled_dot_product_attention`. Reaching for the library is
the fastest way to make this week's exercise pointless.

The week's boundary is deliberate:

- **In scope:** multi-head self-attention, causal masking, padding masking, the
  position-wise MLP, residual connections, and layer normalization.
- **Read now, implement in Week 11:** positional encodings. The roadmap places
  positional embeddings with the mini-GPT's embedding layer, and a Transformer
  block itself is position-agnostic.
- **Out of scope:** training, sampling, the LM head, cross-entropy loss, and
  cross-attention.

Start from the frozen tensor and mask contract in the
[experiment plan](EXPERIMENT_PLAN.md). Small fixture sizes keep the arithmetic
checkable by hand; the mask conventions are the part Week 11 inherits unchanged.

## Files

- [session1.md](session1.md): attention concepts and the shape contract; closed 2026-09-28.
- [session5.md](session5.md): Session 5 review, verification, and Week 11 handoff; closed 2026-10-05.
- [LAB_NOTE.md](LAB_NOTE.md): week lab note - reproduction, what the checks establish, limitations, handoff.
- [session2.md](session2.md): closed attention contract, hand-computed illustration, checklist, and finish condition.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): frozen tensor/mask contract, measurement definition, and failure conditions.
- [RUN_LOG.md](RUN_LOG.md): actual actions, commands, and outcomes.
- [RESULTS.md](RESULTS.md): measured evidence for Sessions 2-4; the Session 5 closure section remains pending.
- [attention.py](attention.py): hand-written `stable_softmax`, both mask builders, and scaled dot-product attention; Session 2 is closed.
- [tests/test_attention.py](tests/test_attention.py): the attention fixture spec, **15/15 checks passing**.
- [configs/README.md](configs/README.md): fields to record before a run.
- [tests/README.md](tests/README.md): planned behavioral checks across the whole week; the Session 2 subset now exists as `tests/test_attention.py`.
- [results/README.md](results/README.md): artifact and provenance policy.

[session5.md](session5.md) and [LAB_NOTE.md](LAB_NOTE.md) close the week.

## Carry-Over: The Prepared LM Dataset

Reassigned to this week on 2026-09-28; Week 9 delivers the tokenizer and its
token streams. The contract, checklist, and finish condition live in
[Week 9 Session 4](../week09_tokenization_text_data/session4.md), and the
implementation is [lm_data.py](../week09_tokenization_text_data/lm_data.py) with
its fixture spec at
[tests/test_session4_lm_data.py](../week09_tokenization_text_data/tests/test_session4_lm_data.py).
That contract is not duplicated here.

What this week added on top of the original scaffold, all closed 2026-10-05:

- `build_split_stream` closed Week 9's redrawn deliverable first, because the
  windows are cut from that stream.
- `build_source_windows` and `make_lm_batch`, the fixture gate, and the
  `RUN_LOG.md` / `RESULTS.md` record were this week's half.
- The padding mask the attention block consumes is the same `attention_mask` the
  LM data produces. Keep the padding mask and the causal mask distinct - that is
  the trap Week 9 Session 1 already flagged.

The prepared LM dataset landed before Session 5's review: [lm_data.py](../week09_tokenization_text_data/lm_data.py)
is implemented, its **9/9 fixture checks pass**, and the real-corpus run is
recorded in [Week 9 RUN_LOG.md](../week09_tokenization_text_data/RUN_LOG.md).
Both parts of the deliverable are therefore done.

## Data And Environment Prerequisites

Nothing new is needed for the block itself. The frozen decisions from Week 9 are
reused:

- The training-only tokenizer artifact and the pinned WikiText-2 corpus under
  `AI_ROADMAP_DATA`, per the [data policy](../../DATA_PLAN.md).
- `T = 128`, stride `T`, PAD 0, EOS 1, `IGNORE_INDEX -100`.
- PyTorch is already available from the Week 1 environment; this week adds no
  dependency.

Session 1 is conceptual and needs no corpus access. Sessions 2-4 start from
hand-written fixtures; corpus data is only needed for the carry-over track.

## Definition Of Done

- Scaled dot-product attention and causal masking are implemented by hand.
- Multi-head attention is implemented by hand and matches a manual per-head reference.
- Residual connections, layer normalization, and the position-wise MLP are implemented and assembled into a block.
- Shape, mask, causality, and gradient checks pass and are recorded in `RUN_LOG.md`.
- The prepared LM dataset ([Week 9 Session 4](../week09_tokenization_text_data/session4.md)) is implemented and its fixture checks are green.
- A lab note explains what the block does, what the checks do and do not establish, and what Week 11 inherits.

Next learning action: begin Week 11 - the mini-GPT. Monday is the conceptual
session on token embeddings, positional embeddings, the decoder stack, the LM
head, and next-token loss; the frozen block contract and the prepared dataset in
this folder are its inputs. See the [Week 11 workspace](../week11_mini_gpt/README.md).
