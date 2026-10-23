# Week 11 - Transformer from Scratch II: mini-GPT

## Purpose

Train a tiny decoder-only language model from scratch, using the pieces the
earlier weeks already froze: the Week 9 tokenizer and per-split streams, the
Week 10 hand-written `TransformerBlock`, and the prepared LM dataset
(`T + 1` windows, one-shifted targets, `attention_mask`, `loss_mask`).

The week adds exactly four model pieces - token embedding, positional embedding,
a stack of decoder blocks, and an LM head - plus a next-token training loop, an
overfit check, one real HPC run, and a loss curve with generated samples.

## Current Checkpoint

**Initialized on 2026-10-05 as a documentation scaffold.** This folder contains
the five-session plan, the contract and decisions to freeze, and the empty
records to fill in. No model code, training run, loss curve, or sample exists
yet. Week 11's own plan is unexecuted; every number in
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) is a proposed default, not a result.

Week 11 is unblocked because both inputs closed on 2026-10-05:

- [Week 9](../week09_tokenization_text_data/README.md) - training-only tokenizer
  (2048 tokens, PAD 0 / EOS 1) and the per-split token streams; 9/9 fixture
  checks green.
- [Week 10](../week10_attention_block/README.md) - tested `TransformerBlock`
  (15/15 attention, 9/9 multi-head, 11/11 block) and the prepared LM dataset.

## What Week 11 Adds

```text
token ids  (B, T)
  |
  v
[ token embedding ]        vocab 2048 -> d_model
  +
[ positional embedding ]   T = 128 positions
  |
  v
[ N x TransformerBlock ]   Week 10, unchanged and position-agnostic
  |
  v
[ final layer norm ]       decision below
  |
  v
[ LM head ]                d_model -> vocab 2048   (tied to token embedding?)
  |
  v
logits (B, T, 2048) -> cross-entropy vs one-shifted targets, masked by loss_mask
```

Week 10's block is deliberately position-agnostic, so Week 11 is where order
enters the model and where the residual stream is read out to vocabulary logits.
That read-out is the only new interface, and the only place the shift and the
masks can be wired wrong.

## Five-Session Plan

Treat these as roughly 90-minute learning sessions; adapt the pacing to actual
progress rather than a calendar deadline.

| Session | Work | Evidence to retain |
| --- | --- | --- |
| 1 | Study embeddings, positional embeddings, the decoder block, the LM head, and next-token loss; freeze the decisions below | [session1.md](session1.md) discussion record and the frozen contract in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) |
| 2 | Implement the mini-GPT: embedding, positions, `N` blocks, final norm, LM head, and the masked next-token loss | `mini_gpt.py` with a red-to-green fixture spec in [tests/](tests/README.md) and a recorded smoke run |
| 3 | Overfit one tiny batch and prove the model can memorize it | A recorded overfit run whose loss collapses and whose greedy generation reproduces the training window |
| 4 | Train on the WikiText-2 splits on HPC | SLURM job file, `RUN_LOG.md` entry with command/exit/hardware, checkpoint and log paths under `AI_ROADMAP_DATA` |
| 5 | Plot train/validation loss, generate samples, write the lab note | Loss-curve figure, fixed-prompt samples with settings, [LAB_NOTE.md](LAB_NOTE.md), Week 12 handoff |

Session 1 is conceptual and needs no code. Sessions 2-3 run locally on CPU or a
single GPU. Session 4 is the only HPC step and is the one that needs the
[data policy](../../DATA_PLAN.md) rule about checkpoints living outside Git.

## Planned Scope

**In scope:** token embedding, learned positional embedding, `N` stacked Week 10
decoder blocks, a final layer norm, a vocabulary LM head, weight tying, masked
next-token cross-entropy, AdamW with warmup and cosine decay, one overfit check,
one HPC training run, a loss curve, and sample generation.

**Read now, implement later:** sampling strategies (temperature, top-k, top-p)
and perplexity reporting belong to Week 12. Week 11 generation uses a fixed,
simple decoding rule stated in the run record.

**Out of scope:** any attention rewrite, cross-attention, KV cache, mixed
precision, distributed training, data mixtures, instruction tuning, and
evaluation beyond loss.

The hard rule from Week 10 still applies: the Transformer block is hand-written,
so do not swap in `torch.nn.TransformerEncoderLayer` or
`torch.nn.MultiheadAttention` to make the model "work". `torch.nn.Embedding`,
`torch.nn.Linear`, and `torch.nn.CrossEntropyLoss` are allowed.

## Decisions To Freeze In Session 1

Each decision below has a recommended default and a reason. Session 1 confirms
or overturns them, and the confirmed set is copied into
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) before any code is written.

| Decision | Recommended default | Why | Rejected alternative |
| --- | --- | --- | --- |
| Positional information | Learned absolute embeddings, `nn.Embedding(T, d_model)` | Matches GPT-2, simplest to implement and inspect, and the roadmap places positional embeddings with the mini-GPT embedding layer | Sinusoidal encodings - no learned parameters, but more arithmetic to write and harder to debug against a known-good reference |
| Where positions are added | Added to the token embedding **before** the first block | Keeps the block position-agnostic and gives every layer the same position signal | Adding inside each block - duplicates what the block should not know |
| Weight tying | Tie the LM head to the token embedding | Saves `vocab x d_model` parameters and is the standard GPT-2 choice; with a 2048-token vocabulary the saving is real | Untied head - simpler to reason about, but more parameters and no accuracy benefit at this scale |
| Final normalization | Layer norm after the last block, before the LM head | Pre-norm stacks need the final norm for a well-scaled read-out | No final norm - runs, but the logits scale depends on residual-stream growth |
| Loss reduction | Cross-entropy with `reduction="none"`, then `loss_mask` weighted mean | Padding positions must not contribute; a plain mean silently averages over padding | Plain mean over all positions - silently trains on padding |
| Initialization | Normal `(0, 0.02)` for embeddings and linears; residual-output projections scaled by `1/sqrt(2 * n_layers)` | Standard GPT-2 init; the scaling keeps residual variance from growing with depth | Default PyTorch init - works at depth 1-2, drifts at depth 4+ |
| Final activation | No softmax in the model; logits only | `CrossEntropyLoss` applies log-softmax internally | Softmax before the loss - double-normalizes and breaks the gradient |
| Dropout | `0.1`, disabled for the Session 3 overfit check | Regularizes the real run; disabled so the overfit test can actually memorize | Always on - the overfit check would then not be a clean memorization test |

## Files

- [session1.md](session1.md): the conceptual session and the decisions to freeze.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): the frozen model, loss, optimizer, and measurement contract.
- [RUN_LOG.md](RUN_LOG.md): initialized; actual actions, commands, and outcomes go here.
- [RESULTS.md](RESULTS.md): initialized with no claims; measured evidence goes here.
- [LAB_NOTE.md](LAB_NOTE.md): to be written in Session 5.
- [configs/README.md](configs/README.md): fields to record per run.
- [tests/README.md](tests/README.md): planned behavioral checks.
- [results/README.md](results/README.md): artifact and checkpoint policy.

## Data And Environment Prerequisites

- `AI_ROADMAP_DATA` per the [data policy](../../DATA_PLAN.md), default
  `/home/rui.gao/data/ai_research_roadmap`.
- The Session 3 tokenizer artifact at
  `<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session3_wikitext_trainonly/tokenizer.json`,
  vocabulary 2048.
- The pinned WikiText-2 splits (`b08601e0`) and the prepared dataset from
  [lm_data.py](../week09_tokenization_text_data/lm_data.py).
- PyTorch is already installed. No new dependency is needed.
- HPC: reuse the [GPU smoke-test template](../../scripts/slurm/gpu_smoke_test.slurm)
  style (`--partition=gpu`, `--gres=gpu:1`); Session 4 writes the training job
  file. Checkpoints and logs live under `AI_ROADMAP_DATA`, never in Git.

## Definition Of Done

- The mini-GPT is implemented by hand except for `nn.Embedding`, `nn.Linear`,
  and the loss function, and its fixture checks pass.
- The overfit check memorizes one tiny batch, shown by a collapsing loss and
  greedy reproduction of the training window.
- One HPC training run completes with command, settings, hardware, and exit
  status recorded.
- A train/validation loss curve and readable generated samples are recorded with
  their settings.
- A lab note states what was and was not established, and hands off to Week 12.
- No quality claim is made that was not measured.

Next learning action: begin [Session 1](session1.md) - the conceptual study of
embeddings, positions, the LM head, and next-token loss - and freeze the decision
table above before writing any model code.
