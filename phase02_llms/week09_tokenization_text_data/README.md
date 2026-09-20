# Week 9 - Tokenization and Text Data

## Purpose

Build the text input pipeline that later attention and mini-GPT projects can
reuse. Learn how a tokenizer turns text into discrete IDs, how its vocabulary
affects sequence length, and how to prepare language-modeling examples without
mixing training and evaluation data.

## Current Checkpoint

Initialized on **2026-09-17**. The documentation scaffold is complete;
[Session 1](session1.md) is ready to begin. No understanding assessment,
tokenizer implementation, corpus preparation, executable test, or measured
comparison has completed for Week 9.

Phase 1, including the Fashion-MNIST GAN extension, remains closed. This is
the first active project in [Phase 2](../README.md).

## Five-Session Plan

Treat these as roughly 90-minute learning sessions; adapt the pacing to the
learner's progress rather than forcing a calendar deadline.

| Session | Work | Evidence to retain |
| --- | --- | --- |
| 1 | Tokens versus embeddings; BPE, WordPiece, unigram; special tokens; padding, truncation, and packing | Concept discussion and agreed corpus/data contract |
| 2 | Trace a tiny BPE example, then train and save a small tokenizer on training text only | Vocabulary/settings, encode/decode examples, reload checks |
| 3 | Compare the custom tokenizer with a fixed pretrained tokenizer on identical held-out text | Token counts, length distributions, round trips, failure cases |
| 4 | Build split-specific token streams and fixed-length next-token examples | Shapes, masks, target alignment, split/provenance checks |
| 5 | Review results and write a lab note about context length and data choices | Reproduction instructions, limitations, Week 10 handoff |

## Planned Scope

Start with a small byte-level BPE tokenizer and the existing WikiText-2 raw
dataset entry. Use a frozen GPT-2 tokenizer as a descriptive reference in
Session 3. These are proposed working defaults, not completed experiments;
the [experiment plan](EXPERIMENT_PLAN.md) records the decisions to freeze
before execution. Understanding WordPiece and unigram does not require
implementing all three algorithms this week.

The deliverable is a saved tokenizer and reproducible LM data preparation
with checks and a short comparison. Transformer training belongs to later weeks.

## Files

- [session1.md](session1.md): starting concepts and discussion prompt.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): corpus, comparisons, metrics, and tensor contract.
- [RUN_LOG.md](RUN_LOG.md): actual actions, commands, and outcomes.
- [RESULTS.md](RESULTS.md): measured evidence, initially pending.
- [configs/README.md](configs/README.md): fields to record before a run.
- [tests/README.md](tests/README.md): planned behavioral checks; no test code yet.
- [results/README.md](results/README.md): artifact and provenance policy.

## Data And Environment Prerequisites

Use `AI_ROADMAP_DATA`, falling back to
`/home/rui.gao/data/ai_research_roadmap`, per the [data policy](../../DATA_PLAN.md).
The [registry](../../data_registry/datasets.yaml) already includes `wikitext2`.
Its descriptive raw path and processed path were absent at initialization;
an alternative Hugging Face cache has not been ruled out. Inspect the actual
cache and installed packages before choosing a preparation command.

The existing preparation helper supports WikiText-2, but its `--dry-run`
still creates directories and a manifest. Its current download path does not
pin a dataset revision or export prepared LM chunks. It has not been run for
this initialization; any later preparation needs recorded revision/fingerprint
and artifact provenance. No dependency or dataset changes were made here.

## Definition Of Done

- A training-only tokenizer is saved and reloads with the same behavior.
- Comparison uses the same validation text and clearly defined metrics.
- Train/validation/test membership survives tokenization and packing.
- Next-token targets, padding, and masks pass meaningful checks.
- Commands, source/data/tokenizer identities, and limitations are recorded.
- A lab note explains how tokenization changes context usage and what the
  comparison cannot establish. Record learner discussion honestly.

Next learning action: open [Session 1](session1.md).
