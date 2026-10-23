# Week 9 - Tokenization and Text Data

## Purpose

Build the text input pipeline that later attention and mini-GPT projects can
reuse. Learn how a tokenizer turns text into discrete IDs, how its vocabulary
affects sequence length, and how to prepare language-modeling examples without
mixing training and evaluation data.

## Current Checkpoint

Initialized on **2026-09-17**. [Session 1](session1.md) closed on
**2026-09-21** as a guided conceptual checkpoint, with the learner's reasoning,
corrections, and assessment limits recorded.
[Session 2](session2.md) started on **2026-09-22** and, as of **2026-09-23**,
has a working README-trained prototype: [tokenizer.py](tokenizer.py) implements
all four functions against the `tokenizers` library, runs the full
reader -> fit -> encode/decode -> save/reload path, and passed an instructor
fresh-process reload check (vocab 651, PAD 0 / EOS 1). The root README is the
practice input; WikiText-2 remains the planned measured corpus.

Session 2's coding work closed on **2026-09-24**: the reserved-literal
`<pad>`/`<eos>` policy check, the empty-corpus guard, and the full ordinary-text
probe set were added and run. Corpus preparation and the measured comparison
remain Session 3 work.

[Session 3](session3.md) was scaffolded and then **executed on 2026-09-24**. It
is the first measured Week 9 result: WikiText-2 was fetched at revision
`b08601e0`, a custom tokenizer was refitted on training text only (2,222 records,
rows 1-3,465, 1,048,468 bytes) to the full 2,048-token target, and both
tokenizers encoded the same 2,461 validation records. The custom tokenizer used
382,916 tokens versus GPT-2's 247,289 (about 1.55x), with no unknown tokens for
either. The cross-split duplicate overlap was audited and a sensitivity refit
moved the total by +0.02%. The test split was not encoded. Session 4's LM chunks,
targets, and masks were implemented on **2026-10-05**; the prepared LM dataset
was Week 10's deliverable outright (redrawn 2026-09-28) and closed there.

[Session 4](session4.md) was scaffolded on **2026-09-25** and then **deferred by
learner decision the same day**, and **rescheduled on 2026-09-26 to be split
across Weeks 9 and 10**. It carries the LM example contract, its task checklist,
its finish condition, and [lm_data.py](lm_data.py) with its fixture spec at
[tests/test_session4_lm_data.py](tests/test_session4_lm_data.py).
**Both halves were implemented and verified on 2026-10-05, with 9/9 fixture
checks passing.** The stream half (`build_split_stream`) closed Week 9's redrawn
deliverable; the example half (`T + 1` windows, one-shifted targets,
`attention_mask`, `loss_mask`) closed as **Week 10's deliverable** - the prepared
LM dataset. Week 11 training is unblocked.

[Session 5](session5.md) closed on **2026-09-25** with the
[Week 9 lab note](LAB_NOTE.md): results reviewed, limitations recorded, and the
Week 10 handoff defined. Documentation only - no code ran and nothing new was
measured.

**Week 9's deliverable was redrawn on 2026-09-28** to *a training-only tokenizer
and the per-split token streams it produces*, not the whole prepared LM dataset.
**It is now complete:** the tokenizer was measured in Session 3, the per-split
streams were implemented on 2026-10-05, and the prepared LM dataset closed under
Week 10 the same day.

Phase 1, including the Fashion-MNIST GAN extension, remains closed. This was
the first project in [Phase 2](../README.md); [Week 11](../week11_mini_gpt/README.md) is next.

## Five-Session Plan

Treat these as roughly 90-minute learning sessions; adapt the pacing to the
learner's progress rather than forcing a calendar deadline.

| Session | Work | Evidence to retain |
| --- | --- | --- |
| 1 | Tokens versus embeddings; BPE, WordPiece, unigram; special tokens; padding, truncation, and packing | Closed 2026-09-21: guided discussion, corrections, and working data contract in [session1.md](session1.md) |
| 2 | Complete the README-trained BPE prototype using the [task checklist](session2.md#task-checklist) | Successful fit, text checks, reload equivalence, a short run/observation note, and code walkthrough; [finish condition](session2.md#finish-condition). No learner locking/hashing task. |
| 3 | Prepare/verify WikiText-2, refit on its declared training subset, then compare with a fixed pretrained tokenizer on identical held-out text | Closed 2026-09-24: pinned revision and split counts, declared fitting subset, 2,048-token training-only artifact, token counts and length distributions, overlap audit, failure cases |
| 4 | Build split-specific token streams and fixed-length next-token examples | Closed 2026-10-05: stream half implemented against Week 9's redrawn deliverable, example half closed under Week 10; **9/9 fixture checks pass**, real-corpus counts in [RUN_LOG.md](RUN_LOG.md) |
| 5 | Review results and write a lab note about context length and data choices | Closed 2026-09-25: [LAB_NOTE.md](LAB_NOTE.md) with reproduction steps, limitations, and the Week 10 handoff |

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
- [session2.md](session2.md): closed README-tokenizer lesson, input/output contract, and run reference.
- [session3.md](session3.md): active session - WikiText-2 corpus preparation, training-only refit, and pretrained-reference comparison contract.
- [session4.md](session4.md): deferred session - split-specific streams, one-shifted targets, attention/loss masks, and the fixture checks.
- [session5.md](session5.md): closed review session - what was reviewed, the closing verdict, and the Week 10 handoff.
- [LAB_NOTE.md](LAB_NOTE.md): the Week 9 lab note - measured results, context-length analysis, limitations, and reproduction steps.
- [tokenizer.py](tokenizer.py): learner-owned byte-level BPE reader, fitting, checks, and save/reload path.
- [lm_data.py](lm_data.py): implemented LM example builder - streams, windows, one-shifted targets, and both masks; 9/9 fixture checks pass.
- [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md): corpus, comparisons, metrics, and tensor contract.
- [RUN_LOG.md](RUN_LOG.md): actual actions, commands, and outcomes.
- [RESULTS.md](RESULTS.md): measured evidence, initially pending.
- [configs/README.md](configs/README.md): fields to record before a run.
- [tests/README.md](tests/README.md): planned behavioral checks plus the implemented `test_session4_lm_data.py` (9/9 checks passing).
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

Rechecked read-only on **2026-09-24** for the Session 3 scaffold: the registry's
WikiText-2 raw and processed paths are still absent, and no WikiText-2 entry
exists in the Hugging Face hub or datasets caches. The cached
`openai-community/gpt2` tokenizer, by contrast, loads offline with vocabulary
50,257 and an exact round trip, so only the corpus side of the comparison needs
a download.

## Definition Of Done

- A training-only tokenizer is saved and reloads with the same behavior.
- Comparison uses the same validation text and clearly defined metrics.
- Train/validation/test membership survives tokenization and packing.
- **Week 10 deliverable:** next-token targets, padding, and masks pass meaningful
  checks. **Done 2026-10-05; 9/9 fixtures green.**
- Commands, source/data/tokenizer identities, and limitations are recorded.
- A lab note explains how tokenization changes context usage and what the
  comparison cannot establish. Record learner discussion honestly.

Next learning action: begin [Week 11](../week11_mini_gpt/README.md) - the
mini-GPT. Week 9 and Week 10 are both closed: the tokenizer is measured, the
per-split streams are implemented, and the prepared LM dataset is green. The
frozen data contract Week 11 consumes is
[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md#lm-example-contract).
