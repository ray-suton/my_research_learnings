# Session 2 - First Tokenizer Training Path

## Status

Started on 2026-09-22 through guided BPE discussion and a Python scaffold.
**Updated 2026-09-23:** the learner reimplemented [tokenizer.py](tokenizer.py)
against the `tokenizers` library, and all four functions now run the full
reader -> fit -> encode/decode -> save/reload path on the root README. Actual
vocabulary 651 (target 2,048), PAD 0 / EOS 1, and an instructor fresh-process
reload check passed. See the [run record](RUN_LOG.md) and [results](RESULTS.md).

Earlier guided discussion: the instructor supplied the pair-count/merge
examples. The learner identified that different text domains have different
patterns and later connected byte-level coverage to the absence of an unknown
token. The tokenizer/model mismatch was recognized; worse predictions are
possible, not guaranteed.

**Closed 2026-09-24.** The literal `<pad>`/`<eos>` policy check, the empty-corpus
guard, and broader ordinary-text fixture coverage were added and run; the code
walkthrough was covered on 2026-09-23. Every finish-condition row is evidenced.

## First Input

Use the current [root README](../../README.md) as one tiny training record.
Keep case, spacing, punctuation, and line breaks. Use it as it is: no locking,
snapshotting, or hashing is required for this lesson. This practice input is
not a validation set.

The Week 5 README would also work as a tiny practice corpus, but its existing
`corpus.split_examples` strips whitespace and splits sentences. Its character
vocabulary and next-token tensor builder are different stages from BPE fitting.
The new reader should preserve the text; no Week 5 code was copied or changed.

WikiText-2 remains the corpus for the [planned measured experiment](EXPERIMENT_PLAN.md).
Using a README to learn the API does not replace its original splits, provenance,
training-only subset, or the later fixed-reference comparison.

## Function Contracts

| Function | Input | Output |
| --- | --- | --- |
| `read_training_text` | Path to a UTF-8 file | One unchanged Python string |
| `train_tokenizer` | List of training strings and target vocabulary size | Fitted byte-level BPE `Tokenizer` |
| `check_round_trip` | Tokenizer and an ordinary-text fixture | Printed pieces/IDs and assertions; returns `None` |
| `save_and_reload` | Fitted tokenizer and external output path | Reloaded tokenizer after behavior checks |

The model-facing encoding output is a list of integer token IDs. Decoding
returns text. Tokenizer fitting needs no class labels, shifted targets, neural
network, optimizer, or loss. LM chunks and masks belong to Session 4.

The starter targets 2,048 vocabulary entries including reserved tokens, with
the complete 256-byte alphabet and separate `<pad>`/`<eos>`. A tiny corpus may
exhaust eligible merges before reaching the target. Record actual vocabulary
size. Ordinary-text preservation, special-token literal handling, and unchanged
encoding after reload are distinct checks.

## Session 2 Scope

Session 2 closes on a working **README-trained prototype**: read text, fit
byte-level BPE, inspect encoding/decoding, save it, and prove reload equivalence.
The README is entirely a declared practice-training record; it supplies no
held-out performance estimate. Small separate strings are behavioral fixtures.

Before Session 3's measured comparison, prepare the planned WikiText-2 corpus,
verify its identity and original splits, and refit on the declared training-only
subset. Keep that tokenizer/run separate from the README prototype. This is an
explicit handoff prerequisite, not a requirement to finish the current prototype.
The frozen GPT-2 comparison, LM chunks/masks, and final lab note remain in
Sessions 3, 4, and 5 respectively. No RNN or Transformer training is required here.

## Task Checklist

Check a task only after its evidence exists. Items are marked against the
2026-09-23 prototype run recorded in [RUN_LOG.md](RUN_LOG.md).

1. **Implement `read_training_text`.**
   - [x] Return the complete UTF-8 file as a `str`, retaining case, repeated
     spaces, tabs, accents, and line endings. Do not strip or split sentences.
   - [x] Verify preservation with a small file fixture containing whitespace
     and Unicode. Reject an empty fitting corpus with a clear error.
2. **Implement `train_tokenizer`.**
   - [x] Build byte-level BPE with a complete 256-byte alphabet, a matching
     decoder, and no lossy normalization or automatic prefix space.
   - [x] Reserve distinct `<pad>` and `<eos>` IDs. Set the target vocabulary to
     2,048 including reserved tokens; inspect actual size even if it is smaller.
   - [x] Fit on the README and retain the fitted tokenizer. Keep separate probe
     strings and any validation/test text out of fitting.
3. **Implement and exercise `check_round_trip`.**
   - [x] Inspect token pieces and integer IDs for an ordinary example.
   - [x] Assert exact reconstruction with automatic special-token insertion
     disabled, IDs in range, and repeatable encoding of each fixture.
   - [x] Cover ordinary ASCII/punctuation, leading/trailing/repeated spaces,
     tabs, LF/CRLF, accented and non-Latin text, emoji, and the empty string.
   - [x] Declare how literal `<pad>`/`<eos>` strings are handled and check that
     policy separately. Do not claim ordinary-text behavior covers reserved strings.
4. **Implement and verify `save_and_reload`.**
   - [x] Save the complete pipeline under `AI_ROADMAP_DATA` so it can be loaded
     again. No artifact hash or versioned run bundle is required for this lesson.
   - [x] Load from that file in a fresh Python process. Assert identical
     vocabulary-to-ID mapping, reserved IDs, and token pieces/IDs on the same
     fixtures; verify the declared pipeline settings and decoded text too.
   - [x] Check the assertions. Two successful round trips alone do not
     establish that encoding stayed unchanged. The instructor can run the
     fresh-process check as part of reviewing the implementation.
5. **Run and record the prototype.**
   - [x] Complete the reader -> fit -> encode/decode -> save/reload path without
     an exception, missing return, or unfinished placeholder in the executed path.
   - [x] Keep a short learning record: input used, actual vocabulary size, one
     example's pieces/IDs, round-trip/reload outcomes, and what you observed.
     The instructor can capture the command, output, and saved path.
6. **Explain the completed path.**
   - [x] Walk through your own code: what fitting learns, what encoding returns,
     and how one of your checks detects an incorrect tokenizer or reload.
     Record guided help honestly; no repeat of the full Session 1 quiz is required.

## Learning Record And Instructor Bookkeeping

Run-and-record remains part of the lesson at the learner's request: a short
note in [RUN_LOG.md](RUN_LOG.md) and [RESULTS.md](RESULTS.md) should help explain
what happened. Record observations and failures together; the instructor can
handle routine command/output capture. Input snapshots, hashes, environment
locking, and formal run bundles are not learner tasks or completion gates.
Later administrative bookkeeping should likewise stay with the instructor.

The [Python file plan](../../.omx/plans/week09-session2-tokenizer-20260922.md)
maps the existing functions, data flow, implementation order, and checks.

## Finish Condition

**Session 2 is complete only when every item in the checklist is evidenced:**

| Required outcome | Evidence needed |
| --- | --- |
| A real fitted tokenizer | The code runs successfully on the current README; inspect vocabulary and distinct PAD/EOS IDs |
| Correct text behavior | All declared reader, encoding, ordinary-text round-trip, and reserved-string checks pass; no required check is skipped |
| Equivalent saved tokenizer | Artifact exists outside Git; fresh-process reload preserves mappings, settings, and fixture encodings |
| Short learning record | Actual vocabulary size, token/ID example, check outcomes, and an observation recorded from the run |
| Learner understanding | A short explanation using the implemented code and one meaningful verification check |

If any row lacks evidence, record Session 2 as **in progress** and name the
missing item. Syntax/import/lint success alone is insufficient. Reaching exactly
2,048 tokens, outperforming GPT-2, training a language model, and proving broad
language coverage are not finish conditions. Neither is administrative
bookkeeping: no snapshots, hashes, or formal run bundles are required from the learner.

Current verdict: **complete (2026-09-24).** Every finish-condition row now has
evidence: a real fitted tokenizer (vocab 651, PAD 0 / EOS 1), ordinary-text
round trips across the full probe set, a declared and checked reserved-literal
policy, an equivalent saved tokenizer verified in a fresh process, the short
learning record, and the learner walkthrough.

## First Coding Step (Completed)

The first step was `read_training_text`, returning the complete decoded text as
one string through `Path.open(encoding="utf-8", newline="")` - without
translating line endings, stripping, lowercasing, or splitting sentences. The
rest of the path followed in the same file: fitting, ordinary-text round trips,
the reserved-literal policy check, and save/reload equivalence. A later empty
input now raises `ValueError` instead of returning silently.

The original starter no longer describes the file. It previously stopped at the
first `NotImplementedError`; the completed behavior is recorded in the
[completion checks](RUN_LOG.md#session-2-completion-checks---2026-09-24) and the
[prototype run](RUN_LOG.md#session-2-prototype-run---2026-09-23). Corpus
preparation, the training-only refit, and the measured comparison moved to
[Session 3](session3.md).

References: [training a tokenizer on a corpus](https://huggingface.co/learn/llm-course/en/chapter6/2),
[Tokenizers training and serialization API](https://huggingface.co/docs/tokenizers/quicktour).
Follow this project's train-only contract; the quicktour's illustrative use of
multiple dataset splits is not our evaluation protocol.
