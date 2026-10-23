# Week 9 Lab Note - Tokenization and Text Data

Date: 2026-09-25. Instructor-authored summary, grounded in the saved runs,
recorded commands, and the learner's discussion. It does not claim independent
learner mastery; the [learning record](session3.md#learning-record---2026-09-24)
and [Session 5](session5.md) carry that assessment separately.

**Status: Week 9's deliverable was redrawn on 2026-09-28.** It is now *a
training-only tokenizer and the per-split token streams it produces*, not the
whole prepared LM dataset. The tokenizer is complete and measured; the stream
half (Session 4) is the single item still owed; the prepared LM dataset - `T + 1`
windows, one-shifted targets, both masks - is Week 10's deliverable outright.

## What Week 9 Set Out To Do

Roadmap goal: *prepare language-modeling data correctly*. Five planned sessions:
concepts, a working tokenizer, a measured comparison, the LM example contract,
and this note plus handoff.

## What Actually Ran

| Session | Outcome |
| --- | --- |
| 1 - Tokens, vocabulary, text data contract | Closed 2026-09-21 as a guided conceptual checkpoint with corrections and assessment limits recorded |
| 2 - README-trained tokenizer prototype | Closed 2026-09-24: learner-implemented `tokenizer.py`, vocabulary 651, PAD 0 / EOS 1, reload-equivalent |
| 3 - WikiText-2 corpus and comparison | Run 2026-09-24: pinned corpus, training-only refit, frozen-GPT-2 comparison, overlap audit, learning check |
| 4 - LM chunks, targets, masks | **Deferred 2026-09-25, rescheduled 2026-09-26 as a Weeks 9-10 split**; scaffold and a red fixture spec exist, no implementation |
| 5 - Review and lab note | This note, 2026-09-25 |

Nothing in Week 9 trained a model. No perplexity, accuracy, or generation quality
was measured; none of those were in scope.

## Measured Results

Corpus: `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. A record is one nonempty source row.

| Split | Rows | Nonempty records | UTF-8 bytes |
| --- | --- | --- | --- |
| train | 36,718 | 23,767 | 10,914,845 |
| validation | 3,760 | 2,461 | 1,144,248 |
| test | 4,358 | 2,891 | 1,287,656 |

The custom tokenizer was fitted on **training text only**: 2,222 nonempty records
(source rows 1-3,465), 1,048,468 of a 1,048,576-byte cap, in 0.15 s, reaching the
full 2,048-token target vocabulary. The test split was never encoded.

| Validation metric (2,461 identical records, no padding or truncation) | Custom, training-only BPE | Frozen GPT-2 |
| --- | --- | --- |
| Vocabulary size | 2,048 | 50,257 |
| Total payload tokens | 382,916 | 247,289 |
| Tokens per UTF-8 byte | 0.3346 | 0.2161 |
| Median tokens per record | 135 | 87 |
| 95th-percentile tokens per record | 414 | 264 |
| Max tokens per record | 727 | 471 |
| Records longer than 128 tokens | 1,259 (51.2%) | 880 (35.8%) |
| Unknown or unsupported outputs | 0 | 0 |

Overlap audit: 243 validation records (9.9%) and 266 test records (9.2%) also
appear verbatim in train; 153 fitting rows (6.9%) are such shared short headings.
Refitting without them moved the total by +0.02% (382,992 tokens), so the overlap
does not move the comparison and the declared split membership was left unchanged.

## How Tokenization Changes Context

This was Week 9's core question, and the numbers answer it concretely.

**Per token, GPT-2 carries more text.** Inverting the density: the custom
tokenizer spends ~2.99 bytes per token, GPT-2 ~4.63 bytes per token.

**The same context window holds less.** A 128-token window covers roughly 383
bytes under the custom tokenizer and 593 bytes under GPT-2 - about **35% less
text** for the same window. That is the mechanism by which a tokenizer choice
becomes a context-length limitation before any model exists.

**Record fit changes.** 51.2% of validation records exceed 128 tokens under the
custom tokenizer versus 35.8% under GPT-2, so a 128-token model would see the
shorter tail of the corpus more often.

**Coverage is not quality.** Both tokenizers represent every validation record
exactly, with zero unknown tokens, because both are byte-level. The difference is
cost, not capability. Non-Latin text and emoji are representable but expensive
under a small vocabulary; WikiText-2's own `@-@`/`@.@` markup costs several
custom pieces while GPT-2 has merged tokens for it.

**Vocabulary size is a trade-off with a model-side bill.** A larger vocabulary
reduces tokens per byte, but `vocab x d_model` grows in both the input embedding
and the output projection, the softmax widens, the uniform-loss floor rises like
`ln V`, and rarely-used entries stay underfit because they receive few gradient
signals. Bigger is not automatically better; it is a budget decision that should
be measured against the downstream model.

## What This Comparison Does Not Show

- It is descriptive, not causal. Vocabulary size (2,048 vs 50,257) and training
  text (1 MiB of WikiText vs GPT-2's large web corpus) both differ, so the 1.55x
  gap cannot be attributed to either.
- It says nothing about language-model quality. No model was trained, and token
  counts are not perplexity or accuracy.
- The custom tokenizer had the distribution advantage - fitted and evaluated on
  the same domain - and still needed ~55% more tokens, which weakens any hope
  that matched distribution alone closes the gap.
- No vocabulary sweep was run. Isolating vocabulary size would need 2k/4k/8k with
  everything else frozen, and replication across splits.

## Limitations And Honest Gaps

- **The prepared-LM-dataset deliverable is not met.** Roadmap Week 9 promises a
  tokenizer *and* prepared LM data; only the tokenizer is done.
- **Code authorship is mixed.** `tokenizer.py` is learner-authored;
  `wikitext_compare.py` (the measured harness) is instructor-authored. That is
  the main soft spot in the week's evidence.
- **The learner-understanding check was partial.** Independently complete on the
  leakage question; partial on fixed-budget reallocation and overlap handling;
  instructor-supplied on why tokens-per-byte rather than tokens-per-record; thin
  on the confound boundary and on claim scope. The recurring gap: reasoning about
  *why a number moves* is strong, stating *what the number licenses* is weak.
- The 1 MiB fitting subset is 9.6% of the training text. Vocabulary *size*
  saturated, but merge *quality* was never tested against a full-corpus refit.
- The README practice input was never frozen or hashed, by the learner's explicit
  choice; the measured corpus was pinned instead.
- The padding attention mask was designed but never exercised, since no model
  consumed it. Its weakness under right padding plus causal attention is recorded
  in the [run log](RUN_LOG.md), not resolved.

## Reproduction

Environment: `.venv`, Python 3.13.5, `tokenizers` 0.22.2, `transformers` 5.12.1,
`datasets` 5.0.0, `torch` 2.11.0+cu128, CPU only. Run everything from the
repository root.

1. **Tokenizer prototype** (README-trained, practice only):

   ```text
   .venv/bin/python phase02_llms/week09_tokenization_text_data/tokenizer.py
   ```

   Artifact: `<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session2_readme/tokenizer.json`

2. **Corpus and measured comparison** (needs the pinned raw splits):

   ```text
   .venv/bin/python phase02_llms/week09_tokenization_text_data/wikitext_compare.py
   ```

   Raw splits: `<AI_ROADMAP_DATA>/raw/huggingface/Salesforce/wikitext/wikitext-2-raw-v1/`
   Artifacts: `.../processed/week09_tokenization_text_data/session3_wikitext_trainonly/tokenizer.json`
   and `metrics.json` (122,433 bytes and the JSON record of every number above).

3. **Fixture spec for the split Session 4 work** (currently red by design):

   ```text
   .venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
   ```

## Week 10 Handoff

- **Finish Session 4's stream half first, then Week 10 owns its example half.**
  The stream half (`build_split_stream`) is the one item owed against Week 9's
  redrawn deliverable; the example half (`T + 1` windows, one-shifted targets,
  both masks) is Week 10's deliverable - the prepared LM dataset - built beside
  the attention block. The attention block itself can be built and tested with
  synthetic tensors, but Week 11 training cannot run without both halves.
- **Frozen decisions to reuse:** `T = 128` with stride `T`, PAD 0, EOS 1,
  `IGNORE_INDEX -100`, and the training-only tokenizer artifact from Session 3.
- **Questions to carry forward:** whether 2,048 is the right vocabulary for the
  downstream model, and how the `@-@` markup interacts with a tiny vocabulary.
- **Habit to carry forward:** when a number moves, state what it licenses. That
  was the recurring gap in the learning check, and Week 10's shape tests are a
  good place to practice it.
