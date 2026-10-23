# Results

Measured experiment status: **the Session 3 WikiText-2 comparison has run
(2026-09-24)** and is the first measured Week 9 result; **Session 4's stream and
example halves were implemented on 2026-10-05** and their 9/9 fixture checks now
pass. The Week 9 lab note was written on 2026-09-25; the prepared LM dataset was
reassigned to Week 10 on 2026-09-28 and closed there on 2026-10-05. Week 9 was
initialized on 2026-09-17, and
[Session 1](session1.md#discussion-record---2026-09-20-21) closed on 2026-09-21
as a guided conceptual checkpoint.

[Session 2](session2.md) now has a **working README-trained prototype**
(2026-09-23). The learner implemented all four functions against the
`tokenizers` library, and the full reader -> fit -> encode/decode ->
save/reload path runs with every assertion passing; an instructor
fresh-process reload check also passed. This is a practice run on the root
README, not the measured WikiText-2 experiment.

[Session 3](session3.md) ran on **2026-09-24**. WikiText-2 was fetched at
revision `b08601e04326c79dfdd32d625aee71d232d685c3`, the custom tokenizer was
refitted on training text only, and both tokenizers encoded the same 2,461
validation records. Full record:
`<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session3_wikitext_trainonly/metrics.json`.

| Corpus split | Rows | Nonempty records | UTF-8 bytes |
| --- | --- | --- | --- |
| train | 36,718 | 23,767 | 10,914,845 |
| validation | 3,760 | 2,461 | 1,144,248 |
| test | 4,358 | 2,891 | 1,287,656 |

Training-only fitting subset: 2,222 records (source rows 1-3,465), 1,048,468
bytes of a 1,048,576-byte cap; fitted in 0.15 s to the full 2,048-token target
vocabulary.

| Validation metric (identical 2,461 rows, no padding/truncation) | Custom, training-only BPE | Frozen GPT-2 |
| --- | --- | --- |
| Vocabulary size | 2,048 | 50,257 |
| Total payload tokens | 382,916 | 247,289 |
| Tokens per UTF-8 byte | 0.3346 | 0.2161 |
| Median tokens per record | 135 | 87 |
| 95th-percentile tokens per record | 414 | 264 |
| Max tokens per record | 727 | 471 |
| Records longer than 128 tokens | 1,259 (51.2%) | 880 (35.8%) |
| Unknown or unsupported outputs | 0 | 0 |

The custom tokenizer needs about 55% more tokens (1.55x) for the same held-out
text. Overlap audit: 243 validation and 266 test records also appear verbatim in
train; 153 fitting rows (6.9%) are such shared headings, and refitting without
them moved the total by +0.02% (382,992 tokens). Recorded rather than removed,
and the declared split membership was left unchanged.

This is a descriptive comparison. Different vocabulary sizes and training
histories mean the token counts are not an isolated algorithm result, and they
are not perplexity or accuracy.

| Deliverable | Current evidence |
| --- | --- |
| Session 1 understanding | Guided discussion closed 2026-09-21; learner reasoning and corrections recorded; no claim of independent mastery of every subtopic |
| Corpus provenance and train-only fitting subset | Verified 2026-09-24: pinned revision, three upstream splits counted, fitting subset declared as rows 1-3,465 / 1,048,468 bytes; cross-split duplicate overlap audited |
| Saved custom tokenizer and reload behavior | README prototype (2026-09-23, vocab 651, PAD 0 / EOS 1) and the training-only WikiText-2 artifact (2026-09-24, vocab 2,048, 122,433 bytes); both reload equivalently in a fresh process |
| Training-only WikiText-2 refit | Run 2026-09-24: fitted in 0.15 s on 2,222 training-only records; separate artifact, README prototype untouched |
| Pretrained-tokenizer comparison | Run 2026-09-24 on 2,461 identical validation rows: custom 382,916 tokens vs GPT-2 247,289 (1.55x); no unknown tokens for either |
| LM chunk/mask/target checks | **Implemented 2026-10-05; 9/9 fixture checks pass.** `build_split_stream`, `build_source_windows`, `make_lm_batch`, `check_fixture_alignment`, and `main()` in [lm_data.py](lm_data.py). Real-corpus run: streams 3,682,072 / 385,377 / 436,253 tokens and 28,767 / 3,011 / 3,409 windows for train / validation / test. The validation stream cross-checks against Session 3's 382,916 tokens plus 2,461 EOS |
| Week 9 deliverable | Redrawn 2026-09-28 to *a training-only tokenizer and the per-split token streams it produces*. **Both halves done 2026-10-05:** tokenizer complete and measured (Session 3); `build_split_stream` implemented with 9/9 fixtures green; the prepared LM dataset (`T + 1` windows, one-shifted targets, both masks) closed under Week 10. Lab note written 2026-09-25 |

Use the [experiment plan](EXPERIMENT_PLAN.md) for measurement definitions.
When runs exist, link their settings, exact commands, metrics, and artifacts.
Distinguish observed behavior from assumptions and guided understanding from
independent demonstration. Do not insert estimated results into measured tables.

## Week 9 Closure - 2026-09-25

[Session 5](session5.md) reviewed these results and wrote the
[Week 9 lab note](LAB_NOTE.md). **Week 9's deliverable was redrawn on
2026-09-28:** it is now *a training-only tokenizer and the per-split token streams
it produces*. The tokenizer is complete and measured; the stream half is the
single item still owed; the prepared LM dataset ([Session 4](session4.md)) is
Week 10's deliverable outright.

No new measurement was added at closure. Every number above comes from the
Session 3 run recorded in the run directory's `metrics.json`; the context-length
analysis in the lab note is derived from those same figures rather than from a
fresh run.

## Session 4 Implementation - 2026-10-05

Session 4's two halves were implemented and verified, closing Week 9's redrawn
deliverable and Week 10's prepared-LM-dataset deliverable at the same time.

```text
.venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
```

Result: **9/9 checks passed**, exit 0.

| Split | Rows | Records | Stream tokens | Windows |
| --- | --- | --- | --- | --- |
| train | 36,718 | 23,767 | 3,682,072 | 28,767 |
| validation | 3,760 | 2,461 | 385,377 | 3,011 |
| test | 4,358 | 2,891 | 436,253 | 3,409 |

Cross-check: the validation stream equals the Session 3 custom-tokenizer count
(382,916) plus one EOS per nonempty record (2,461) - independent evidence that
the stream builder applies the record/EOS convention exactly once per record.
All three splits report `padded_final=True` and `lone_trailing=None`.

Limitations recorded in [RUN_LOG.md](RUN_LOG.md): a literal `<eos>` inside a
source row would split a record (zero such rows in the pinned splits), and the
`lone_trailing_token` branch is fixture-only.
