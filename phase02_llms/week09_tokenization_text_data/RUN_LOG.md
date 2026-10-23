# Run Log

## Initialization - 2026-09-17

- Created the Week 9 documentation scaffold and Session 1 starter.
- Linked the project from the Phase 2 overview/roadmap and root status.
- Read the existing dataset registry and preparation helper.
- Checked the registry's descriptive WikiText-2 raw path and processed path:
  both are absent in the canonical external data root. This is not an exhaustive
  Hugging Face cache search.
- Reviewed official tokenizer, padding, causal-LM, and dataset documentation;
  references are linked in Session 1 and the experiment plan.

No corpus preparation, package installation, tokenizer fitting, model run,
executable Week 9 test, or learner assessment occurred during initialization.
Earlier Week 8 test results are not Week 9 verification.

## Session 1 Discussion And Documentation Closure - 2026-09-21

- Completed the conceptual discussion begun on September 20; the learner
  requested documentation and wrap-up after the final masking example.
- Recorded the learner's reasoning, guided corrections, and assessment limits
  in [session1.md](session1.md#discussion-record---2026-09-20-21).
- Covered tokenization/embeddings, BPE and alternatives, normalization/decoding,
  special tokens, batching/packing/chunking, masks, and split/selection leakage.
- Added the hand-written input/target/loss-mask example. It is explanatory
  material, not a measured tokenizer output or an executable test result.
- Updated the project and root handoffs to Session 2 while keeping every
  implementation, preparation, and experiment deliverable pending.
- Checked Markdown structure, local links, status consistency, and the document
  diff during closure; the [closure report](../../.omx/reports/task-2026-09-21-140451.md)
  records the checks. Earlier checkpoint reports retain their dated history.

No package installation, corpus preparation, tokenizer fit, model run, or
executable Week 9 test occurred during this closure. Session 2 has not started.

## Session 2 Scaffold - 2026-09-22

- The learner explicitly requested the Python structure and discussion of inputs
  and outputs. Created [tokenizer.py](tokenizer.py) with unimplemented reader,
  fitting, round-trip, and save/reload functions; no completed solution was supplied.
- Added [Session 2 notes](session2.md) and selected the current root README as
  a tiny practice input. WikiText-2 remains the measured-experiment proposal.
- Inspected Week 5's reader and character tokenizer. Its whitespace-stripping
  sentence preparation was not reused for the preservation contract.
- Syntax/import, default README path, flake8, scoped mypy, and documentation
  checks passed. The [session report](../../.omx/reports/task-2026-09-22-194922.md)
  records commands and mypy's external typing limitation. No tokenizer fitting,
  model training, downloads, or measured comparison ran.

## Session 2 Task And Finish Contract - 2026-09-22

- Recorded the learner-requested [task checklist](session2.md#task-checklist)
  and [finish condition](session2.md#finish-condition).
- Bounded this session to a README-trained prototype with a preserved input,
  passing behavioral checks, a saved tokenizer verified in a fresh process,
  reproducible run evidence, and a short learner code walkthrough.
- Kept WikiText preparation/provenance and training-only refitting as explicit
  prerequisites for Session 3's measured comparison. No comparison was removed.
- All practical tasks remain unchecked. No learner code was changed and no
  fitting, behavioral tests, or data preparation ran for this documentation task.

## Learner Workload Correction - 2026-09-22

- The learner rejected input locking as unnecessary work for this learning
  project and assigned any desired bookkeeping to the instructor.
- Removed README snapshots, source/artifact hashes, and manual run bundles from
  the Session 2 learner checklist and finish condition. The earlier task-contract
  entry above is superseded on those requirements.
- Use the current README as it is. Reading, fitting, encoding/decoding, saved
  tokenizer equivalence, and explaining the code remain the learning goals.
- The instructor handles useful run notes and supporting execution checks.
  No input was frozen, and no learner code or dataset was changed in this update.

## Python File Plan And Learning Record - 2026-09-22

- The learner clarified that run-and-record is useful for understanding and
  requested planning `tokenizer.py` before implementation.
- Saved the [file plan](../../.omx/plans/week09-session2-tokenizer-20260922.md)
  covering the four functions, main data flow, BPE components, and checks.
- Retained a brief actual run/observation note in Session 2's completion criteria.
  Snapshots, hashes, locking, and formal run bundles remain outside learner tasks.
- The reader currently opens a file without reading/returning text; that routine
  correction was explained in the plan. No Python source was edited or executed.

## Session 2 Prototype Run - 2026-09-23

The learner-implemented `tokenizer.py` now runs end to end on the root README
using the `tokenizers` library (0.22.2) under the project `.venv`
(Python 3.13.5), CPU only.

- Purpose: fit a byte-level BPE tokenizer on the practice README, inspect one
  encode/decode, save it, and prove reload equivalence.
- Command: `.venv/bin/python phase02_llms/week09_tokenization_text_data/tokenizer.py`,
  run from the repository root; exit status 0.
- Input: the current root `README.md` (1165 characters). Nothing was frozen,
  hashed, or snapshotted; this is a practice record, not a held-out evaluation.
- Result: target vocabulary 2048, **actual vocabulary 651** (the tiny corpus
  exhausts eligible merges), PAD id 0, EOS id 1.
- One example, probe `"Hello, world!\n Café 🙂 "`:

  ```
  pieces: ['H', 'e', 'll', 'o', ',', 'Ġw', 'or', 'l', 'd', '!', 'Ċ', 'Ġ', 'C', 'af', 'Ã', '©', 'Ġ', 'ð', 'Ł', 'Ļ', 'Ĥ', 'Ġ']
  ids:    [41, 70, 307, 80, 13, 337, 328, 77, 69, 2, 200, 222, 36, 443, 129, 104, 222, 174, 255, 249, 226, 222]
  ```

- Round trip and reload: pieces/IDs and decoded text were identical before and
  after saving and loading; every assertion passed.
- Saved artifact (outside Git):
  `<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session2_readme/tokenizer.json`,
  30,480 bytes. The default data root resolved to
  `/home/rui.gao/data/ai_research_roadmap`.
- Instructor fresh-process check: a separate interpreter loaded the artifact and
  reproduced vocabulary size 651, PAD 0 / EOS 1, pipeline `ByteLevel`
  pre-tokenizer + `ByteLevel` decoder, and exact round trips for five fixtures
  (`"Hello, world!\n  Café 🙂"`, `""`, `"a\tb  c"`, `"naïve — 日本語"`,
  `"line1\r\nline2"`).

Observation: because the configuration is byte-level, characters absent from
the training README (emoji, non-Latin) still encode and decode exactly — all
256 byte values are in the alphabet. The vocabulary simply has no dedicated
merged token for them, so such characters cost several tokens each.

Failures retained (not hidden): earlier iterations of this file raised a
`SyntaxError` (an accidental non-code token and a single `=` on the `__main__`
guard) and a `NameError` (undefined, singular-vs-plural `<pad>`/`<eos>`
constants) before reaching the green run above.

Still pending for Session 2: the reserved-string (`<pad>`/`<eos>`) policy check,
the short learner code walkthrough, and status-text sync in `TODO.md`, the
Week 9 `README.md`, and `session2.md`. The measured WikiText-2 comparison
remains a Session 3 prerequisite.

## Session 2 Completion Checks - 2026-09-24

Added the remaining checklist items and re-ran the prototype. Command:
`.venv/bin/python phase02_llms/week09_tokenization_text_data/tokenizer.py`
from the repository root; exit status 0.

- Empty-corpus guard: `read_training_text` now raises `ValueError` on an empty
  file. Checked with a temporary empty file.
- Preservation fixture: a temporary UTF-8 file containing leading/repeated
  spaces, a tab, LF and CRLF line endings, and an emoji was read back byte-for-
  byte (`'  Café\t\nline\r\n🙂  '`).
- Probe coverage: the executed path now round-trips eight probes — ordinary
  ASCII/punctuation, leading/trailing/repeated spaces, a tab, LF, CRLF,
  accented + non-Latin text, emoji, and the empty string — before and after
  reload.
- Reserved-literal policy (declared and checked): a literal `<pad>`/`<eos>` in
  the text is matched as the **reserved IDs** (0 and 1), even with
  `add_special_tokens=False`. The default `decode` (`skip_special_tokens=True`)
  drops them, so such text round trips only with `skip_special_tokens=False`;
  `check_reserved_literal_policy` asserts both the collision and the drop.
  Ordinary text is unaffected, which is why this is a separate check.

All assertions passed before and after reload; the saved artifact was
regenerated at the same path as the 2026-09-23 run.

## Session 2 Closure And Status Sync - 2026-09-24

- Closed Session 2 against its
  [finish condition](session2.md#finish-condition): a real fitted tokenizer
  (vocab 651, PAD 0 / EOS 1), ordinary-text round trips across the full probe
  set, a declared and checked reserved-literal policy, an equivalent saved
  tokenizer verified in a fresh process, the short learning record, and the
  learner code walkthrough.
- Corrected stale status text that still described Session 2 as in progress: the
  root `README.md`, the [Phase 2 overview](../README.md), the
  [Phase 2 roadmap](../phase02_llms_roadmap.md), `TODO.md`, and
  `PROJECT_DESCRIPTION.md`.
- Documentation only. No learner code, artifact, or dataset was changed in this
  sync.

## Session 3 Scaffold - 2026-09-24

- Created [session3.md](session3.md): learning question, working contract to
  freeze, task checklist, finish condition, and first learning step for the
  training-only WikiText-2 refit and the frozen-GPT-2 comparison.
- Created the file plan
  [.omx/plans/week09-session3-wikitext-comparison-20260924.md](../../.omx/plans/week09-session3-wikitext-comparison-20260924.md)
  describing `wikitext_compare.py`, which does not exist yet.
- Read-only prerequisite checks: the external data root exists; the Session 2
  artifact is present at
  `processed/week09_tokenization_text_data/session2_readme/tokenizer.json`;
  the registry's WikiText-2 raw and processed paths are absent; no WikiText-2
  entry exists in the Hugging Face hub or datasets caches; the cached
  `openai-community/gpt2` tokenizer loads offline (vocabulary 50,257, exact
  round trip on a probe); `.venv` reports Python 3.13.5 with `tokenizers`
  0.22.2, `transformers` 5.12.1, `datasets` 5.0.0, `torch` 2.11.0+cu128, and
  `huggingface_hub` 1.19.0.
- The single-fixture token-count example in `session3.md` is illustrative and
  explicitly not a measured Session 3 result.
- At the time of this scaffold entry: no corpus was downloaded, no
  training-only refit ran, and no comparison was computed. The corpus fetch
  required explicit authorization and was carried out later the same day, as
  recorded below.

## Session 3 Corpus Fetch And Measured Comparison - 2026-09-24

The learner authorized starting the session, which included the network fetch
that the scaffold step had deferred.

- Corpus: `Salesforce/wikitext`, config `wikitext-2-raw-v1`, pinned revision
  `b08601e04326c79dfdd32d625aee71d232d685c3`, three parquet files fetched
  directly with `huggingface_hub` into
  `<AI_ROADMAP_DATA>/raw/huggingface/Salesforce/wikitext/wikitext-2-raw-v1/`
  (6,357,543 / 657,209 / 732,610 bytes). The preparation helper was not used
  because it pins no revision and its dry-run writes directories and a manifest.
- Split counts: train 36,718 rows / 23,767 nonempty records / 10,914,845 bytes;
  validation 3,760 / 2,461 / 1,144,248; test 4,358 / 2,891 / 1,287,656. The test
  split was not encoded.
- Command: `.venv/bin/python phase02_llms/week09_tokenization_text_data/wikitext_compare.py`
  from the repository root; exit status 0.
- Fitting subset: 2,222 nonempty training records, source rows 1-3,465,
  1,048,468 of a 1,048,576-byte cap; the cap stopped at the first row that would
  have exceeded 1 MiB.
- Custom tokenizer: fitted in 0.15 s on CPU, actual vocabulary 2,048 (target
  reached this time), PAD 0 / EOS 1, saved to
  `<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session3_wikitext_trainonly/tokenizer.json`
  (122,433 bytes). Round trips, the reserved-literal policy check, and
  fresh-process reload equivalence all passed. The Session 2 README artifact was
  not modified.
- Reference: `openai-community/gpt2` tokenizer files from cache, vocabulary
  50,257; no model weights loaded and no fitting on this corpus.
- Comparison on the same 2,461 validation records with no padding, truncation,
  or automatic special tokens: custom 382,916 tokens / 0.3346 per byte / median
  135 / p95 414 / 1,259 records over 128 tokens; GPT-2 247,289 tokens / 0.2161 /
  median 87 / p95 264 / 880 over 128. The custom tokenizer needs about 55% more
  tokens (1.55x). Neither tokenizer produced an unknown token.
- Overlap audit: 243 validation records (9.9%) and 266 test records (9.2%) also
  appear verbatim in train, and 153 fitting rows (6.9%) are such shared short
  headings. A sensitivity refit without those rows gave 382,992 tokens (+0.02%)
  and median 136, so the overlap does not move the comparison. Split membership
  was left unchanged, as the experiment plan requires.
- Notable failure/edge case: WikiText-2's `@-@`/`@.@` preprocessing markup costs
  several custom byte pieces while GPT-2 has merged tokens for it. Largest
  per-row gaps were ordinary long prose (+289, +255, +252 tokens).
- First run of the harness aborted on a deliberately strict assertion that
  treated any train/validation text overlap as leakage. The overlap is upstream
  duplication of short heading lines, so the check was converted into the
  measured audit above plus the sensitivity arm. The failure is retained here
  rather than hidden.
- Artifacts: `metrics.json` and `tokenizer.json` under the run directory above.
  This is a descriptive comparison; no superiority, perplexity, or accuracy
  claim is made, and the test split remains reserved.

## Session 3 Learning Checks - 2026-09-24

After the measured run, the session's learning points were checked in guided
discussion mode, one question at a time. The full tally is recorded in the
[Session 3 learning record](session3.md#learning-record---2026-09-24).

- Tokens-per-byte and the confound boundary illustrate the difference between
  *why a number moves* and *what the number licenses a claim to say*.
- Learner independently complete on 1 of 6 topics (tokenizer-level leakage),
  partial on 2 (fixed-budget reallocation, overlap handling), instructor-supplied
  on 1 (unit of comparison), and thin on 2 (the confound boundary and claim
  scope).
- Guidance supplied during the check: the leakage fairness asymmetry, the
  per-record versus per-byte denominator, the frequency-rank correction to
  "niche", and the design consequences of deleting duplicated rows.
- No files, artifacts, or measurements changed during the check. This closes
  finish-condition row 7 on a dated record rather than on instructor-written
  documentation alone.

## Session 4 Scaffold - 2026-09-25

- Created [session4.md](session4.md): the LM example contract restated from the
  experiment plan, a hand-written worked illustration, the task checklist with
  its finish condition, the common traps, and the first learning step.
- Created [lm_data.py](lm_data.py) as a **learner scaffold**: frozen constants
  (`T = 128`, stride `128`, `PAD 0`, `EOS 1`, `IGNORE_INDEX -100`), four function
  signatures with docstrings, and `NotImplementedError` bodies. This mirrors how
  Session 2 started.
- Verified the scaffold: `py_compile` passes, the module imports, calling
  `build_split_stream` raises `NotImplementedError` as intended, and
  `flake8 --max-line-length=100` is clean.
- Created the file plan
  [.omx/plans/week09-session4-lm-data-20260925.md](../../.omx/plans/week09-session4-lm-data-20260925.md).
- No stream, window, mask, or executable check exists yet. Nothing was fitted,
  encoded, or measured, and no corpus file was touched.
- Attribution note: the scaffold's names and docstrings are instructor-authored
  boilerplate. The bodies are the learner's, and `wikitext_compare.py` from
  Session 3 remains instructor-authored - the soft spot recorded in the Session 3
  learning record.

## Session 4 Fixture Spec - 2026-09-25

- Added [tests/test_session4_lm_data.py](tests/test_session4_lm_data.py): nine
  hand-written fixture checks that encode the LM example contract as executable
  assertions. They need no corpus and no network.
- Covered cases: a full block, the `[5, 6, 1]` padded illustration, the
  masks-differ trap, a real EOS target with padding after it, a lone trailing
  token, an empty stream, one-token window overlap, single-shift integrity, and
  one EOS appended per record through the real tokenizer path.
- Declared the return contract explicitly in `lm_data.py`: `build_source_windows`
  returns `(windows, info)` with the keys `stream_length`, `windows`,
  `final_window_padded`, and `lone_trailing_token`; `make_lm_batch` returns a dict
  with `input_ids`, `targets`, `attention_mask`, and `loss_mask`, each a list of
  per-window lists.
- Current state: the checks are **red**. The first one stops at
  `NotImplementedError` in `build_source_windows`, as intended. `flake8
  --max-line-length=100` is clean on both files.
- No corpus was read, nothing was fitted or encoded, and no measurement was
  taken. The fixture spec is instructor-authored; the implementations it checks
  are the learner's.

## Session 4 Deferral - 2026-09-25

The learner chose to defer the Session 4 implementation to the start of Week 10,
where masks and targets can be tested against a real cross-entropy loss.

- Recorded, not abandoned: the pickup point is the start of Week 10, and Week 11
  (mini-GPT training) cannot run without streams, `T + 1` windows, one-shifted
  targets, and masks.
- Week 9's deliverable is therefore half done: tokenizer complete and measured,
  prepared LM data deferred. That is now stated that way in the Week 9
  [README](README.md), `TODO.md`, and the Phase 2 roadmap.
- Costs accepted and recorded: data-vs-model debugging becomes entangled in the
  same week, and the fixtures exist to keep them separable.
- Also recorded from the concept discussion: `attention_mask` is pointwise
  ("is this input real?") and `loss_mask` is pairwise ("are input and target
  real?"), so they diverge only at a right-padding boundary. Under right padding
  plus causal attention the padding attention mask is nearly idle; it becomes
  load-bearing with left-padded generation batches or non-causal attention.
- No code, corpus, or artifact changed in this entry. The scaffold and fixture
  spec remain as left, with the fixtures still red.

## Session 5 Review And Week 9 Closure - 2026-09-25

- Created [session5.md](session5.md) and the [Week 9 lab note](LAB_NOTE.md).
- Reviewed the week end to end: Session 1's conceptual checkpoint, Session 2's
  README tokenizer, Session 3's corpus and comparison, the learner-understanding
  check, and Session 4's deferral.
- Derived the context-length statement from the Session 3 metrics rather than a
  new run: at 0.3346 tokens per byte the custom tokenizer spends ~2.99 bytes per
  token against GPT-2's ~4.63, so a 128-token window covers roughly 383 bytes
  versus 593 - about 35% less text for the same window. 51.2% of validation
  records exceed 128 tokens under the custom tokenizer versus 35.8% under GPT-2.
- Recorded the closing verdict: Week 9 is closed with one deferred deliverable.
  The roadmap promises "tokenizer and prepared LM dataset"; only the tokenizer
  side is done, so the week is recorded as half done in the project and phase
  status surfaces rather than as complete.
- Recorded limitations at closure: no model was trained; the comparison is
  descriptive with two confounded variables; `wikitext_compare.py` is
  instructor-authored while `tokenizer.py` is learner-authored; the practice
  README input was deliberately unfrozen; the 1 MiB fitting subset is 9.6% of the
  training text; and the padding attention mask was designed but never exercised.
- Reproduction steps recorded in the lab note: the Session 2 prototype command,
  the Session 3 comparison command with its artifact paths, and the red Session 4
  fixture command.
- Documentation only. No code ran, nothing was fitted or encoded, and no
  measurement was taken. The Session 4 scaffold and fixture spec remain unchanged
  and still red.

## Session 4 Reschedule - 2026-09-26

The learner rescheduled the deferred Session 4 work: instead of being parked in
one block at the start of Week 10, it is **split across Weeks 9 and 10**.
Documentation only; no code ran and the scaffold and fixture spec are unchanged
and still red.

- **Week 9 half - the stream:** `build_split_stream` (encode nonempty records in
  stable order, one EOS per record) and the stream-level fixture rows. This is
  the half that touches the tokenizer/corpus contract Week 9 established, so it
  stays owed against Week 9's deliverable.
- **Week 10 half - the examples:** `build_source_windows` and `make_lm_batch`
  (`T + 1` windows at stride `T`, the short-final-window padding decision, the
  lone-trailing-token record, one-shifted targets with `-100`, and the distinct
  `attention_mask` / `loss_mask`), plus the fixture gate and the record in this
  file and `RESULTS.md`. This half travels with the attention block.
- **Week 11 gate, unchanged:** mini-GPT training cannot run until both halves are
  done, and the masks and targets are only validated against a real
  cross-entropy loss there.
- Reason for the split: the earlier deferral rationale (masks "tested against a
  real cross-entropy loss") describes Week 11's mini-GPT work, while Week 10 is
  attention-only and needs no LM batches at all. Splitting puts the tokenizer-
  facing half where its inputs already exist and leaves the model-facing half
  with the model work, instead of holding all three functions for a week that
  does not consume them.
- Week 9's deliverable stays **part done**: tokenizer complete and measured,
  stream half open, example half scheduled for Week 10.

## Deliverable Redraw - 2026-09-28

Documentation only. No code ran; `lm_data.py` and the fixture spec are unchanged
and still red.

- Learner challenge: the stream half (`build_split_stream`) is not tokenizer work,
  so why does it count against Week 9? Correct as stated - nothing technical puts
  it there.
- Root cause: the roadmap's Week 9 deliverable line ("tokenizer and prepared LM
  dataset") and its Thursday slot ("tokenized chunks, attention masks, and
  train/val split") claimed the LM example builder for a single 90-minute
  session, which is why the work slipped on 2026-09-25.
- Decision: keep the Weeks 9-10 split, redraw Week 9's deliverable to *a
  training-only tokenizer and the per-split token streams it produces*, and make
  the prepared LM dataset Week 10's deliverable outright.
- Rationale for keeping the stream in Week 9: it is the only function that
  applies Week 9's record/EOS convention to the Week 9 tokenizer artifact, it is
  one function plus one fixture row, and it keeps the model week focused on the
  example half.
- Rejected: moving the stream to Week 10 with the rest. That would leave Week 9
  with only the tokenizer but load every data-preparation step into the attention
  week.
- Week 9's remaining item is therefore exactly one function, not "half of a
  deliverable": `build_split_stream` plus the stream-level fixture row. The
  prepared LM dataset (windows, one-shifted targets, both masks, fixture gate,
  and this file's record) is Week 10 work.
- Week 11 remains the hard gate for both halves.
- Surfaces updated: root roadmap Week 9 and Week 10 deliverable lines, Phase 2
  roadmap, Phase 2 README, Week 9 README, `RESULTS.md`, `LAB_NOTE.md`,
  `session4.md`, `session5.md`, root `README.md`, `PROJECT_DESCRIPTION.md`,
  `TODO.md`.

## Session 4 Stream And Example Half Implementation - 2026-10-05

Both halves of Session 4 are now implemented and green. This closed Week 9's
redrawn deliverable (the per-split token streams) and Week 10's prepared
LM-dataset deliverable (windows, one-shifted targets, both masks).

Implemented in [lm_data.py](lm_data.py), by the learner for the stream and
example halves with instructor review:

- `build_split_stream` - skip blank rows, encode the rest with
  `add_special_tokens=False`, append exactly one EOS per nonempty record.
- `build_source_windows` - `block_length + 1` windows at stride `block_length`,
  right-padding the final short window only when it holds a real input/target
  pair, and logging a lone trailing token instead of windowing it.
- `make_lm_batch` - `input_ids` = window[0:T], `targets` = window[1:T+1] with
  padding mapped to `-100`, `attention_mask` at real inputs, `loss_mask` only
  where input and next token are both real.
- `check_fixture_alignment` - the five hand-written fixture cases, and `main()`.

Defect found and fixed during implementation: a `for start in range(0, ...)`
window loop re-read the previous window's last token as a lone trailing token.
The fixture `test_full_block_needs_no_padding` caught it; the fix is the
`while len(stream) - start >= 2` loop, since window `k + 1` legitimately starts
on window `k`'s last token.

Fixture command, run from the repository root:

```text
.venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
```

Result: **9/9 checks passed**, exit 0.

Real-corpus command:

```text
AI_ROADMAP_DATA=/home/rui.gao/data/ai_research_roadmap \
  .venv/bin/python phase02_llms/week09_tokenization_text_data/lm_data.py
```

```text
train       rows= 36718 records= 23767 stream= 3682072 windows= 28767 padded_final=True lone_trailing=None
validation  rows=  3760 records=  2461 stream=  385377 windows=  3011 padded_final=True lone_trailing=None
test        rows=  4358 records=  2891 stream= 436253 windows=  3409 padded_final=True lone_trailing=None
```

Independent cross-check: 382,916 (Session 3 custom-tokenizer count on the same
2,461 validation records) + 2,461 EOS tokens = 385,377, exactly the measured
validation stream. The first train window decodes to
`' = Valkyria Chronicles III = \n Senjō no Valkyria'`.

Known limitation, recorded rather than hidden: a source row containing a literal
`<eos>` would tokenize to id 1 and split one record into two. The pinned
WikiText-2 splits contain zero such rows, so the edge case is fixture-visible
only by construction, not observed on the corpus. `lone_trailing_token` is also
fixture-only: all three splits end on a full window.

Week 9's deliverable is therefore complete: tokenizer measured (Session 3) and
per-split streams implemented (this session). The prepared LM dataset closed
under Week 10 the same day; see
[week10 RUN_LOG](../week10_attention_block/RUN_LOG.md).

## Future Run Record

The instructor maintains detailed records when useful for an actual experiment.
Session 2 requires only the short learning note above, not every field below.

For each actual run, retain:

1. Date, purpose, exact command, working directory, and exit status.
2. Source identity, package versions, hardware, and elapsed time.
3. Dataset revision/fingerprints, split and subset IDs/counts, preprocessing.
4. Tokenizer settings/identity, saved artifact paths and hashes.
5. Measured results, test output, failures, interpretation, and next action.

Add commands only once their scripts exist. Label examples or proposals as
unexecuted. Never overwrite an earlier result to hide a failed attempt.
