# Session 3 - WikiText-2 Corpus Preparation And Tokenizer Comparison

## Status

**Prepared and then executed on 2026-09-24; Session 3 is complete.** The corpus
was fetched at a pinned revision, verified, and used for a training-only refit;
the frozen GPT-2 comparison ran on identical validation rows. The numbers are in
[Measured Result](#measured-result---2026-09-24) below and in
[RUN_LOG.md](RUN_LOG.md) and [RESULTS.md](RESULTS.md).

The contract was scaffolded earlier the same day, immediately after
[Session 2](session2.md) closed on its README-trained prototype. Corpus
preparation, training-only refitting, and the measured comparison were left here
as an explicit prerequisite in
[Session 2's scope note](session2.md#session-2-scope).

## Handoff From Session 2

What already exists and can be reused:

- [tokenizer.py](tokenizer.py) with a working byte-level BPE path:
  `read_training_text`, `train_tokenizer`, `check_round_trip`,
  `check_reserved_literal_policy`, `save_and_reload`, and `main`.
- A saved README-trained artifact at
  `<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session2_readme/tokenizer.json`
  (actual vocabulary 651, PAD 0 / EOS 1, reload-equivalent).
- Declared behavior rules: preserve case/accents/whitespace, no prefix space, no
  normalization, and a separate checked policy for literal `<pad>`/`<eos>`.

What is still missing and is this session's work:

- A verified WikiText-2 corpus with preserved upstream splits.
- A declared training-only fitting subset.
- A training-only refit that is kept separate from the README prototype.
- A frozen pretrained reference and the measured comparison on identical
  validation text.

## Learning Question

From the [experiment plan](EXPERIMENT_PLAN.md): how does a small byte-level BPE
tokenizer fitted on training text only compare with a fixed pretrained tokenizer
in sequence length, text preservation, and failure cases, when both see exactly
the same held-out validation text?

The expected outcome is ordinary held-out text that round-trips without unknown
bytes and a reloadable saved tokenizer. The custom tokenizer is expected to need
**more** tokens per record than GPT-2. No superiority claim, perplexity claim, or
language-model accuracy claim is in scope.

## Working Contract To Freeze Before Running

These are the plan's proposed defaults. Freeze them before the measured run and
record the resolved values with the result.

| Item | Frozen choice |
| --- | --- |
| Corpus | `Salesforce/wikitext`, config `wikitext-2-raw-v1`, `raw` only |
| Corpus identity | Record the resolved revision, dataset fingerprints, and split row counts before fitting |
| Splits | Preserve upstream train/validation/test exactly; never re-split after chunking |
| Fitting subset | Up to 1 MiB of UTF-8 training text, whole nonempty train records in source order; record included row range and actual bytes |
| Custom tokenizer | Byte-level BPE, target vocabulary 2,048 including reserved tokens, complete 256-byte alphabet |
| Text policy | Preserve case, accents, and whitespace; no automatic prefix space, no text normalization |
| Special tokens | Distinct `<pad>` and `<eos>`; record assigned IDs and the literal-string policy |
| Reference | Frozen `openai-community/gpt2` tokenizer files only (no model weights, no fitting on this corpus) |
| Comparison text | The same fixed validation rows for both tokenizers, with no padding, truncation, or automatically inserted special tokens |
| Compute | CPU only |

The comparison is descriptive. The two tokenizers have different vocabulary
sizes and different training histories, so a token-count difference is not an
isolated algorithm result.

## Verified Prerequisites - Read-Only Checks On 2026-09-24

These were inspected, not downloaded. No corpus was fetched and nothing was
fitted during this scaffolding step.

| Check | Result |
| --- | --- |
| External data root | `/home/rui.gao/data/ai_research_roadmap` exists with `raw/`, `processed/`, `manifests/`, `checkpoints/`, `tmp/` |
| Session 2 artifact | Present at `processed/week09_tokenization_text_data/session2_readme/tokenizer.json` |
| WikiText-2 raw/processed paths from the [registry](../../data_registry/datasets.yaml) | Absent (`raw/huggingface/...` and `processed/wikitext2` do not exist) |
| WikiText-2 in the Hugging Face caches | Not found in `~/.cache/huggingface/hub` or `~/.cache/huggingface/datasets`; a cache-wide name search returned no match |
| Frozen GPT-2 tokenizer | Available offline in `~/.cache/huggingface/hub/models--openai-community--gpt2` (`tokenizer.json`, `vocab.json`, `merges.txt` present) |
| GPT-2 offline load | `Tokenizer.from_pretrained("openai-community/gpt2")` with `HF_HUB_OFFLINE=1` returned vocabulary size 50,257 and an exact round trip on a probe |
| Project environment | `.venv` Python 3.13.5 with `tokenizers` 0.22.2, `transformers` 5.12.1, `datasets` 5.0.0, `torch` 2.11.0+cu128, `huggingface_hub` 1.19.0 |
| Preparation helper | `scripts/data/prepare_standard_datasets.py --dataset wikitext2` exists and is `huggingface`-kind, but it pins no dataset revision, and its `--dry-run` still creates directories and a manifest |

Two consequences followed, and both were resolved later on 2026-09-24:

1. The reference side needed no download. The cached GPT-2 tokenizer was enough;
   no model weights were loaded.
2. The corpus side needed a **download** because WikiText-2 was absent from the
   checked caches. That fetch was authorized and run at a pinned revision
   `b08601e04326c79dfdd32d625aee71d232d685c3`, and the three raw split files were
   placed under `<AI_ROADMAP_DATA>/raw/huggingface/Salesforce/wikitext/wikitext-2-raw-v1/`.
   The split counts above were then re-measured from the downloaded files. The
   [registry](../../data_registry/datasets.yaml) path matches this location.

Illustrative only - not the validation comparison. On the probe
`"Hello, world!\n  Café 🙂"`, the cached GPT-2 tokenizer costs 8 tokens
(`['Hello', ',', 'Ġworld', '!', 'Ċ', 'Ġ', 'ĠCafÃ©', 'ĠðŁĻĤ']`), while the
Session 2 README tokenizer costs 22 tokens on the same string. Those numbers
describe one fixture, not held-out corpus text, and must not be reported as the
Session 3 result.

## Implemented Surface

[wikitext_compare.py](wikitext_compare.py) owns corpus access, the reference
tokenizer, and the comparison. It **imports and reuses** the existing Session 2
functions rather than duplicating them.

| Step | Responsibility | Output |
| --- | --- | --- |
| 1 | Resolve the corpus location and load the three upstream splits without re-splitting | `train`/`validation`/`test` row lists plus counts |
| 2 | Select the training-only fitting subset | Selected rows, excluded-row note, row range, actual UTF-8 bytes |
| 3 | Reuse `train_tokenizer` on the subset only | Fitted custom `Tokenizer`, separate from `session2_readme` |
| 4 | Reuse `check_round_trip` / `check_reserved_literal_policy` / `save_and_reload` | Saved training-only artifact plus reload equivalence |
| 5 | Load the frozen GPT-2 reference from cache | Reference `Tokenizer` (no weights) |
| 6 | Encode the identical fixed validation rows with both | Raw per-row token counts, no padding or truncation |
| 7 | Compute the measurement contract and write the comparison | Metrics table, largest-difference examples, observed failure cases |

Keep the fitting input and the comparison input strictly separate: the fitting
iterator must never receive validation or test text, and a held-out sentinel
fixture must not appear in the fitted vocabulary check.

## Measurement Contract

Report on the identical validation rows, alongside validation text identity and
UTF-8 byte counts:

- Actual custom vocabulary size and saved artifact size.
- Ordinary-text round-trip checks and the reserved-literal policy outcome.
- Total payload tokens and tokens per UTF-8 byte for each tokenizer.
- Median and 95th-percentile token length per record, with the row convention
  stated (one nonempty row is one record).
- Fraction of records longer than 128 tokens, as a context-budget diagnostic.
- Unsupported or unknown outputs, if any, plus examples involving spacing,
  punctuation, accents, non-Latin text, emoji, and unusual words.
- Measured fitting and encoding time with package versions and hardware.

An empty denominator is reported as not applicable, never as zero. Test text
stays out of fitting and out of tuning. Vocabulary-size changes are separate
runs, not edits to this comparison.

## Measured Result - 2026-09-24

Command: `.venv/bin/python phase02_llms/week09_tokenization_text_data/wikitext_compare.py`
from the repository root; exit status 0. Harness:
[wikitext_compare.py](wikitext_compare.py). Full record:
`<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session3_wikitext_trainonly/metrics.json`.

### Corpus

| Split | Rows | Nonempty records | UTF-8 bytes |
| --- | --- | --- | --- |
| train | 36,718 | 23,767 | 10,914,845 |
| validation | 3,760 | 2,461 | 1,144,248 |
| test | 4,358 | 2,891 | 1,287,656 |

Source `Salesforce/wikitext`, config `wikitext-2-raw-v1`, pinned revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. A record is one nonempty source row;
empty rows are blank lines, not empty documents. The test split was **not**
encoded for this comparison.

The training-only fitting subset was 2,222 nonempty records (source rows
1-3,465), 1,048,468 of the 1,048,576-byte cap. The byte cap stopped it at the
first row that would have exceeded 1 MiB. Fitting took 0.15 s on CPU and reached
the full 2,048-token target vocabulary. The saved artifact is 122,433 bytes at
`<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session3_wikitext_trainonly/tokenizer.json`;
the Session 2 README artifact is untouched.

### Comparison On Identical Validation Rows

Both tokenizers encoded the same 2,461 validation records (1,144,248 UTF-8
bytes), in source order, with no padding, truncation, or automatic special
tokens.

| Metric | Custom (training-only BPE) | Frozen GPT-2 |
| --- | --- | --- |
| Vocabulary size | 2,048 | 50,257 |
| Total payload tokens | 382,916 | 247,289 |
| Tokens per UTF-8 byte | 0.3346 | 0.2161 |
| Median tokens per record | 135 | 87 |
| 95th-percentile tokens per record | 414 | 264 |
| Max tokens per record | 727 | 471 |
| Records longer than 128 tokens | 1,259 (51.2%) | 880 (35.8%) |
| Unknown or unsupported outputs | 0 | 0 |
| Encoding time (2,461 rows) | 0.29 s | 0.30 s |

The custom tokenizer needed about **55% more tokens** (1.55x) to represent
exactly the same held-out text. Every validation record round-trips: the visible
prediction this feeds is that the same 128-token context window would hold
noticeably fewer words for the custom tokenizer, and a larger share of wiki
records would overflow it.

### Overlap Audit And Sensitivity

WikiText-2 repeats short nonempty lines across splits. Measured on this
revision: 243 validation records (9.9%) and 266 test records (9.2%) also appear
verbatim in train. Of the declared fitting subset, 153 rows (6.9%) are such
shared lines - generic wiki headings like `" = = Plot = = \n"`.

That is a real structural overlap, so it was measured rather than silently
removed. Refitting on the same declared subset with every held-out duplicate
removed (2,069 records, 1,044,992 bytes) changed the validation totals from
382,916 to **382,992 tokens (+0.02%)** and the median from 135 to 136. The
overlap does not meaningfully move the comparison. The declared split
membership was left unchanged, as the experiment plan requires.

### Failure And Edge Cases

- Largest per-row gaps are ordinary prose: `+289` on a sentence about an
  extradition from Germany, `+255` on a Roger Ebert review, `+252` on an
  architectural description. Longer records diverge most, which is what a
  2,048- versus 50,257-entry vocabulary predicts.
- WikiText-2 carries `@-@` and `@.@` markup from its preprocessing. The custom
  tokenizer spells these out as separate byte pieces, while GPT-2 has single
  merged tokens for them - a concrete example of why a token count is not a
  language-model quality measure.
- Byte-level coverage held for both: emoji and non-Latin text round-trip
  exactly, and neither tokenizer produced an unknown token. The custom tokenizer
  simply spends more tokens per unfamiliar character, e.g. the held-out fixture
  `"naïve — 日本語 café 🙂 42"` costs 26 custom tokens.

### What This Does Not Establish

Token counts are not accuracy or comparable perplexity. The two tokenizers have
different vocabulary sizes and different training histories, so this is a
descriptive comparison and not an isolated algorithm ablation. Lower custom
token counts were never required, and no superiority claim is made. The test
split remains untouched for a later fixed-protocol audit, and a tokenizer is not
automatically compatible with a model trained against different token IDs.

## Task Checklist

Checked against the [measured run](#measured-result---2026-09-24) below.

1. **Resolve and inspect corpus access.**
   - [x] Confirmed the source, pinned the revision, and recorded upstream split
     names and row counts. Rows are individual wiki lines/records, not complete
     articles, so the record convention is "one nonempty row is one record".
   - [x] Recorded the resolved revision and split counts in the run note and
     `metrics.json`.
2. **Declare the training-only fitting subset.**
   - [x] Selected whole nonempty training records in source order up to the byte
     cap; recorded source rows 1-3,465 and 1,048,468 actual bytes.
   - [x] Kept validation and test text out of fitting and out of every probe.
3. **Refit the custom tokenizer on that subset only.**
   - [x] Reused `train_tokenizer`; the Session 2 README artifact was neither
     edited nor overwritten. The new artifact went to its own run directory.
   - [x] Recorded the actual vocabulary size: 2,048, this time reaching the target.
4. **Re-verify text behavior on the new artifact.**
   - [x] Ordinary-text round trips, the reserved-literal `<pad>`/`<eos>` policy
     check, and save/reload equivalence all passed against the training-only
     tokenizer.
5. **Load the frozen reference.**
   - [x] Loaded the cached GPT-2 tokenizer files only; no model weights were
     loaded and the reference was not fitted on this corpus.
6. **Compare on identical validation text.**
   - [x] Encoded the same 2,461 validation records with both tokenizers, with no
     padding, truncation, or automatically inserted special tokens.
   - [x] Computed the contract quantities and built the comparison table from
     measured numbers.
7. **Inspect failure and edge cases.**
   - [x] Recorded the largest per-row token gaps and the byte-level edge cases;
     see [Failure and edge cases](#failure-and-edge-cases).
8. **Record and explain.**
   - [x] Wrote the comparison into [RUN_LOG.md](RUN_LOG.md) and
     [RESULTS.md](RESULTS.md), and audited the cross-split duplicate overlap.
   - [x] Stated what the comparison shows and what it cannot establish.

## Finish Condition

**Session 3 is complete only when every row below is evidenced:**

| Required outcome | Evidence needed |
| --- | --- |
| Verified corpus and original splits | Recorded revision/fingerprints, split counts, and the fitting-subset row range; no validation or test text in the fitting iterator |
| Training-only refit | A saved tokenizer whose fitting input is the declared training subset, separate from the README prototype, with its actual vocabulary size |
| Text behavior retained | Ordinary round trips, the reserved-literal policy, and reload equivalence all pass on the new artifact |
| Frozen reference | GPT-2 tokenizer loaded from cache using tokenizer files only; both tokenizers see identical validation rows |
| Measured comparison | A table with the contract's metrics from a real run; empty denominators reported as not applicable |
| Failure and edge cases | Recorded examples, including where the custom tokenizer needs more tokens, with no hidden failures |
| Learner understanding | A short explanation of what the comparison does and does not show |

If any row lacks evidence, record Session 3 as **in progress** and name the
missing item. Reaching exactly 2,048 tokens, beating GPT-2 on token count,
training a language model, and building LM chunks or masks are **not** finish
conditions. LM chunks, next-token targets, and masks belong to
[Session 4](README.md#five-session-plan); the Week 9 lab note belongs to
Session 5.

Current verdict: **complete (2026-09-24).** Every row is evidenced: the corpus
was fetched at a pinned revision and its splits were counted, the training-only
subset is declared with its row range and bytes, the refit reached 2,048
tokens in a separate artifact, the text-behavior and reload checks passed, the
frozen reference was loaded without weights, and the comparison table holds
measured numbers. The learner-understanding row is evidenced by the dated
[learning record](#learning-record---2026-09-24) below, which separates what the
learner demonstrated from what was supplied. The remaining Week 9 work is
Session 4's LM chunks, targets, and masks.

## Learning Record - 2026-09-24

A guided check of the session's learning points, run after the measured result.
It is not an exam and does not establish independent mastery. The learner's
reasoning is recorded alongside the guidance needed to complete it.

| Topic | Learner evidence | Guidance and assessment limit |
| --- | --- | --- |
| Tokenizer-level leakage | Identified that fitting on validation text is leakage and makes the reported number "not real". | The mechanism and the fairness asymmetry were supplied. Held-out text stops being held out, the reference tokenizer never saw this corpus, and better merges for the evaluation text would *shrink* the observed gap - so leakage flatters the custom tokenizer. Tokenizer-level, model-level, and test-set leakage remain distinct. |
| Unit of comparison (tokens per byte) | No independent rationale was produced. | Supplied: validation records range from 4 to 727 tokens, so a per-record figure mixes record length with segmentation; UTF-8 bytes are the invariant denominator, giving ~3.0 vs ~4.6 bytes per token. Bytes measure volume, not information. |
| Confound boundary | Proposed that GPT-2 wins on coverage and generalizability while a matched distribution could close the gap. | The prediction is contradicted by the run itself: the custom tokenizer was fitted and evaluated on the same domain and still cost 55% more tokens. The second half - what must be given up - was not produced: no causal attribution to vocabulary size, algorithm, or training text is supported. |
| Fixed-budget reallocation | Predicted fewer validation tokens from the larger training split, then explained a possible increase by slots going to pieces frequent across the whole corpus. | Correct in outline. "Niche" was corrected to *frequency rank within the fitting corpus*; the trainer does not remember strings, and test text cannot enter this reasoning. A worse validation score from the fuller fit would indicate lucky slice/validation alignment, not a better tokenizer. |
| Overlap handling | Recognized the duplicated short lines as real distribution whose removal changes tokenization. | The design consequences were supplied: deleting rows silently redefines the benchmark, the deletion decision would itself be informed by held-out text, and it removes an effect the sensitivity arm measured at +0.02%. |
| Claim scope (consolidation) | Stated the result cannot be generalized because the settings differ and the corpus matters. | Tightened: name vocabulary size and training text as the two confounded variables, note the missing vocabulary sweep, and note that quality was never measured at all. |

Learner demonstration: 1 of 6 topics independently complete (leakage
identification), 2 partial (fixed-budget reallocation, overlap handling), 1
supplied by the instructor (unit of comparison), and 2 thin (confound boundary,
claim scope).

Recurring gap: the learner reasons well about *why a number moves* - dilution,
high-frequency lines, merge re-ranking all came unprompted - but tends to skip
*what the number licenses one to claim*, which is the specific skill this
session's comparison was designed to build. Revisit it when Week 15 and Week 16
run their controlled comparisons; no repeat quiz is scheduled.

## First Step (Completed) And Next Steps

The first step was resolving corpus access, and it settled the one real blocker:
the reference side was already available offline, so only the corpus needed a
fetch. The WikiText-2 raw splits were downloaded directly from Hugging Face at a
pinned revision rather than through
`scripts/data/prepare_standard_datasets.py --dataset wikitext2`, because the
helper pins no revision and its dry-run still writes directories and a manifest.

Two follow-ups stay open and belong to the instructor's bookkeeping, not to the
learner's checklist:

1. Record the WikiText-2 download in the manifest/provenance surfaces, including
   the resolved revision and file sizes, and reconcile the registry's
   `processed: processed/wikitext2` path with the Week 9 run directory.
2. Carry the `@-@`/`@.@` markup observation into Session 4, since it affects how
   much of the corpus is spent on preprocessing artifacts rather than words.

References: [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext),
[GPT-2 tokenizer reference](https://huggingface.co/openai-community/gpt2),
[tokenizers quicktour](https://huggingface.co/docs/tokenizers/quicktour).
