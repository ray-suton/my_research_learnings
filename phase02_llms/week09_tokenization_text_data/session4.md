# Session 4 - Split-Specific LM Chunks, Targets, And Masks

## Status

**Deferred by learner decision on 2026-09-25, then rescheduled on 2026-09-26 to
be split across Weeks 9 and 10.** The contract, checklist, finish condition,
learner scaffold ([lm_data.py](lm_data.py)), and a red fixture spec
([tests/test_session4_lm_data.py](tests/test_session4_lm_data.py)) exist. No
implementation has run: no token stream, no window, no mask, and no passing
check.

**Redrawn 2026-09-28:** Week 9's deliverable is *a training-only tokenizer and
the per-split token streams it produces*, so only `build_split_stream` is owed
against Week 9. Everything below that builds windows, targets, and masks is
**Week 10's deliverable** - the prepared LM dataset - not a Week 9 carry-over.
The split stays; the promise moved to match it.

Schedule, recorded so this cannot drift into limbo. The split follows the
function that touches the tokenizer/corpus contract from the functions that build
model-facing examples:

- **Week 9 half - the stream.** `build_split_stream`: encode nonempty records in
  stable order with one EOS appended per record, and pass the stream-level
  fixture rows. This is pure data-layer work over the Session 3 training-only
  artifact; it borrows nothing from Week 10.
- **Week 10 half - the examples.** `build_source_windows` and `make_lm_batch`:
  `T + 1` windows at stride `T`, the short-final-window padding decision, the
  lone-trailing-token record, one-shifted targets with `-100`, and the distinct
  `attention_mask` / `loss_mask`, plus the fixture gate and the `RUN_LOG.md` /
  `RESULTS.md` record. This half **is Week 10's deliverable** - the prepared LM
  dataset - built with the attention block, where shapes and mask semantics
  become concrete.
- **Week 11 gate, unchanged.** Mini-GPT training cannot run until both halves are
  done, and targets and masks are only ever validated against a real
  cross-entropy loss there.

Week 9's deliverable was **redrawn on 2026-09-28** (learner decision) to *a
training-only tokenizer and the per-split token streams it produces*. The
tokenizer side is complete and measured; the stream half is the single item still
owed against that redrawn deliverable; and the prepared LM dataset - this
session's example half - is **Week 10's deliverable outright**. The original
Week 9 line ("tokenizer and prepared LM dataset") is what made the week look half
done; the redraw moves the promise to match the work rather than moving the work
to protect an over-wide promise.

Two consequences accepted when the split was recorded:

- Cost accepted: building data and model in the same week makes a flat loss
  ambiguous between a data-alignment bug and a model bug. The fixtures, plus
  moving the stream half out of the model week, exist to keep those separable.
- Benefit accepted: masks and targets acquire concrete meaning once a
  cross-entropy loss exists, which is why the mask half travels with the model
  work instead of with the tokenizer work.

[Session 3](session3.md) closed on 2026-09-24 with a measured comparison and a
recorded learning check. The tokenizer side of Week 9 is done; this session turns
token IDs into training examples.

## Where This Sits

```text
text -> tokenizer -> integer token IDs -> [SESSION 4] -> next-token examples -> model (Week 10+)
```

Session 2 taught what a tokenizer is. Session 3 measured one. Session 4 builds
the data an actual language model consumes. Session 5 reviews results and writes
the Week 9 lab note.

## Handoff From Session 3

Available to reuse:

- The training-only tokenizer artifact at
  `<AI_ROADMAP_DATA>/processed/week09_tokenization_text_data/session3_wikitext_trainonly/tokenizer.json`
  (2,048 entries, PAD 0 / EOS 1, reload-equivalent).
- The pinned corpus at
  `<AI_ROADMAP_DATA>/raw/huggingface/Salesforce/wikitext/wikitext-2-raw-v1/`
  (revision `b08601e04326c79dfdd32d625aee71d232d685c3`).
- The declared record convention: one nonempty source row is one record; ~35% of
  raw rows are blank lines.
- Reusable functions in [tokenizer.py](tokenizer.py) and the harness pattern in
  [wikitext_compare.py](wikitext_compare.py).

Not yet available: streams, windows, `input_ids`, `targets`, `attention_mask`,
`loss_mask`, or any executable check of them.

## The Contract To Implement

Copied from the [experiment plan](EXPERIMENT_PLAN.md#lm-example-contract) - that
file remains the authority if these ever diverge.

- For each split **independently**, encode nonempty records in stable order,
  append one EOS per record, and form a continuous token stream.
- Packing may cross row boundaries **within** a split. The simple causal model
  may attend to earlier records in its block; document-isolated attention is
  not claimed.
- Take windows of `T + 1` stream tokens with stride `T`. The one-token overlap
  preserves successive prediction pairs across windows.
- Right-pad a final short window if it contains at least one valid
  input/target pair; log a lone trailing token that has no target.
- Carry remainders across preprocessing batches rather than silently dropping
  each batch's tail.

| Quantity | Shape | Rule |
| --- | --- | --- |
| Source window | `(B, T + 1)` | Split-specific real tokens plus optional right padding |
| `input_ids` | `(B, T)` | Source positions `0:T`; integer vocabulary IDs |
| `targets` | `(B, T)` | Source positions `1:T + 1`; invalid targets replaced by `-100` |
| `attention_mask` | `(B, T)` | 1 at actual input positions, 0 at padding |
| `loss_mask` | `(B, T)` | 1 only where both input and next-token target are real |

Frozen defaults: `T = 128` (block length), stride `T` (128), fixture batch size
`B = 2`, `PAD = 0`, `EOS = 1`, `IGNORE_INDEX = -100`, CPU only.

Fixture checks use plain Python lists of lists for the `(B, T)` quantities, so
Session 4 needs no tensors. Converting examples to tensors belongs to the model
work in Week 10, not here.

Real EOS targets stay trainable. Padding is identified by position and length.
These targets are **already shifted** for a manual training loop; a model API
that shifts labels internally needs an adapter, not a second shift. Week 10
supplies causal attention separately.

### Worked Illustration

Hand-written, not tokenizer output. Take a stream `[5, 6, 1]` with `T = 3`, where
`1` is EOS. The window needs 4 tokens, so it is right-padded:

```text
Source:       5      6      1      PAD
input_ids:    5      6      1
targets:      6      1      -100
attention:    1      1      1
loss_mask:    1      1      0
```

The `5 -> 6` and `6 -> 1` predictions are trainable, so stopping is still
learned. The padded position contributes no loss. If the stream were `[5, 6, 7]`
with a record boundary after `7`, the EOS append would make it `[5, 6, 7, 1]` and
the same window would be full with no padding and loss `[1, 1, 1]`.

## Planned Surface

A new module, [lm_data.py](lm_data.py), owns stream building, windowing,
shifting, and masks. It should import and reuse the existing Session 2 and
Session 3 code rather than duplicating the tokenizer.

The file already exists as a **learner scaffold**: signatures and docstrings,
with `NotImplementedError` bodies. That mirrors how Session 2 started. The
instructor reviews; the learner implements.

The contract is also encoded as executable checks in
[tests/test_session4_lm_data.py](tests/test_session4_lm_data.py). Those fixtures
are the specification: they currently fail with `NotImplementedError`, and each
task below is finished when its fixture goes green without weakening the
assertion. Run them with
`.venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py`.

## Task Checklist

Check an item only after its evidence exists.

1. **Build the split-specific stream.**
   - [ ] Encode nonempty records in stable order with the Session 3 training-only
     tokenizer and append one EOS per record.
   - [ ] Verify the stream length equals the summed per-record token counts plus
     the record count, and that two runs give the same stream.
2. **Cut windows correctly.**
   - [ ] Produce windows of `T + 1` tokens at stride `T`; confirm consecutive
     windows share exactly one token.
   - [ ] Right-pad the final short window only when it holds at least one valid
     input/target pair, and log a lone trailing token that has no target.
3. **Shift once, and only once.**
   - [ ] Build `input_ids` from positions `0:T` and `targets` from `1:T + 1`.
   - [ ] Replace invalid targets with `-100` and keep real EOS targets
     trainable. State clearly that these targets are pre-shifted.
4. **Build both masks.**
   - [ ] `attention_mask` is 1 at real input positions and 0 at padding.
   - [ ] `loss_mask` is 1 only where both the input and its next token are real,
     so it is not a copy of `attention_mask`.
5. **Prove split isolation.**
   - [ ] Show that no window mixes train, validation, and test text, and that
     packing crosses row boundaries only within a split.
6. **Prove invariance and no silent loss.**
   - [ ] Show that changing the preprocessing batch size does not change the
     resulting stream or examples.
   - [ ] Show that no remainder or tail is dropped between batches.
7. **Pass the hand-written fixture checks.**
   - [ ] Cover a full block, a short final block, a single-token input, a block
     ending at EOS, and an empty input, asserting shapes, targets, and masks.
8. **Record and explain.**
   - [ ] Record counts, shapes, one worked example, check outcomes, and failures
     in [RUN_LOG.md](RUN_LOG.md) and [RESULTS.md](RESULTS.md).
   - [ ] Explain which check would catch a double-shifted target and which would
     catch a loss mask that hides a real EOS target.

## Finish Condition

**Session 4 is complete only when every row below is evidenced:**

| Required outcome | Evidence needed |
| --- | --- |
| Split-specific streams | One stream per split with recorded token counts, EOS appended per record, and stable re-run output |
| Correct windowing | Windows of `T + 1` at stride `T`, verified one-token overlap, and a recorded decision for a lone trailing token |
| One correct shift | `input_ids` and `targets` aligned to positions `0:T` and `1:T + 1`; invalid targets `-100`; real EOS targets retained |
| Two distinct masks | `attention_mask` and `loss_mask` built by their own rules and shown to differ where padding or a next-token-free position exists |
| Split isolation | A check showing no window crosses a split boundary |
| Invariance evidence | Batch/stride settings change nothing; no tail is dropped |
| Passing fixture checks | The hand-written cases above pass, with output retained |
| Learner understanding | A short explanation of the shift, the two masks, and what each check catches |

If any row lacks evidence, record Session 4 as **in progress** and name the
missing item. Training a model, implementing causal attention, reaching a
specific loss, and writing the Week 9 lab note are **not** finish conditions;
attention is Week 10 and the lab note is Session 5.

Per-half finish condition: the **stream half** is done when row 1 is evidenced
(that row is Week 9 work), and the **example half** is done when rows 2-8 are
evidenced (those rows are Week 10 work). Week 11 cannot run until both halves
are done.

Current verdict: **deferred then split, not started (2026-09-25, rescheduled
2026-09-26).** Only the contract, the checklist, the learner scaffold, and the
red fixture spec exist. The stream half carries over against Week 9's
deliverable; the example half is scheduled with Week 10's attention block.

## Common Traps

- Shifting an already shifted target a second time.
- Treating `attention_mask` as the `loss_mask`, or the reverse. They differ at
  the last real position before padding.
- Assuming EOS isolates records from attention. In this packing, an earlier
  record is still visible.
- Dropping each processing batch's tail instead of carrying the remainder.
- Treating one source row as an entire article; rows are lines, and ~35% of raw
  rows are blank.
- Reading `-100` as a vocabulary ID. It is a loss-ignore marker, not a token.

## First Learning Step

Implement `build_split_stream`, `build_source_windows`, and `make_lm_batch`
against the hand-written fixtures in the checklist **before** touching
WikiText-2. Get `[5, 6, 1]` with `T = 3` producing the padded example above, and
get a full block producing all-ones `loss_mask`. Only then point it at the real
corpus, where mistakes are much harder to see.

References: [causal LM guide](https://huggingface.co/docs/transformers/tasks/language_modeling),
[padding/truncation guide](https://huggingface.co/docs/transformers/main/en/pad_truncation).
