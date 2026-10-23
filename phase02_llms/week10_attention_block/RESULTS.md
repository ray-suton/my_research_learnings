# Results

Measured implementation status: **Week 10 is closed (2026-10-05).** Both
deliverables are implemented and green. Attention passes 15/15 checks, multi-head
attention 9/9, the assembled block 11/11, and the prepared LM dataset 9/9; all
four suites were re-run on 2026-10-05. The block smoke run reports 12,704
parameters at the frozen fixture sizes, and the data pipeline produces 3,682,072
/ 385,377 / 436,253 stream tokens and 28,767 / 3,011 / 3,409 windows for the
train / validation / test splits. No model has been trained, so there is no
perplexity, accuracy, or quality claim anywhere in this file. Session 5's review,
lab note, and Week 11 handoff are recorded in [session5.md](session5.md) and
[LAB_NOTE.md](LAB_NOTE.md).

Use the [experiment plan](EXPERIMENT_PLAN.md) for measurement definitions. When
runs exist, link their settings, exact commands, metrics, and artifacts.
Distinguish observed behavior from assumptions, and guided understanding from
independent demonstration. Do not insert estimated results into measured tables.

## Deliverable Status

| Deliverable | Current evidence |
| --- | --- |
| Session 1 understanding | Closed 2026-09-28 as a guided conceptual checkpoint; the [discussion record](session1.md#discussion-record) separates learner reasoning from instructor explanation. No independent mastery claimed |
| Hand-written scaled dot-product attention and causal mask | **Implemented 2026-09-29; 15/15 fixture checks pass.** `stable_softmax`, `build_causal_mask`, `padding_to_mask`, `scaled_dot_product_attention`, and `main()` all written. Smoke run prints shapes `(2, 4, 8, 8)`, row sums 1.0, 248 of 512 weights exactly zero |
| Hand-written multi-head attention | **Implemented 2026-09-30; 9/9 fixture checks pass.** `MultiHeadAttention` in [multihead.py](multihead.py): four `nn.Linear(d_model, d_model)` projections, head split and merge, mask passed through. Smoke run prints `x (2, 8, 32)` -> `out (2, 8, 32)`, `H=4`, `d_head=8`, 4224 parameters |
| Residual, layer norm, and MLP assembled into a block | **Implemented 2026-10-01; 11/11 fixture checks pass.** The block uses hand-written layer norm, a GELU MLP, two pre-norm residual additions, mask forwarding, and a gradient-presence check. |
| Shape, mask, causality, and gradient checks | **15/15 attention, 9/9 multi-head, and 11/11 block checks pass.** The suites cover future-token perturbation, independent per-head equivalence, pre-norm composition, residual preservation, and parameter gradient presence. |
| Prepared LM dataset (carry-over) | **Implemented 2026-10-05; 9/9 fixture checks pass.** [lm_data.py](../week09_tokenization_text_data/lm_data.py) builds per-split streams with one EOS per nonempty record, `T + 1` windows at stride `T`, one-shifted targets, and separate attention and loss masks. Real-corpus run recorded in [Week 9 RUN_LOG](../week09_tokenization_text_data/RUN_LOG.md) |
| Lab note and Week 11 handoff | **Closed 2026-10-05.** [LAB_NOTE.md](LAB_NOTE.md) records reproduction, what the checks do and do not establish, and the limitations; [session5.md](session5.md) records the review, the verification pass, and the Week 11 handoff contract |

## Session 2 Run - 2026-09-29

Command, run from the repository root:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py
```

Result: **15/15 checks passed**, exit 0.

Settings: seed 0, `float32`, CPU. Most fixtures use `B=2`, `H=2`, `T=4`,
`d_head=2`; the scaling checks use `d_head=16` and `d_head=64`. Torch 2.11.0,
Python 3.13.

Smoke run at the module sizes (`B=2`, `H=4`, `T=8`, `d_head=8`), with the last
two positions of every row marked padding and the causal and padding masks
combined by addition:

```text
q, k, v          (2, 4, 8, 8)
mask             (2, 1, 8, 8)
out              (2, 4, 8, 8)
weights          (2, 4, 8, 8)
row-sum range    1.000000 .. 1.000000
exact-zero weights  248 of 512
```

The row sums confirm the weights are distributions; the 248 exact zeros come
from the causal triangle plus the two padded columns. Neither number is a
quality measure - no model was trained.

Supporting measurement, loop versus vectorized `padding_to_mask`, 200
repetitions per size, identical output at every size:

| shape | loop | vectorized |
| --- | --- | --- |
| `(2, 4)` | 0.065 ms | 0.0059 ms |
| `(16, 129)` | 14.739 ms | 0.0081 ms |
| `(128, 129)` | 118.434 ms | 0.0264 ms |

## Session 4 Run - 2026-10-01

The learner implemented the hand-written layer norm, position-wise MLP, and
pre-norm Transformer block. The instructor added only the requested smoke
runner. `attention.py` and `multihead.py` were left unchanged.

Fixture command:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py
```

Result: **11/11 checks passed**, exit 0. The checks cover feature-axis
normalization, constant-input stability, trainable affine parameters, MLP
nonlinearity, pre-norm composition, residual preservation, gradient presence,
and mask forwarding.

Smoke command:

```text
.venv/bin/python phase02_llms/week10_attention_block/block.py
```

```text
x               (2, 8, 32)
out             (2, 8, 32)
parameters      12704
mean |out - x|  0.259858
```

The unchanged attention and multi-head suites were re-run after the block:
**15/15** and **9/9** checks passed. This verifies the hand-written block at
fixture scale; it does not establish training quality or mini-GPT readiness.

## Week 10 Closure - 2026-10-05

Session 5 reopened every suite rather than trusting the Sessions 2-4 summaries,
re-ran the carry-over data pipeline on the pinned corpus, reviewed the module
sources against the week's "hand-written only" rule, and wrote
[session5.md](session5.md) and [LAB_NOTE.md](LAB_NOTE.md).

Re-verification, run from the repository root on 2026-10-05, Python 3.13,
PyTorch 2.11.0, CPU, seed 0:

| Suite | Result | Exit |
| --- | --- | --- |
| `tests/test_attention.py` | 15/15 | 0 |
| `tests/test_multihead.py` | 9/9 | 0 |
| `tests/test_block.py` | 11/11 | 0 |
| `week09.../tests/test_session4_lm_data.py` | 9/9 | 0 |

The block smoke run still reports `x (2, 8, 32)` -> `out (2, 8, 32)`, 12,704
parameters, `mean |out - x| 0.259858`.

The prepared-dataset run on `b08601e0` WikiText-2:

| Split | Rows | Records | Stream tokens | Windows | Padded final |
| --- | --- | --- | --- | --- | --- |
| train | 36,718 | 23,767 | 3,682,072 | 28,767 | True |
| validation | 3,760 | 2,461 | 385,377 | 3,011 | True |
| test | 4,358 | 2,891 | 436,253 | 3,409 | True |

Independent cross-check: the validation stream is 382,916 (the Session 3
measured custom-tokenizer token count on the same 2,461 records) plus 2,461 EOS
tokens, which is what "one EOS per nonempty record" predicts. Every split reports
`lone_trailing_token=None`, so the leftover-token branch has fixture coverage
only.

### Verdict

**Week 10 passes on correctness, at fixture scale, with no training.** The block
contract is implemented and tested: causality is verified by future-token
perturbation rather than by mask inspection, multi-head matches an independently
written per-head reference, pre-norm composition is exact, gradients reach every
parameter, and masked rows do not produce `NaN`. The prepared LM dataset closes
with green fixtures and a real-corpus run whose counts cross-check against
Session 3. No library shortcut appears in the module sources.

### Limitations

- No model was trained; there is no perplexity, accuracy, or sample here.
- All block fixtures are small (`T = 8`), so nothing here is evidence at
  `T = 128` or at depth.
- The block is position-agnostic and contains no embedding, LM head, or loss;
  that is Week 11.
- A literal `<eos>` in a source row would split a record, and
  `lone_trailing_token` is fixture-only. Both are contract edge cases recorded
  in [LAB_NOTE.md](LAB_NOTE.md), not observed defects.
- "Hand-written only" was verified by source review, not by a runtime trap.

### Reproduction

```bash
export AI_ROADMAP_DATA="$HOME/data/ai_research_roadmap"
.venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_multihead.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py
.venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
.venv/bin/python phase02_llms/week10_attention_block/block.py
.venv/bin/python phase02_llms/week09_tokenization_text_data/lm_data.py
```

Week 11 inherits the frozen block contract, the mask conventions, and the
prepared dataset; training is the next step, not a Week 10 claim.
