# Run Log

## Initialization - 2026-09-28

- Created the Week 10 documentation scaffold: this run log, the
  [README](README.md), the [experiment plan](EXPERIMENT_PLAN.md),
  [Session 1](session1.md), [RESULTS.md](RESULTS.md), and the
  [configs](configs/README.md), [tests](tests/README.md), and
  [results](results/README.md) guides.
- Linked the project from the Phase 2 overview and roadmap, the root status
  surfaces, and Week 9's next-action line.
- Recorded the carry-over: the prepared LM dataset
  ([Week 9 Session 4](../week09_tokenization_text_data/session4.md)) is now this
  week's deliverable rather than a Week 10 carry-over.
- Reviewed the official attention references; they are linked in the experiment
  plan and Session 1.

No code was written, no model was implemented or run, no test was executed, and
no measurement was taken during initialization. Week 9's tokenizer results,
Week 8's GAN results, and Week 7's VAE results are not Week 10 verification.

## Session 1 Discussion - 2026-09-28

- Held the Session 1 conceptual discussion: seven questions, asked one at a time,
  with corrections between each.
- Recorded the outcome in
  [session1.md](session1.md#discussion-record), separating learner reasoning from
  instructor explanation.
- Topics covered: Q/K/V roles and the asymmetry argument, causal versus padding
  masks and the dropped-mask failure mode, where padding occurs in this pipeline,
  the `1/sqrt(d_head)` scale and its two failure modes, multi-head input sharing
  and parameter separation, permutation-equivariance, the residual stream, the
  MLP's role, and layer norm's per-position statistics.
- Two recurring gaps recorded rather than smoothed: answering an adjacent
  question instead of the one asked, and reaching for CNN or batch intuitions
  where the operation is strictly per-token.

Documentation only. No code was written, no model was implemented or run, and no
measurement was taken. Session 2 - scaled dot-product attention and the causal
mask - is next.

## Session 2 Scaffold - 2026-09-29

- Created [session2.md](session2.md): the four-function contract, a
  hand-computed illustration (`T = 2`, `d_head = 1`, uniform and causal panels), a
  task checklist, a finish condition, the common traps, and the Session 3
  handoff.
- Created [attention.py](attention.py) as a learner scaffold: frozen fixture
  constants, four signatures with docstrings, and `NotImplementedError` bodies.
- Created [tests/test_attention.py](tests/test_attention.py): 14 hand-written
  checks covering softmax stability, both mask builders, output and weight
  shapes, row-sum and exact-zero-weight properties, two hand-computed examples,
  the future-token perturbation causality test, the masks-must-be-summed check,
  the fully masked row, and the `sqrt(d_head)` scaling.
- Verified the scaffold is red for the right reason: the first check raises
  `NotImplementedError` from the stub.
- Verified the fixture spec is a sound target rather than a broken test: a
  throwaway reference implementation, held only in a scratch process and never
  written into the repo, passes **14/14**.

No attention computation has been recorded against the repository's own code.
`attention.py` is unimplemented and the fixture spec is red until the learner
writes the bodies. Nothing was measured, and no model was run.

## Session 2 Implementation - 2026-09-29

Learner-written bodies, reviewed function by function:
`build_causal_mask`, `stable_softmax`, `padding_to_mask`,
`scaled_dot_product_attention`, and `main()`.

Fixture spec, run from the repository root:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py
```

Result: **15/15 checks passed**, exit 0. Shapes, row sums, exact-zero masking,
both hand-computed examples, the future-token perturbation causality test, the
masks-must-be-summed check, the fully masked row, and both scaling checks all
green.

Smoke run:

```text
.venv/bin/python phase02_llms/week10_attention_block/attention.py
```

```text
q, k, v          (2, 4, 8, 8)
mask             (2, 1, 8, 8)
out              (2, 4, 8, 8)
weights          (2, 4, 8, 8)
row-sum range    1.000000 .. 1.000000
exact-zero weights  248 of 512
```

Settings: seed 0, `float32`, CPU, `B=2`, `H=4`, `T=8`, `d_head=8`. Mask
convention: additive `-inf`, causal and padding summed. Torch 2.11.0, Python
3.13.

Two defects found and repaired during the session, both recorded rather than
smoothed:

1. **A loose scaling check.** The saturation test alone did not discriminate:
   dividing by `sqrt(D_HEAD)` (the module constant, 8) instead of the tensor's
   `sqrt(d_head)` still left the peak at 0.9692 against a 0.999 threshold. A
   check comparing the weights against a reference built from
   `sqrt(q.shape[-1])` was added and **verified to fail under the wrong
   constant** while the saturation check passes under it.
2. **A shape-source mistake, repeated.** Module constants were used as though
   they described the incoming tensor, in `padding_to_mask` and again in the
   attention scaling. The constants are smoke-run configuration; every function
   reads its shapes from its arguments.

Cost note, measured: the loop form of `padding_to_mask` is ~4,500x slower than
the vectorized form at `(128, 129)` and produces identical output.

## Session 3 Scaffold - 2026-09-30

- Created [session3.md](session3.md): the multi-head contract, shape
  walkthrough, checklist, finish condition, traps, and the Session 4 handoff.
- Created [multihead.py](multihead.py) as a learner scaffold:
  `MultiHeadAttention` with `NotImplementedError` bodies, plus a `main()`
  placeholder. `attention.py` from Session 2 was **not touched**.
- Created [tests/test_multihead.py](tests/test_multihead.py): 9 checks covering
  output shape and dtype, the `d_model % n_heads` validation, a single-head
  configuration, equivalence against a hand-written per-head reference (with and
  without the combined causal and padding mask), causal blocking through the
  layer, padded columns being unreadable, batch-row independence, and
  repeatability.
- Verified the scaffold is red for the right reason: check #1 raises
  `NotImplementedError` at `multihead.py:63`.
- Verified the spec is a sound target: a throwaway reference implementation, held
  only in a scratch process and never written to the repo, passes **9/9**.
- Verified `attention.py` is unchanged: Session 2's spec still reports 15/15.

The reference in the fixture spec deliberately builds each head with explicit
slicing (`q[..., i*d_head:(i+1)*d_head]`) rather than reusing any split or merge
helper from the implementation. A reference that shared that code could not
catch a bug in it.

No Session 3 implementation exists yet. Everything above is scaffold and
verification.

## Session 3 Implementation - 2026-09-30

Learner-written `MultiHeadAttention` in [multihead.py](multihead.py), with
instructor corrections across six review rounds. `attention.py` was not touched.

Fixture spec, run from the repository root:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_multihead.py
```

Result: **9/9 checks passed**, exit 0.

Smoke run:

```text
.venv/bin/python phase02_llms/week10_attention_block/multihead.py
```

```text
x           (2, 8, 32)
mask        (1, 1, 8, 8)
out         (2, 8, 32)
H, d_head   4, 8
parameters  4224
```

Settings: seed 0, `float32`, CPU, `d_model=32`, `n_heads=4`, `d_head=8`, four
`nn.Linear` projections with default bias. Torch 2.11.0, Python 3.13.

Session 2's spec re-run after this session: **15/15**, confirming `attention.py`
was left alone.

Bugs found and repaired during the session, in the order they surfaced:

1. `nn` used without importing it.
2. `self.d_heads = d_model / n_heads` - wrong name against the fixture contract,
   and true division producing a float where a shape needs an int.
3. `self.w_o` renamed to the contract's `self.w_out`.
4. `q.shape()` - attribute called as a method.
5. **The missing transpose pair.** `sdpa` needs `(B, H, T, d_head)`; the code
   passed `(B, T, H, d_head)`, so attention ran over the *head* axis with
   positions as the batch. Output shape still came out correct. Verified
   separately that a missing *backward* transpose changes the output by 0.4 at
   H = 2 while being undetectable at H = 1.
6. `sdpa`'s `(out, weights)` tuple not unpacked.

Test-coverage finding, measured: `test_single_head_is_a_valid_configuration`
exercises nothing about the head split or the transposes. With `H = 1`, a missing
backward transpose produces an **identical** output, because swapping a size-1
axis against a size-`T` axis reorders nothing. The H=2 fixtures carry that
weight.

## Session 4 Implementation - 2026-10-01

The learner implemented `LayerNorm`, `FeedForward`, and `TransformerBlock` in
`block.py`. The implementation keeps the frozen pre-norm order, registers
`gamma`/`beta` as parameters, applies layer norm over the final feature axis,
and forwards the additive mask unchanged. The instructor added only the
requested `main()` smoke runner after the learner implementation was complete.
`attention.py` and `multihead.py` were not changed.

Fixture command, run from the repository root:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py
```

Result: **11/11 checks passed**, exit 0.

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

Settings: seed 0, `float32`, CPU, `B=2`, `T=8`, `d_model=32`, `n_heads=4`,
`d_ff=128`, causal additive mask. The unchanged Session 2 and Session 3
fixtures were re-run afterward: **15/15** and **9/9** checks passed.

Review corrections recorded during the learner implementation: registered
normalization parameters, retained the reduced feature axis for broadcasting,
used tensor square root with `eps` inside it, corrected the attention attribute
name to `attn`, and preserved the exact pre-norm residual composition.

The post-code discussion also distinguished a connected parameter with a zero
gradient from a disconnected parameter. `attn.w_k.bias` had zero norm in one
backward inspection because a shared key-bias shift cancels inside softmax; the
fixture's gradient check still confirmed `grad is not None` for every parameter.

## Session 5 Closure - 2026-10-05

Review and verification pass. No module source was changed this session.

All four suites re-run from the repository root, Python 3.13, PyTorch 2.11.0,
CPU, seed 0:

```text
.venv/bin/python phase02_llms/week10_attention_block/tests/test_attention.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_multihead.py
.venv/bin/python phase02_llms/week10_attention_block/tests/test_block.py
.venv/bin/python phase02_llms/week09_tokenization_text_data/tests/test_session4_lm_data.py
```

| Suite | Result | Exit |
| --- | --- | --- |
| Attention | 15/15 | 0 |
| Multi-head | 9/9 | 0 |
| Block | 11/11 | 0 |
| Prepared LM dataset | 9/9 | 0 |

Block smoke run, `.venv/bin/python phase02_llms/week10_attention_block/block.py`:

```text
x               (2, 8, 32)
out             (2, 8, 32)
parameters      12704
mean |out - x|  0.259858
```

Carry-over data run, `AI_ROADMAP_DATA=/home/rui.gao/data/ai_research_roadmap
.venv/bin/python phase02_llms/week09_tokenization_text_data/lm_data.py`:

```text
train       rows= 36718 records= 23767 stream= 3682072 windows= 28767 padded_final=True lone_trailing=None
validation  rows=  3760 records=  2461 stream=  385377 windows=  3011 padded_final=True lone_trailing=None
test        rows=  4358 records=  2891 stream=  436253 windows=  3409 padded_final=True lone_trailing=None
```

Cross-check: the validation stream equals the Session 3 measured token count
(382,916) plus one EOS per nonempty record (2,461), i.e. 385,377. The four
batch keys are `attention_mask`, `input_ids`, `loss_mask`, `targets`; the first
train window decodes to `' = Valkyria Chronicles III = \n Senjō no Valkyria'`.

Forbidden-call scan over `attention.py`, `multihead.py`, and `block.py` found no
`torch.nn.MultiheadAttention`, `torch.nn.TransformerEncoderLayer`,
`torch.nn.functional.scaled_dot_product_attention`, or `torch.nn.LayerNorm` call
in the module sources - only docstring warnings and the learner's own function
name.

Outcome: Week 10 closes with both deliverables implemented and green. The review,
lab note, and handoff are in [session5.md](session5.md) and
[LAB_NOTE.md](LAB_NOTE.md); measured evidence is in [RESULTS.md](RESULTS.md).

## Future Run Record

The instructor maintains detailed records when useful for an actual experiment.
Session 2 onward requires only the short learning note above, not every field
below.

For each actual run, retain:

1. Date, purpose, exact command, working directory, and exit status.
2. Source identity, package versions, hardware, and elapsed time.
3. Seed, fixture sizes, and tensor dtype.
4. Mask convention used, and whether the causal and padding masks were combined.
5. Measured results, test output, failures, interpretation, and next action.

Add commands only once their scripts exist. Label examples or proposals as
unexecuted. Never overwrite an earlier result to hide a failed attempt.
