# Week 11 Experiment Plan - Mini-GPT Training

## Status

**Frozen 2026-10-05 by Session 1.** Initialized 2026-10-05 as proposed defaults
and confirmed the same day. Every decision row below was reviewed in
[Session 1](session1.md), which records the discussion, the reasons, and the two
precision corrections it found. Nothing in this repository has been trained;
every number here is a frozen setting or an arithmetic prediction that Session 2
checks.

The [experiment plan](../week10_attention_block/EXPERIMENT_PLAN.md) for Week 10
remains the authority on the block's internal tensor and mask contract. This file
only covers the model that wraps it.

## Session 1 Freeze Record - 2026-10-05

| Decision | Frozen choice | Reason |
| --- | --- | --- |
| Positional information | Learned absolute, `nn.Embedding(128, 128)` | `T` is frozen at 128 by the dataset contract, so 128 rows cover the whole reachable input space, and a widened window fails loudly with an index error instead of returning an unvalidated vector |
| Where positions are added | To the token embedding, before the first block | Keeps the block position-agnostic and gives every layer the same ordered stream |
| Weight tying | LM head tied to the token embedding | Saves 262,144 of 1,071,872 parameters (about 24%) and imposes an equality constraint that removes 262,144 degrees of freedom, which regularizes in a 2-3 epoch regime |
| Final normalization | `LayerNorm(d_model)` after the last block, before the LM head | Pre-norm stacks need it for a well-scaled read-out |
| Loss reduction | `reduction="none"`, then a `loss_mask`-weighted mean | The denominator is declared, auditable, and assertable, and the explicit mask survives a change of pad sentinel, which the default `mean` does not |
| Initialization | Normal `(0, 0.02)`; residual-output projections scaled by `1/sqrt(2 * n_layers)` | Keeps residual-stream variance from growing with depth |
| Softmax | Nowhere in the model; logits only | `CrossEntropyLoss` applies log-softmax internally |
| Dropout | `0.1` for the training run, `0.0` for the overfit check | The overfit check has to be a clean memorization test |

Accepted costs and overturn criteria:

- Weight tying accepts that the head cannot learn a scoring geometry independent
  of the input embedding, and that logit scale couples to `||E[v]||`. It is
  overturned only by an underfitting signature attributable to the shared
  geometry, confirmed by an untied ablation at matched seed and budget.
- Learned absolute positions bind the position table to `T`. If a later week
  changes the window length, the positional table has to be resized with it.

Session 1 criteria frozen here:

- **Session 3 overfit criterion.** On one fixed batch with dropout `0.0`, train
  loss below `0.1` (correct-token probability above roughly 90%) within 300
  optimizer steps, plus greedy generation from that batch's prefix reproducing
  the batch's own tokens exactly. The 300-step bound is a proposed bound to be
  validated in Session 3, not a measured constant.
- **Session 4 HPC budget.** One GPU via SLURM (`--partition=gpu`, `--gres=gpu:1`),
  `--time` no longer than `01:00:00`, `B = 32`, `T = 128`, 2,000-3,000 steps,
  checkpoints every 250 steps under `AI_ROADMAP_DATA`, seed `0`.

## Purpose

Measure whether a hand-written decoder-only Transformer can be trained from
scratch on the prepared WikiText-2 splits: does the loss fall below the uniform
baseline, does a single batch overfit to near-zero loss, and do the samples look
like the corpus rather than noise.

This week measures **correctness and training behavior**, not quality at scale.
One small model on one small corpus cannot support a generalization claim, and
none will be made.

## Model Contract

```text
input_ids: (B, T)      int64, vocab 2048, padded with 0
targets:   (B, T)      int64, already shifted by the data layer, padding = -100
attention_mask: (B, T) 1 = real input, 0 = padding      -> goes to attention
loss_mask:      (B, T) 1 = score this position, 0 = skip -> goes to the loss
```

| Component | Input | Output | Rule |
| --- | --- | --- | --- |
| `nn.Embedding(vocab, d_model)` | `(B, T)` | `(B, T, d_model)` | `padding_idx=0` allowed |
| position ids | `(T,)` = `0..T-1` | `(T,)` | built once, broadcast over batch |
| `nn.Embedding(T, d_model)` | `(T,)` | `(T, d_model)` | added to the token embedding |
| `N x TransformerBlock` | `(B, T, d_model)` + additive mask | `(B, T, d_model)` | Week 10, unchanged |
| final `LayerNorm(d_model)` | `(B, T, d_model)` | `(B, T, d_model)` | hand-written, from Week 10 |
| `nn.Linear(d_model, vocab)` | `(B, T, d_model)` | `(B, T, vocab)` | weight-tied to the token embedding if frozen that way |
| loss | logits + `targets` + `loss_mask` | scalar | `CrossEntropyLoss(reduction="none")` then masked mean |

The mask handed to the block is additive: `padding_to_mask(attention_mask) +
build_causal_mask(T)`, per the Week 10 contract. Do **not** pass `loss_mask` to
attention or `attention_mask` to the loss.

## Loss Contract

```text
per_position = CrossEntropyLoss(reduction="none")(logits.reshape(-1, vocab),
                                                  targets.reshape(-1))
per_position = per_position.view_as(targets)
loss = (per_position * loss_mask).sum() / loss_mask.sum()
```

- The loss is a **mean over scored positions**, not over `B * T`.
- Padding positions are `-100` targets and are excluded by `loss_mask`.
- No softmax before the loss.
- A useful reference: an untrained model with uniform logits over 2048 tokens
  gives `ln(2048) = 11 * ln(2) = 7.6246`. A step-0 loss far below that is a sign
  of a leak; a loss stuck at that value is a sign nothing is learning.

## Working Defaults (Frozen 2026-10-05)

| Item | Proposed choice |
| --- | --- |
| Framework | PyTorch, `float32`, hand-written except `nn.Embedding` and `nn.Linear` |
| Forbidden | `torch.nn.TransformerEncoderLayer`, `torch.nn.MultiheadAttention`, `torch.nn.functional.scaled_dot_product_attention`, `torch.nn.LayerNorm` |
| Vocabulary | 2048 (Week 9 tokenizer artifact) |
| `T` | 128, stride 128 (frozen with the dataset) |
| `d_model` | 128 |
| `n_heads` | 4 (`d_head = 32`) |
| `n_layers` | 4 |
| `d_ff` | 512 |
| Dropout | 0.1 for the training run; 0.0 for the overfit check |
| Positional embedding | Learned absolute, `nn.Embedding(128, 128)` |
| Weight tying | Token embedding tied to the LM head |
| Final norm | Present, before the LM head |
| Init | Normal `(0, 0.02)`; residual-output projections scaled by `1/sqrt(2 * n_layers)` |
| Predictions | Softmax applied nowhere in the model; logits only |
| Optimizer | AdamW, `lr = 3e-4`, betas `(0.9, 0.95)`, `eps = 1e-8` |
| Weight decay | 0.1 on weight matrices only; 0.0 on norms and biases |
| Schedule | 100-step linear warmup, then cosine decay to 10% of peak |
| Gradient clip | Global norm 1.0 |
| Batch | `B = 32` windows = 4,096 tokens per step |
| Steps | 2,000-3,000 steps (about 2.2-3.3 epochs over the 3,682,072-token train stream), capped by the SLURM time limit |
| Validation | Every 100 steps on the validation split; full validation at the end |
| Checkpoints | Every 250 steps, under `AI_ROADMAP_DATA`, never in Git |
| Seed | 0 for every recorded run |
| Hardware | Session 3: local CPU or one GPU. Session 4: one GPU via SLURM, `--partition=gpu`, `--gres=gpu:1` |
| Time budget | SLURM `--time` no longer than 01:00:00 unless a longer run is explicitly justified |

Predicted parameter count at the proposed sizes, to be verified in Session 2
rather than assumed:

| Part | Parameters |
| --- | --- |
| Token embedding (tied) | 2048 x 128 = 262,144 |
| Positional embedding | 128 x 128 = 16,384 |
| Attention per block (4 projections, with bias) | 66,048 |
| MLP per block | 131,712 |
| Layer norms per block | 512 |
| Block total | 198,272 |
| 4 blocks | 793,088 |
| Final layer norm | 256 |
| **Predicted total** | **1,071,872** |

If Session 2 measures a different count, the discrepancy is a finding and is
recorded before training, not explained away after.

## Measurement Contract

Reported for every run:

- Exact command, working directory, exit status, date.
- Seed, `T`, `B`, `d_model`, `n_heads`, `n_layers`, `d_ff`, dropout, and dtype.
- Step-0 loss, final train loss, and validation loss / perplexity.
- The full train and validation loss curve, at the recorded step interval.
- Wall time, steps/second, and tokens/second.
- Peak GPU memory if run on GPU.
- Parameter count, measured.
- Generated samples with the exact decoding rule and prompt, saved verbatim.
- Checkpoint path and the settings needed to reload it.

Not reported: any quality, generalization, or comparison-to-other-models claim
that this week's runs cannot support.

## Failure Conditions

Stop before declaring success if any of these hold:

- Loss is `NaN` or `Inf`, or the gradient norm explodes rather than clipping.
- The overfit batch does not reach a memorizing loss.
- Step-0 loss is far below `ln(2048)`, which suggests a target or mask leak.
- The loss is unchanged from `ln(2048)` after warmup, which suggests a broken
  gradient path or a zero learning rate.
- Padding contributes to the loss, or the loss is a plain mean over `B * T`.
- A softmax sits between the logits and the loss.
- A library Transformer block or attention replaces the hand-written one.
- Greedy generation from a memorized window does not reproduce that window.
- Evaluation touches the test split, or validation windows leak into training.

## Completion

Completion requires: fixture checks passing for the model and loss; a recorded
overfit run; one recorded HPC training run with its log and checkpoint; a loss
curve; reader-facing samples with settings; and a lab note. See
[tests/README.md](tests/README.md) for the planned checks and
[RESULTS.md](RESULTS.md) for where measured evidence goes.
