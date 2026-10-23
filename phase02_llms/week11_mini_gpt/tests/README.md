# Planned Behavioral Checks

No executable tests exist yet. These are acceptance criteria for the Session 2
implementation, not recorded passes.

## Model

- Forward pass preserves `(B, T, d_model)` through embedding, positions, `N`
  blocks, and final norm, and produces `(B, T, vocab)` logits.
- Parameter count matches the [predicted total](../EXPERIMENT_PLAN.md#working-defaults-to-freeze-before-execution)
  of 1,071,872, or the difference is recorded as a finding.
- Weight tying is real: the LM head's weight tensor is the token embedding's
  weight tensor, not a copy.
- Permuting the input positions changes the logits, proving positional
  information reaches the model. Without positional embeddings this test must
  fail; that contrast is the point.
- A future-token perturbation does not change an earlier position's logits.
- `d_model` divisibility by `n_heads` raises a clear error.
- All parameters receive a non-`None` gradient after one backward pass.

## Loss

- `loss_mask` weighting: appending padding to a batch leaves the loss unchanged.
- Shifting: logits at position `t` score `input_ids[t + 1]`, verified on a tiny
  hand-built batch where the correct target is known.
- An all-`-100` target row does not produce `NaN`.
- Step-0 loss on a fresh model sits near `ln(2048) = 7.6246`, not far below it.

## Training Behavior

- The overfit batch's loss collapses and greedy generation reproduces the
  training window.
- Validation loss is computed with the model in eval mode and dropout disabled.
- Two runs with the same seed agree.

## Execution

Local fixtures are small (`B = 2`, `T = 16-32`) and need no corpus data. The full
training run belongs to Session 4 and its command, log, and checkpoint paths are
recorded in [RUN_LOG.md](../RUN_LOG.md).
