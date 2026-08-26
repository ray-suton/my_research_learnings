# Experiment Plan

## Learning Question

How much do optimizer choice, learning-rate scale, scheduler choice, batch size,
and seed control affect the Week 3 CIFAR-10 training workflow when the model and
data pipeline are held mostly fixed?

## Baseline

Use the Week 3 final MiniResNet path as the conceptual baseline:

- model: `mini_resnet`
- data: CIFAR-10 train/validation/test split from Week 3
- augmentation: random crop and horizontal flip for training
- optimizer baseline: AdamW
- scheduler baseline: cosine annealing
- selected Week 3 final test accuracy: 0.8772

For Week 4 ablations, compare primarily on validation accuracy and validation
loss. Test accuracy should be used only for the selected final configuration.

## Metrics

- train loss
- train accuracy
- validation loss
- validation accuracy
- best validation accuracy
- final validation accuracy
- runtime
- GPU memory note when available
- final learning rate when a scheduler is used
- repeated-run variance when compute allows

## Config Fields

The config-driven training path should eventually control:

- `model_name`
- `epochs`
- `batch_size`
- `seed`
- `optimizer`
- `learning_rate`
- `weight_decay`
- `scheduler`
- `scheduler_min_lr`
- `data_root`
- `num_workers`
- `run_name`

## Planned Ablations

| Run ID | Change | Purpose |
| --- | --- | --- |
| `adamw_cosine_baseline` | Re-run config-driven version of the Week 3 final setup | Prove the config path can reproduce the existing training recipe. |
| `sgd_momentum` | Swap AdamW for SGD with momentum | Compare adaptive optimizer behavior against momentum SGD. |
| `lr_low` | Lower learning rate | Check whether smaller steps stabilize or slow learning. |
| `lr_high` | Higher learning rate | Check whether larger steps improve speed or destabilize training. |
| `no_scheduler` | Disable cosine annealing | Test whether the schedule mattered beyond optimizer and weight decay. |
| `batch_size_variant` | Change batch size | Observe runtime and generalization effects. |
| `seed_repeat_a` / `seed_repeat_b` | Repeat selected config with different seeds | Estimate run-to-run variance if compute allows. |

## Falsification Conditions

- If the config-driven run cannot match the old code path on a smoke test, the
  refactor is not yet trustworthy.
- If a one-change ablation changes multiple settings accidentally, the run is
  not interpretable.
- If validation accuracy shifts less than normal seed variance, do not claim the
  ablation caused a real improvement.
- If the best result comes from repeated test-set checking, do not treat it as a
  valid comparison.

## Expected Output

- reusable config-driven training entrypoint
- pytest smoke tests
- per-run records under `results/runs/`
- summary table in `RESULTS.md`
- Week 4 lab note with reproducibility limitations
