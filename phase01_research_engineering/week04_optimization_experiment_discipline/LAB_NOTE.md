# Week 4 Lab Note - Optimization and Experiment Discipline

## Goal

Learn controlled ablations, experiment tracking, basic testing, and
configuration-driven training.

## Reading Completed

- Adam and AdamW
- SGD with momentum
- learning-rate schedules
- regularization and weight decay
- batch-size effects

## What I Implemented

- Config-driven training entrypoint for Week 4.
- Shared config loading and validation.
- Optimizer selection from config: AdamW or SGD with momentum.
- Scheduler selection from config: cosine annealing or no scheduler.
- Run-directory records with `config.yaml`, `metrics.json`, and `notes.md`.
- Pytest smoke tests for config loading, model forward passes,
  optimizer/scheduler construction, fake-batch training, and run-record writing.

## Experiment Setup

- Dataset: CIFAR-10 with the Week 3 train/validation/test split.
- Model: Week 3 MiniResNet.
- Baseline config: AdamW, learning rate 0.001, weight decay 0.001, cosine
  annealing, batch size 64, seed 42.
- Ablation configs: SGD momentum, lower LR, higher LR, no scheduler, larger
  batch size, and two seed repeats.
- Metric: validation accuracy for ablations; test accuracy only for the
  selected final config.
- Hardware: NVIDIA RTX 5000 Ada Generation using
  `/home/rui.gao/anaconda3/bin/python`.

## Results

The selected final config was `selected_final_seed7`: AdamW, learning rate
0.001, weight decay 0.001, cosine annealing, batch size 64, seed 7.

| Run ID | Best Val Acc | Final Val Acc | Test Acc | Takeaway |
| --- | ---: | ---: | ---: | --- |
| `adamw_cosine_baseline` | 0.8874 | 0.8820 | n/a | Strong baseline. |
| `sgd_momentum` | 0.8300 | 0.8290 | n/a | Too weak at this setting. |
| `lr_low` | 0.8664 | 0.8654 | n/a | Too slow/conservative for 50 epochs. |
| `lr_high` | 0.8868 | 0.8842 | n/a | Competitive but not clearly better. |
| `no_scheduler` | 0.8748 | 0.8642 | n/a | Cosine schedule helped. |
| `batch_size_variant` | 0.8742 | 0.8730 | n/a | Faster but weaker. |
| `seed_repeat_a` | 0.8918 | 0.8888 | n/a | Best validation-only repeat. |
| `seed_repeat_b` | 0.8770 | 0.8738 | n/a | Demonstrates seed variance. |
| `selected_final_seed7` | 0.8930 | 0.8906 | 0.8781 | Final selected run. |

## What Failed

- The first sandboxed suite launch had to be interrupted because it ran inside
  the Codex sandbox rather than the CUDA-visible environment.
- SGD with momentum was not competitive at LR 0.001.
- Low LR underused the 50-epoch budget.
- Larger batch size reduced runtime but also reduced validation accuracy.
- The winning validation repeat did not produce a meaningful test improvement
  over Week 3 because no best-validation checkpointing exists yet.

## What I Understand Better Now

- AdamW is a strong default for this small CIFAR-10 MiniResNet workflow.
- A scheduler is not just a cosmetic setting; cosine annealing improved the
  late-training validation behavior compared with fixed LR.
- Larger batch size can improve wall-clock time while hurting the validation
  result.
- Seed variance is large enough that small one-run differences are not strong
  evidence. Repeated runs changed best validation accuracy from 0.8770 to
  0.8930 for the same recipe.
- Configs matter because they make one-variable changes visible. The lesson is
  not YAML; it is making experimental claims auditable.

## Next Steps

- Carry config files, smoke tests, run records, and validation-first evaluation
  into Week 5 sequence modeling.
- Add checkpointing before future final test evaluations so the selected
  validation epoch can be evaluated, not merely the final epoch.
