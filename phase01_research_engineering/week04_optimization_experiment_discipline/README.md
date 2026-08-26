# Week 4 - Optimization and Experiment Discipline

## Purpose

Week 4 turns the Week 3 CIFAR-10 training code into a more disciplined
experiment workflow. The learning goal is not a new architecture; it is
controlled comparison, configuration, testing, tracking, and reproducibility.

## Scope

- Study optimizer behavior: SGD, SGD with momentum, Adam, AdamW.
- Compare learning-rate choices and scheduler choices.
- Make training config-driven.
- Add small smoke tests for dataset shape, model forward pass, config loading,
  and a tiny run that finishes quickly.
- Save meaningful ablation runs under `results/runs/`.
- Write a compact experiment table with metrics, runtime, memory notes, and
  reproducibility caveats.

## Starting Point

The Week 3 selected model is `mini_resnet_cosine_confusion`, with:

- best observed validation accuracy: 0.8856
- final validation accuracy: 0.8836
- final test accuracy: 0.8772
- final test loss: 0.4102
- confusion matrix recorded: yes

Week 4 should use that codebase as the source to make experiments cleaner, not
as an excuse to tune repeatedly against the test set.

## Working Rules

- Define each ablation before running it.
- Compare one change at a time where possible.
- Use fixed seeds for comparable runs.
- Save run config, metrics, stdout, runtime notes, and limitations.
- Treat validation accuracy as the comparison metric during ablations.
- Avoid repeated test-set checks; reserve test evaluation for selected final
  configurations.
- Keep checkpoints and large artifacts outside Git.

## Folder Guide

- `configs/`: config files and config-field notes.
- `tests/`: smoke tests for config loading, model shape, training wiring, and run records.
- `results/`: small tracked summaries only.
- `results/runs/`: per-run metadata layout; large outputs stay outside Git if
  needed.
- `RUN_LOG.md`: commands and outcomes after they actually run.
- `RESULTS.md`: measured comparison table after runs actually finish.
- `LAB_NOTE.md`: final Week 4 learning note.

## Current Checkpoint

Week 4 now has a config-driven training copy with shared config validation,
optimizer/scheduler builders, fake-batch smoke tests, and run-record writing.
The measured ablation suite is complete, and `selected_final_seed7` is the
recorded final config. Next active roadmap area: Week 5 RNNs, LSTMs, and
sequence modeling.
