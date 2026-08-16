# Week 3 - CNNs and Mini-ResNets on CIFAR-10

## Purpose

This project resumes the Phase 1 neural-network roadmap after the Basic ML
bridge. The goal is to learn image pipelines, convolutional layers, residual
connections, and controlled model comparison on CIFAR-10.

## Scope

- Use CIFAR-10 through the repository data policy under `AI_ROADMAP_DATA`.
- Train a small CNN baseline.
- Add augmentation, weight decay, learning-rate scheduling, and a confusion
  matrix.
- Train a mini-ResNet-style model.
- Compare CNN and mini-ResNet results with validation/test metrics,
  limitations, and failure cases.

## Working Rules

- Keep datasets, caches, checkpoints, and large outputs outside Git.
- Record commands and outcomes in `RUN_LOG.md`.
- Record measured results only after they actually run.
- Save plots and small summaries under `results/` only when they are small and
  worth keeping in the repo.
- Do not tune repeatedly against the test set.

## Current Checkpoint

Week 3 is scaffolded but not implemented. Start by confirming CIFAR-10
preparation through `scripts/data/prepare_standard_datasets.py`.
