# Week 3 - CNNs and Mini-ResNets on CIFAR-10

## Purpose

This project resumes the Phase 1 neural-network roadmap after the Basic ML
bridge. The goal is to learn image pipelines, convolutional layers, residual
connections, and controlled model comparison on CIFAR-10.

## Scope

- Use CIFAR-10 through the repository data policy under
  `/home/rui.gao/data/ai_research_roadmap`.
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

Week 3 is complete from a learning-work perspective. CIFAR-10 was prepared
under `/home/rui.gao/data/ai_research_roadmap`, the small CNN and mini-ResNet
paths were implemented, and the selected final run is
`mini_resnet_cosine_confusion`.

Final selected result:

- best observed validation accuracy: 0.8856
- final validation accuracy: 0.8836
- final test accuracy: 0.8772
- final test loss: 0.4102
- recorded confusion matrix: yes

The main remaining limitation is methodological rather than implementation
based: no checkpoint was saved at the best validation epoch, so the final test
accuracy comes from the final epoch.
