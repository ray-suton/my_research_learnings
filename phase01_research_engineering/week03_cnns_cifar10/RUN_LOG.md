# Run Log

Use this file to record commands, settings, and outcomes. Do not record results
until the run has actually happened.

## Data Preparation

| Date | Command | Data Root | Status | Notes |
| --- | --- | --- | --- | --- |
| TBD | `python scripts/data/prepare_standard_datasets.py --dataset cifar10 --dry-run` | TBD | not run | Confirm target path first. |
| TBD | `python scripts/data/prepare_standard_datasets.py --dataset cifar10` | TBD | not run | Keep downloaded data outside Git. |

## Smoke Checks

| Date | Purpose | Command | Status | Notes |
| --- | --- | --- | --- | --- |
| TBD | dataloader/model/loss smoke test | TBD | not run | Add after implementation. |
| TBD | tiny training smoke run | TBD | not run | Use a few batches before full training. |

## Training Runs

| Run ID | Date | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- |
| `cnn_baseline` | TBD | TBD | not run | TBD | Small CNN baseline. |
| `cnn_aug_regularized` | TBD | TBD | not run | TBD | Add augmentation/regularization/schedule. |
| `mini_resnet` | TBD | TBD | not run | TBD | Residual comparison. |

## Issues

| Date | Issue | Evidence | Resolution |
| --- | --- | --- | --- |
| TBD | TBD | TBD | TBD |
