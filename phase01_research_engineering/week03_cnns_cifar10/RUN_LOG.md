# Run Log

Use this file to record commands, settings, and outcomes. Do not record results
until the run has actually happened.

## Data Preparation

| Date | Command | Data Root | Status | Notes |
| --- | --- | --- | --- | --- |
| 2026-08-16 | `export AI_ROADMAP_DATA="$HOME/data/ai_research_roadmap"` | `/home/rui.gao/data/ai_research_roadmap` | set | Canonical future data root. |
| 2026-08-16 | `python scripts/data/prepare_standard_datasets.py --dataset cifar10 --dry-run` | `/home/rui.gao/data/ai_research_roadmap` | passed | Confirmed target path before download. |
| 2026-08-16 | `python scripts/data/prepare_standard_datasets.py --dataset cifar10` | `/home/rui.gao/data/ai_research_roadmap` | passed | Downloaded CIFAR-10 outside Git; train=50000, test=10000. |

## Smoke Checks

| Date | Purpose | Command | Status | Notes |
| --- | --- | --- | --- | --- |
| 2026-08-16 | dataloader smoke test | import `dataset.py` and inspect one CIFAR-10 batch | passed | train=45000, val=5000, test=10000; batch shape `(64, 3, 32, 32)`. |
| 2026-08-17 | model smoke test | import `model.py`, run fake batch through `build_model()` | passed | Output shape `(4, 10)`; parameter count 545098. |
| 2026-08-17 | tiny training smoke run | monkeypatch `get_dataloaders()` to one fake batch and call `train.train(1)` | passed | Verified train/val/test code path without full CIFAR run. |
| 2026-08-17 | one-epoch CIFAR smoke run | `/home/rui.gao/anaconda3/bin/python -c "import sys, time; sys.path.insert(0, 'phase01_research_engineering/week03_cnns_cifar10'); import train; t=time.time(); train.train(1); print('elapsed_seconds', round(time.time()-t, 2))"` | passed | Host GPU run; val accuracy 0.6166, test accuracy 0.6262, elapsed 9.87s. |
| 2026-08-19 | augmented one-epoch CIFAR smoke run | `/home/rui.gao/anaconda3/bin/python -c "import sys; sys.path.insert(0, 'phase01_research_engineering/week03_cnns_cifar10'); import train; train.train(1)"` | passed | Train augmentation plus AdamW path ran; val accuracy 0.5632, test accuracy 0.5663. |
| 2026-08-20 | mini-ResNet one-epoch CIFAR smoke run | `python3 phase01_research_engineering/week03_cnns_cifar10/train.py --model mini_resnet --epochs 1` | passed | CLI model-selection path ran end to end; val accuracy 0.5298, test accuracy 0.5289. |

## Training Runs

| Run ID | Date | Command | Status | Key Result | Notes |
| --- | --- | --- | --- | --- | --- |
| `cnn_baseline` | 2026-08-17 | `/home/rui.gao/anaconda3/bin/python -c "import sys, time; sys.path.insert(0, 'phase01_research_engineering/week03_cnns_cifar10'); import train; t=time.time(); train.train(20); print('elapsed_seconds', round(time.time()-t, 2))"` | passed | Final test accuracy 0.7030; best observed validation accuracy 0.7230 at epoch 5. | Host GPU run outside Codex sandbox; training accuracy rose to 0.9765 while validation accuracy plateaued then declined, indicating overfitting. |
| `cnn_aug_regularized` | 2026-08-19 | `/home/rui.gao/anaconda3/bin/python -c "import sys; sys.path.insert(0, 'phase01_research_engineering/week03_cnns_cifar10'); import train; train.train(50)"` | passed | Final test accuracy 0.7862; best observed validation accuracy 0.7872 at epoch 48. | Random crop, horizontal flip, and AdamW weight decay improved test accuracy by 0.0832 over the unregularized baseline. |
| `mini_resnet` | 2026-08-20 | `/home/rui.gao/anaconda3/bin/python -B phase01_research_engineering/week03_cnns_cifar10/train.py --model mini_resnet --epochs 50` | passed | Final test accuracy 0.8649; best observed validation accuracy 0.8776 at epoch 40. | Host GPU run outside Codex sandbox; final train accuracy 0.9278 and final validation accuracy 0.8732. |
| `mini_resnet_cosine_confusion` | 2026-08-20 | `/home/rui.gao/anaconda3/bin/python -B phase01_research_engineering/week03_cnns_cifar10/train.py --model mini_resnet --epochs 50` | passed | Final test accuracy 0.8772; best observed validation accuracy 0.8856 at epoch 48. | Intentional final Week 3 run with cosine annealing and confusion matrix output; final train accuracy 0.9422, final validation accuracy 0.8836, final test loss 0.4102, final learning rate 1.0977e-05. |

## Issues

| Date | Issue | Evidence | Resolution |
| --- | --- | --- | --- |
| 2026-08-17 | Sandboxed Python could not see CUDA | `nvidia-smi` saw RTX 5000 Ada, but sandboxed PyTorch reported `cuda_available False` and `/dev/nvidia*` missing. | Ran GPU checks and training outside the sandbox with `/home/rui.gao/anaconda3/bin/python`; CUDA was available there. |
| 2026-08-17 | Baseline overfits without regularization | Train accuracy reached 0.9765 by epoch 19 while validation accuracy stayed near 0.71 and validation loss rose above 2.16. | Next run should add augmentation, weight decay, and/or learning-rate scheduling. |
| 2026-08-19 | Regularized CNN is validation-limited rather than strongly overfit | At epoch 49, train accuracy was 0.8016, validation accuracy was 0.7782, and final test accuracy was 0.7862. | Use this as the comparison baseline for Day 4 mini-ResNet work. |
| 2026-08-20 | Mini-ResNet improved over the regularized CNN baseline | Final test accuracy rose from 0.7862 to 0.8649, with best observed validation accuracy 0.8776. | This was superseded by the final scheduled run with cosine annealing and confusion-matrix output. |
| 2026-08-20 | Final scheduled MiniResNet run completed Week 3 measurement loop | The intentional final run reached 0.8772 test accuracy, 0.8856 best observed validation accuracy, and printed a 10x10 confusion matrix. | Use `mini_resnet_cosine_confusion` as the selected final Week 3 result. |
