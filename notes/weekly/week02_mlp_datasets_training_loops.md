# Week 2 - MLPs, Datasets, and Training Loops

## Goal

Build the first proper supervised-learning experiment in the roadmap: an MLP image classifier with a reproducible dataset path, train/validation/test split, metrics, checkpoint, plots, and lab note.

## Dataset Decision

Use Fashion-MNIST for the first Week 2 experiment.

Rationale:

- It is already listed in `data_registry/datasets.yaml` with `script_supported` status.
- It is slightly less toy-like than MNIST while still small enough for local or HPC smoke runs.
- It can be prepared with the existing command documented in `DATA_PLAN.md`.

Preparation command:

```bash
python scripts/data/prepare_standard_datasets.py --dataset fashion_mnist
```

Use an external data root before running the command:

```bash
export AI_ROADMAP_DATA="$HOME/data/ai_research_roadmap"
```

HPC/scratch data root used for the first Week 2 preparation:

```bash
export AI_ROADMAP_DATA="/scratch/$USER/ai_research_roadmap_data"
```

Dataset preparation status:

- Dry run to local default data root succeeded.
- Dry run to scratch data root succeeded.
- User reported actual Fashion-MNIST preparation succeeded.
- Expected dataset target: `/scratch/rui.gao/ai_research_roadmap_data/raw/torchvision`.

## Learning Question

Can a small fully connected neural network learn a useful Fashion-MNIST classifier using a clean PyTorch training loop, fixed seed, validation split, and saved experiment artifacts?

## Baseline

Start with a simple MLP:

- flatten 28 x 28 images
- one hidden layer
- ReLU activation
- cross-entropy loss
- Adam optimizer

The baseline should be intentionally plain so later Week 2 runs can compare learning rate and hidden-size changes.

## Metric

Track:

- training loss
- validation loss
- validation accuracy
- final test accuracy after model selection

Do not tune directly on the test split.

## Split Plan

Use Fashion-MNIST official train/test split.

Within the official training split:

- train: 50,000 examples
- validation: 10,000 examples

Use the official test split only once the baseline and variants are selected.

## Falsification Condition

The Week 2 setup is not yet a useful supervised-learning baseline if:

- the training loop cannot finish a small smoke run
- validation accuracy does not improve beyond a weak baseline
- metrics, config, and artifacts are not saved reproducibly
- the run cannot be repeated from command-line instructions

## Planned Runs

1. Baseline MLP.
2. Lower or higher learning-rate variant.
3. Smaller or larger hidden-size variant.

Each meaningful run should save:

- config
- metrics
- stdout log or run note
- loss/accuracy plot
- checkpoint

## Lab Note

### Summary

Week 2 completed the first real supervised-learning experiment in the roadmap: a one-hidden-layer MLP classifier trained on Fashion-MNIST with a fixed train/validation/test split plan and three planned configurations.

The training loop is useful enough to carry forward. It can load Fashion-MNIST from the external data root, split the official training set into 50,000 training examples and 10,000 validation examples, train for multiple epochs, report train loss, validation loss, and validation accuracy, and run small smoke checks before full experiments.

The best recorded run was the hidden-size variant:

- hidden size: `256`
- learning rate: `0.001`
- epochs: `5`
- validation accuracy: `0.8822`
- test accuracy: `0.8734`

### Recorded Setup

Fashion-MNIST was selected because it is small enough for quick iteration but less trivial than MNIST. Preparation outside Git under `AI_ROADMAP_DATA` was user-reported successful on 2026-07-02, following the repo data policy; the preparation output was not captured.

Implemented pieces:

- `dataset.py`: Fashion-MNIST loading plus 50,000/10,000 train/validation split
- `model.py`: one-hidden-layer MLP
- `train.py`: command-line training loop with batch-limited smoke controls
- `tests/test_smoke.py`: one-batch dataloader/model/loss smoke test

The following smoke checks were recorded as passing before the full runs:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider phase01_research_engineering/week02_mlp_datasets_training_loops/tests/test_smoke.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 python train.py --epochs 1 --max-train-batches 2 --max-val-batches 2 --batch-size 64
```

Tiny smoke output:

```text
epoch 1: train_loss=2.1816 val_loss=1.7975 val_acc = 0.4844
```

On 2026-07-13, fresh dataset-backed rechecks were blocked because neither the fallback home data root nor the documented scratch root was visible in the current session. The test stopped during dataset construction with downloads disabled, before model construction. A separate model-only forward recheck passed with finite logits of shape `(64, 10)`, so the current failure is an environment/data-availability issue rather than evidence of a model-code regression.

### Run Results

| Run ID | Hidden Size | Learning Rate | Epochs | Validation Accuracy | Test Accuracy | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 128 | 0.001 | 5 | 0.8722 | 0.8644 | solid baseline |
| lr_variant | 128 | 0.003 | 5 | 0.8608 | 0.8548 | worse than baseline |
| hidden_size_variant | 256 | 0.001 | 5 | 0.8822 | 0.8734 | best recorded run |

Baseline epoch stdout:

```text
epoch 1: train_loss=0.4850 val_loss=0.4126 val_acc = 0.8485
epoch 2: train_loss=0.3650 val_loss=0.3696 val_acc = 0.8653
epoch 3: train_loss=0.3291 val_loss=0.3473 val_acc = 0.8720
epoch 4: train_loss=0.3037 val_loss=0.3397 val_acc = 0.8770
epoch 5: train_loss=0.2845 val_loss=0.3405 val_acc = 0.8722
```

Recorded loss plots:

- `phase01_research_engineering/week02_mlp_datasets_training_loops/results/baseline_losses.png`: generated from the recorded baseline loss values without retraining
- `phase01_research_engineering/week02_mlp_datasets_training_loops/results/rl_variant_losses.png`: preserves five train/validation epochs and test loss `0.4073`; raw stdout and the generation command were not retained
- `phase01_research_engineering/week02_mlp_datasets_training_loops/results/hs_variant_losses.png`: preserves five train/validation epochs and test loss `0.3524`; raw stdout and the generation command were not retained

### Interpretation

The baseline learned quickly and reached useful Fashion-MNIST performance for such a simple model. Validation accuracy improved from `0.8485` after epoch 1 to `0.8722` after epoch 5. Validation loss improved until epoch 4, then slightly worsened at epoch 5 while validation accuracy dipped from `0.8770` to `0.8722`. That suggests the baseline was close to its useful limit after a few epochs.

The higher learning-rate variant did not help. Increasing the learning rate from `0.001` to `0.003` with the same hidden size produced lower validation and test accuracy than the baseline. For this model and training setup, the larger step size appears less stable or less well matched to the optimizer dynamics.

The hidden-size variant was the best recorded run. Increasing the hidden layer from `128` to `256` improved validation accuracy from `0.8722` to `0.8822` and test accuracy from `0.8644` to `0.8734`. The gain is modest, but it is consistent enough to show that capacity matters even in this small MLP setup.

### What This Teaches

The main Week 2 lesson is that a clean experiment loop matters as much as the model. The useful parts were not just the MLP, but the fixed split, smoke test, command-line configuration, run log, and clear separation between validation and test metrics.

The modeling lesson is that a plain MLP can learn Fashion-MNIST to a reasonable recorded level, but it ignores image structure. After the compact pure-PyTorch basic-ML bridge, Week 3 should move to CNNs rather than spending more time tuning this MLP against the test set.

### Limitations

- Full epoch stdout was captured only for the baseline.
- Full epoch stdout for the learning-rate and hidden-size variants was not captured.
- The exact final test-evaluation command/output was not captured.
- Three loss plots exist, but the exact generation commands and structured numeric inputs for the two variant plots were not retained.
- No checkpoints are recorded under `results/runs/`.
- The external Fashion-MNIST cache is not visible in the current 2026-07-13 environment, so the dataset-backed smoke test cannot presently be rerun without restoring or re-preparing the external data.
- Test accuracies have already been recorded, so the test split should not be used for more tuning.

### Next Step

Complete the compact pure-PyTorch basic-ML bridge, then move on to Week 3 CNN work. Do not tune the MLP further against the Fashion-MNIST test split.

## What I learned:

1. dataloaders
