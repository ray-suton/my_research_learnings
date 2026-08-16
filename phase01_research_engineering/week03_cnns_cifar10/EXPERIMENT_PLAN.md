# Experiment Plan

## Learning Question

How much does a small convolutional architecture improve over a plain MLP-style
image learner on CIFAR-10, and does a mini-ResNet-style residual model improve
optimization or validation accuracy over a simple CNN?

## Dataset

- Dataset: CIFAR-10
- Storage: external data root from `AI_ROADMAP_DATA`
- Registry entry: `data_registry/datasets.yaml`
- Preparation command:

```bash
python scripts/data/prepare_standard_datasets.py --dataset cifar10
```

## Planned Runs

| Run ID | Model | Purpose |
| --- | --- | --- |
| `cnn_baseline` | small CNN | establish image baseline |
| `cnn_aug_regularized` | small CNN with augmentation/weight decay/schedule | test training discipline improvements |
| `mini_resnet` | small residual CNN | compare residual optimization against baseline CNN |

## Metrics

- train loss
- validation loss
- validation accuracy
- final test accuracy once per selected model
- confusion matrix for the selected final model

## Falsification Conditions

- If the CNN cannot beat a weak baseline after smoke debugging, inspect data
  normalization, labels, train/eval mode, and learning-rate scale.
- If mini-ResNet does not improve over the simple CNN, record that result
  instead of tuning against the test set.

## Outputs To Keep Small

- one results table
- one loss/accuracy plot per meaningful comparison
- one confusion matrix for the selected final model
- no large checkpoints in Git
