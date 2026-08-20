# Week 3 Lab Note - CNNs and Mini-ResNets on CIFAR-10

## Goal

The goal for Week 3 was to move from basic dense neural-network training into
image models: CIFAR-10 dataloading, convolutional classifiers, data
augmentation, weight decay, learning-rate scheduling, residual connections, and
confusion-matrix analysis.

## What Was Built

- CIFAR-10 data loading with train/validation/test splits.
- A small CNN baseline.
- Training improvements with random crop, horizontal flip, AdamW weight decay,
  and cosine annealing.
- A mini-ResNet-style model with a stem convolution, residual blocks, global
  average pooling, and a classifier head.
- A confusion-matrix helper used by the final evaluation path.

## Runs

| Run ID | Model | Best Validation Accuracy | Final Validation Accuracy | Final Test Accuracy |
| --- | --- | ---: | ---: | ---: |
| `cnn_baseline` | small CNN | 0.7230 | 0.7078 | 0.7030 |
| `cnn_aug_regularized` | small CNN with augmentation and weight decay | 0.7872 | 0.7782 | 0.7862 |
| `mini_resnet` | mini-ResNet | 0.8776 | 0.8732 | 0.8649 |
| `mini_resnet_cosine_confusion` | mini-ResNet with cosine schedule and confusion matrix | 0.8856 | 0.8836 | 0.8772 |

The selected final Week 3 result is `mini_resnet_cosine_confusion`.

## Interpretation

The plain CNN established that convolutional features work much better for
CIFAR-10 than a dense-only mental model, but it overfit quickly. Training
accuracy kept rising while validation accuracy plateaued and validation loss
increased.

Augmentation and weight decay improved the small CNN from 0.7030 to 0.7862 test
accuracy. This reduced the overfitting pattern and made the model more
validation-limited.

The mini-ResNet was the largest improvement. Residual connections let the model
go deeper without making optimization fragile, and global average pooling kept
the classifier head small. The final scheduled run reached 0.8772 test
accuracy.

## Confusion-Matrix Takeaways

The hardest class was cat, with 732 correct examples out of 1000. The largest
failure mode was cat confused with dog, followed by dog confused with cat. This
is a sensible image-recognition failure because those classes share shape,
texture, and pose variation more than classes such as automobile, ship, or
truck.

The strongest classes were automobile, frog, truck, and ship. Those categories
have more distinctive global shape or background cues in CIFAR-10.

## Lessons

- A convolution is a spatial feature extractor, not just a dense layer with
  fewer parameters.
- Padding matters because it controls whether spatial size is preserved across
  convolutional layers.
- Batch normalization normalizes channel activations, so `BatchNorm2d` uses the
  output-channel count.
- Global average pooling averages each channel over spatial height and width,
  producing one feature per channel.
- Cosine annealing gradually lowers the learning rate so later epochs make
  smaller updates instead of continuing with the original step size.
- A confusion matrix shows which classes are failing, not just whether the
  total accuracy is high.

## Limitations

- No best-validation checkpoint was saved, so final test accuracy is from the
  final epoch rather than the best validation epoch.
- The test set was used only for recorded final evaluations, but there is no
  automated guard preventing repeated test-set checks.
- The final model was not tuned extensively; this was a learning milestone, not
  a leaderboard run.

## Week 3 Status

Week 3 is complete from a learning-work perspective.
