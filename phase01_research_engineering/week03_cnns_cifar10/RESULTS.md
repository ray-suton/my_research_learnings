# Results

Week 3 has measured small-CNN and mini-ResNet runs on CIFAR-10. The selected
final result is the MiniResNet run with augmentation, AdamW weight decay, cosine
annealing, and a recorded confusion matrix.

## Summary Table

| Run ID | Model | Validation Accuracy | Test Accuracy | Notes |
| --- | --- | ---: | ---: | --- |
| `cnn_baseline` | two-conv small CNN | 0.7230 best observed at epoch 5; 0.7078 final | 0.7030 final | 20-epoch host-GPU run. No checkpoint was saved, so the reported test result is from the final epoch, not the best validation epoch. |
| `cnn_aug_regularized` | same two-conv CNN with random crop, horizontal flip, and AdamW weight decay | 0.7872 best observed at epoch 48; 0.7782 final | 0.7862 final | 50-epoch host-GPU run. Uses the learner's single-setting workflow, so test was evaluated once at the end of the run. |
| `mini_resnet` | small residual CNN with stem, residual blocks, and global average pooling | 0.8776 best observed at epoch 40; 0.8732 final | 0.8649 final | 50-epoch host-GPU run. Improved final test accuracy by 0.0787 over `cnn_aug_regularized`, then was superseded by the final scheduled run. |
| `mini_resnet_cosine_confusion` | same mini-ResNet with cosine annealing and final confusion matrix | 0.8856 best observed at epoch 48; 0.8836 final | 0.8772 final | Selected final Week 3 run. Final learning rate was 1.0977e-05, so training ended near the chosen minimum rather than at zero. |

## Interpretation

The unregularized baseline learns CIFAR-10 quickly, reaching about 0.62
validation accuracy after one epoch and about 0.72 best observed validation
accuracy by epoch 5. After that, training accuracy continues rising while
validation loss increases, which is the expected overfitting pattern for a
small unregularized CNN.

The augmented and regularized CNN is a clear improvement: final test accuracy
increased from 0.7030 to 0.7862. The final train/validation accuracies were
0.8016 and 0.7782, so the model is now much less overfit and is better
described as validation-limited.

The mini-ResNet is a second clear improvement. The first full mini-ResNet run
reached 0.8776 best observed validation accuracy and 0.8649 final test
accuracy. Compared with the augmented and regularized small CNN, final test
accuracy improved by 0.0787.

The final scheduled mini-ResNet run is the selected Week 3 result. It reached
0.8856 best observed validation accuracy and 0.8772 final test accuracy, giving
another 0.0123 absolute test-accuracy gain over the earlier mini-ResNet run.
The final train/validation accuracies were 0.9422 and 0.8836, so the residual
model is strong but still has a visible train-validation gap.

## Final Confusion Matrix

Class order is the standard CIFAR-10 order: airplane, automobile, bird, cat,
deer, dog, frog, horse, ship, truck. Rows are true labels and columns are
predicted labels.

```text
[
 [896, 8, 28, 8, 7, 1, 7, 10, 24, 11],
 [10, 937, 1, 2, 1, 0, 1, 1, 6, 41],
 [23, 1, 830, 38, 37, 24, 32, 8, 3, 4],
 [16, 3, 34, 732, 43, 112, 28, 21, 3, 8],
 [4, 0, 24, 28, 881, 16, 27, 18, 1, 1],
 [7, 2, 19, 90, 28, 825, 9, 20, 0, 0],
 [3, 1, 27, 17, 8, 8, 930, 4, 1, 1],
 [13, 1, 13, 16, 24, 30, 3, 896, 0, 4],
 [41, 7, 6, 5, 2, 2, 4, 2, 916, 15],
 [20, 29, 4, 5, 0, 2, 1, 3, 7, 929]
]
```

The weakest class is cat, with 732 correct out of 1000. The largest error pair
is cat predicted as dog (112 examples), followed by dog predicted as cat (90
examples). The strongest classes are automobile, frog, truck, and ship.

## Limitations

- No checkpoint was saved at the best validation epoch, so reported test
  accuracy is from the final epoch of each run.
- The final test set was evaluated once for each recorded run; repeated
  test-set tuning was intentionally avoided.
