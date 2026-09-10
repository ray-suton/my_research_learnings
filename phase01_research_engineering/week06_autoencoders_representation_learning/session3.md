# Session 3 - Reconstruction Run and Visuals

## Status

Completed on 2026-09-02.

This session turned the first working plain autoencoder into a measured run.
The goal was to make the run reproducible, record reconstruction losses, compare
against a simple non-learned baseline, and save a clean reconstruction grid.

## What Was Built

- `dataloader.py` now uses a seed-controlled train/validation split:
  55,000 train examples, 5,000 validation examples, and 10,000 official test
  examples.
- `train.py` now has CLI defaults for epochs, input size, hidden size, latent
  size, batch size, learning rate, seed, output path, and reconstruction sample
  count.
- `train.py` prints train/validation reconstruction MSE each epoch.
- `train.py` prints final test reconstruction MSE.
- `train.py` computes an average-image validation baseline.
- `train.py` saves the reconstruction grid under the Week 6 project instead of
  writing it into the repository root.

## Run

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex python3 phase01_research_engineering/week06_autoencoders_representation_learning/train.py --epochs 5 --input 784 --hidden 128 --latent 32 --batch 128 --lr 0.001 --seed 42
```

## Results

```text
epoch 0 train MSE: 0.039581
epoch 0 validation MSE: 0.022648
epoch 1 train MSE: 0.020107
epoch 1 validation MSE: 0.018560
epoch 2 train MSE: 0.017536
epoch 2 validation MSE: 0.016824
epoch 3 train MSE: 0.015981
epoch 3 validation MSE: 0.015495
epoch 4 train MSE: 0.014930
epoch 4 validation MSE: 0.014738
test MSE: 0.014750
average-image validation MSE: 0.088029
runtime: 20.764s on CPU
```

The model beats the average-image baseline by a wide margin, so the
autoencoder is learning image-specific reconstruction rather than producing one
generic output.

## Visual Check

Saved grid:

```text
results/session3_reconstructions.png
```

The reconstructions preserve broad clothing silhouettes, but edges and texture
are blurry. That is a reasonable first result for a small fully connected
autoencoder.

## Takeaways

- A plain autoencoder can reconstruct held-out Fashion-MNIST images with much
  lower MSE than an average-image baseline.
- The validation loss and visual grid agree: the model preserves coarse shapes
  but loses detail.
- The latent vector should be described conservatively. This run shows
  reconstruction under a bottleneck, not semantic understanding.

## Next Session

Session 4 should add a denoising autoencoder variant:

- corrupt the input image with Gaussian noise or random masking
- reconstruct the clean original image
- compare denoising output against the plain Session 3 baseline
- record whether denoising improves visual robustness or mainly changes the
  training objective
