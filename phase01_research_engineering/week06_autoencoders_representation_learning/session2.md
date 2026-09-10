# Session 2 - Plain Autoencoder

## Status

Completed on 2026-09-01.

This session built the first working Fashion-MNIST autoencoder path. The goal
was not to make a strong image model; it was to understand the mechanics:
loading image tensors, compressing them through a bottleneck, reconstructing
the image, and comparing the reconstruction against the original.

## What Was Built

- `dataloader.py` loads Fashion-MNIST from
  `/home/rui.gao/data/ai_research_roadmap/raw/torchvision`.
- The loader returns train and test dataloaders.
- Images arrive as tensors shaped `(batch_size, 1, 28, 28)`.
- Pixel values are normalized to `[0, 1]` by `transforms.ToTensor()`.
- Labels are returned by Fashion-MNIST but ignored for autoencoder training.
- `model.py` defines a fully connected autoencoder:

```text
image -> flatten -> 784 -> 128 -> 32 -> 128 -> 784 -> reshape image
```

- The decoder ends with `Sigmoid()` so outputs match the normalized grayscale
  pixel range.
- `train.py` trains with MSE reconstruction loss:

```text
loss = MSE(reconstruction, original_image)
```

## Checks Completed

- Fashion-MNIST was downloaded successfully.
- Dataset sizes checked:

```text
train: 60000
test: 10000
```

- Batch shape checked:

```text
images: (batch_size, 1, 28, 28)
labels: (batch_size,)
pixel range: [0, 1]
```

- Model forward pass checked:

```text
input:  (batch_size, 1, 28, 28)
output: (batch_size, 1, 28, 28)
```

- A small training smoke test ran successfully.
- The original-versus-reconstruction visual comparison works and is the right
  way to inspect whether the model has learned image structure.
- The visual output was a manual Session 2 check, not a polished Session 3
  result artifact. A cleaner grid should be saved under the Week 6 project
  results folder during the measured run.

## Current Simplifications

- This learning pass uses train/test dataloaders only.
- There is no separate validation split yet.
- There is no config file yet.
- There are no formal pytest smoke tests yet.
- The current visual check is manual with Matplotlib.
- The current top-level plot artifacts are exploratory and should not be used
  as the final reconstruction figure.

These are acceptable for the Session 2 learning goal. A cleaner validation
split, config-driven runs, and saved result artifacts can be added later after
the basic autoencoder behavior is understood.

## Run Command

From the repository root:

```bash
python3 -B phase01_research_engineering/week06_autoencoders_representation_learning/train.py
```

If Matplotlib cache warnings appear, run with a writable cache directory:

```bash
MPLCONFIGDIR=/tmp/slurm-rui.gao-169554/matplotlib-codex python3 -B phase01_research_engineering/week06_autoencoders_representation_learning/train.py
```

## Learning Takeaways

- An autoencoder does not use class labels.
- The target is the input image itself.
- The encoder compresses the image into a latent vector.
- The decoder tries to reconstruct the original image from that latent vector.
- Blurry but recognizable output is normal for a small MLP autoencoder.
- Side-by-side original and reconstruction images are more informative than
  looking at reconstruction loss alone.

## Next Session

Session 3 starts on 2026-09-02.

The next goal is to turn the working training path into a measured run:

- train for a small fixed number of epochs
- record train/test reconstruction loss
- save an original-versus-reconstruction grid
- write down what the reconstruction preserves and what it loses
- avoid overclaiming representation quality from one blurry image
