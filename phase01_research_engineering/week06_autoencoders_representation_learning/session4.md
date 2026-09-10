# Session 4 - Denoising Autoencoder

## Status

Completed on 2026-09-03.

Session 4 adds a denoising variant to the Week 6 Fashion-MNIST autoencoder
workflow. The model receives corrupted images as input, but the reconstruction
target remains the clean original image.

## What Changed

- Added `--noise-std` to control Gaussian input noise from the CLI.
- Kept plain-autoencoder behavior as the default with `--noise-std 0.0`.
- Added automatic Session 4 output paths when noise is enabled.
- Added `--metrics-output` so each run can save a structured JSON record.
- Added a noisy-input validation baseline for denoising runs.
- Updated reconstruction grids so denoising runs show:

```text
Original
Noisy
Denoised
```

## Run

```bash
MPLCONFIGDIR=/tmp/matplotlib-codex python3 phase01_research_engineering/week06_autoencoders_representation_learning/train.py --epochs 5 --input 784 --hidden 128 --latent 32 --batch 128 --lr 0.001 --seed 42 --noise-std 0.3
```

Configuration:

- dataset: Fashion-MNIST
- train/validation/test sizes: 55,000 / 5,000 / 10,000
- model: fully connected autoencoder
- encoder/decoder: `784 -> 128 -> 32 -> 128 -> 784`
- latent dimension: 32
- optimizer: Adam
- learning rate: 0.001
- noise: Gaussian, standard deviation 0.3, clamped to `[0, 1]`
- device: CPU

## Results

| Metric | Value |
| --- | ---: |
| Final train MSE | 0.018452 |
| Final validation MSE | 0.018124 |
| Test MSE | 0.018131 |
| Average-image validation MSE | 0.088029 |
| Noisy-input validation MSE | 0.052527 |
| Runtime | 21.559s |

Artifacts:

```text
results/session4_denoising_metrics.json
results/session4_denoising_reconstructions.png
```

## Interpretation

The denoising autoencoder learned to remove substantial visible noise while
preserving coarse Fashion-MNIST shape. Its validation MSE is worse than the
plain autoencoder because it solves a harder input-corruption task, but it is
far better than simply returning the noisy input.

The visual grid matches the metrics: the denoised row is smoother and less
noisy than the corrupted row, while shoes, pants, shirts, and tops remain
recognizable. Fine detail is still blurred, which is expected for a small fully
connected autoencoder with a 32-dimensional bottleneck.

## Carry-Forward

Session 5 should write the Week 6 lab note and explain the latent bottleneck
without overclaiming semantic structure. The strongest supported claim so far
is that the bottleneck preserves enough image information for reconstruction
and denoising on Fashion-MNIST; latent-space organization still needs direct
visualization or probing before stronger representation claims.
