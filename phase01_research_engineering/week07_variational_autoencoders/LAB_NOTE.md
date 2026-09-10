# Week 7 Lab Note - Variational Autoencoders

## Goal

Build a small VAE, understand the reconstruction/KL tradeoff, and distinguish
reconstructing an input from generating an image using the prior.

## What Was Implemented

- A Fashion-MNIST VAE with separate mu/logvar heads and reparameterization.
- Explicit mean-MSE scaling and separately reported reconstruction, KL, and
  total objective values.
- Sample-weighted training/validation metrics, isolated validation randomness,
  deterministic mean-code diagnostics, and a reproducible command-line runner.
- Eight model/training smoke checks and five experiment checks.
- A five-epoch baseline, a four-size latent comparison, and a matched AE/VAE
  comparison with final held-out evaluation.

## Experiment Setup

Seed 42; 55,000 training images and 5,000 validation images; hidden size 128;
batch size 128; Adam learning rate 0.001; beta 1; five epochs; one CPU thread.
Validation uses one posterior draw per image with isolated seed 123.

For normalized grayscale images, the objective is:

```text
reconstruction term = 784 * mean pixel MSE
KL = sum over latent dimensions, then mean over images
total = reconstruction term + beta * KL
```

Loss values are aggregated using actual batch image counts. The final small
batch therefore does not receive the weight of a full batch.

## Results

The initial latent-32 VAE's validation sampled MSE improved from 0.035588 to
0.025294. KL stayed finite and ended at 7.970769 nats per image. Reconstructions
preserved broad garment shapes, while generated samples were varied but blurry.

Latent sizes 8/16/32/64 produced final validation sampled MSE values
0.024347/0.025318/0.025294/0.026851. Latent size 8 was locked as the final choice
before test evaluation.

| Final latent-8 comparison | Test per-pixel MSE |
| --- | ---: |
| Plain AE | 0.020632 |
| VAE, posterior draw | 0.024081 |
| VAE, decode(mu) | 0.020514 |

The decoded mean and a sampled latent answer different reconstruction
questions. Comparing the VAE's total objective directly with AE MSE would hide
that distinction and mix incompatible scales.

## What Failed Or Stayed Weak

- Small details, logos, textures, and edges are often lost in reconstructions.
- Some generated images are ambiguous or malformed.
- More latent dimensions did not help automatically within this training budget.
- Only 9 of the baseline's 32 dimensions exceeded the chosen 0.01-nat average
  KL threshold. This is a usage diagnostic, not a semantic interpretation.
- No broad posterior-collapse diagnosis follows from KL alone. The nonzero
  aggregate KL and input-dependent reconstructions are evidence against a fully
  input-ignoring model here, but are not a complete causal analysis.

## Intuition To Carry Forward

- An AE gives each image a code. A VAE gives it a distribution of possible codes.
- Reconstruction rewards preserving image-specific information. KL puts a
  price on moving the posterior away from the prior used for generation.
- Reparameterization puts randomness in epsilon while keeping a differentiable
  path through mu and logvar.
- During reconstruction, the input image determines mu and logvar. During
  generation, the decoder receives a prior sample without an input image.
- Beta and the reduction convention jointly set the relative loss weights.
- `eval()` controls module behavior; it does not switch off an explicitly
  programmed random sampling step.

The core intuition was reviewed in guided conversation on 2026-09-10. See
`session5.md` for the demonstrated ideas and corrections; formal derivations
were not assessed.

## Limits And Reproduction

One seed, five epochs, one dataset, and a small MLP do not establish statistical
superiority, semantic latent structure, or production generation quality.
Historical Week 6 metrics are separately labeled because their latent size and
batch-averaging convention differ from the fresh matched baseline.

`README.md` contains the full reproduction command. Metrics, configs, selected
checkpoint hash, and figures are retained under `results/completed_20260908/`.
Small stdout logs are present but ignored by the repository's Git rules.
Checkpoints use the temporary `/tmp/week07_vae_20260908/` storage exception and
can expire; the recorded commands regenerate them in fresh output directories.

## Next Step

The basic intuition review is complete. Move next to Week 8 GANs and compare
reconstruction, KL regularization, and adversarial learning as different ways
to shape a generator.
