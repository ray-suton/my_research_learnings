# Week 6 Lab Note - Autoencoders and Representation Learning

## Goal

Week 6 moved from supervised prediction into reconstruction-based
representation learning. The central question was whether a small autoencoder
could compress Fashion-MNIST images through a bottleneck and reconstruct
recognizable image structure better than simple non-learned baselines.

This was not a state-of-the-art image-modeling exercise. The goal was to learn
the mechanics of encoders, decoders, reconstruction loss, validation discipline,
denoising, and conservative interpretation of latent vectors.

## What Was Built

The Week 6 implementation uses Fashion-MNIST from the roadmap data root and a
small fully connected autoencoder:

```text
image -> flatten -> 784 -> 128 -> 32 -> 128 -> 784 -> reconstructed image
```

The 32-dimensional vector is the latent bottleneck. It is similar to PCA in the
basic sense that a high-dimensional image is compressed into a lower-dimensional
representation, but it is not PCA. PCA is a linear projection chosen to preserve
variance. This autoencoder learns nonlinear encoder and decoder weights chosen
to minimize reconstruction loss.

Training uses mean squared error between the reconstructed image and the clean
original image. Fashion-MNIST labels are not used by the objective.

## Results

| Run ID | Objective | Final Val MSE | Test MSE | Baseline Comparison |
| --- | --- | ---: | ---: | --- |
| `session3_plain_autoencoder` | Clean image to clean reconstruction | 0.014738 | 0.014750 | Beats average-image validation MSE 0.088029 |
| `session4_denoising_autoencoder` | Noisy image to clean reconstruction | 0.018124 | 0.018131 | Beats noisy-input validation MSE 0.052527 and average-image validation MSE 0.088029 |

The plain autoencoder result supports a conservative claim: with a
32-dimensional bottleneck, this model preserves enough information to
reconstruct coarse Fashion-MNIST image structure better than an average-image
baseline.

The denoising result supports a second conservative claim: when input images
are corrupted with Gaussian noise, the model learns a mapping that removes much
of the visible noise while preserving coarse clothing shapes better than simply
returning the corrupted input.

## Visual Evidence

The main visual artifacts are:

```text
results/session3_reconstructions.png
results/session4_denoising_reconstructions.png
results/session3_vs_session4_metrics.png
results/latent_interpolation_two_real_images.png
```

The reconstruction grids match the scalar metrics. Clothing silhouettes remain
recognizable, but fine details, textures, and sharp edges are blurred. This is
expected for a shallow fully connected model with a small bottleneck.

The latent-interpolation probe encodes two real test images, linearly
interpolates between their latent vectors, and decodes the intermediate points.
The result is useful evidence that the latent space can support smooth
interpolation between two encoded examples. It is not evidence of high-quality
generation from random noise.

## Supervised Learning vs Representation Learning

A supervised Fashion-MNIST classifier learns features that help predict a class
label such as shirt, trouser, or ankle boot. It is rewarded only when the final
label prediction is better. It may ignore pixel-level details if those details
do not help classification.

This autoencoder learns features that help reconstruct the input image. It is
rewarded for pixel-level similarity between the output and the original image.
That makes the learned representation useful for reconstruction, but it does
not automatically mean the representation is semantic, class-aligned, or useful
for downstream classification.

Denoising changes the input-target pair:

```text
noisy image -> clean original image
```

The model still reconstructs the clean image, but now it cannot simply copy the
input pixel by pixel. It has to learn enough regularity about Fashion-MNIST
images to remove noise while keeping the main object shape.

## Failure Modes

Important failure modes for this kind of autoencoder:

- Falling validation MSE can still hide visually poor reconstructions.
- A model can collapse toward average-looking outputs and still reduce some
  pixel loss.
- A bottleneck that is too small may lose class-relevant details.
- A bottleneck that is too large may copy too much and learn a weak compression
  pressure.
- Smooth latent interpolation between two examples does not prove semantic
  organization of the whole latent space.
- Test loss should not be used repeatedly for architecture or latent-size
  tuning.
- One seed and one small architecture are not enough to make broad claims about
  representation quality.

## Carry-Forward To Week 7 VAEs

The main Week 7 question is how a variational autoencoder changes this setup.
A plain autoencoder maps each image to one learned latent vector and decodes it
back to an image. A VAE will instead learn a latent distribution and add KL
regularization plus the reparameterization trick.

Carry these Week 6 lessons forward:

- Keep reconstruction quality and visual inspection paired with scalar loss.
- Compare against simple baselines before claiming progress.
- Be precise about generation: decoding interpolated latent vectors is not the
  same as sampling new images from a trained latent prior.
- Watch the tradeoff between reconstruction loss and latent-space structure.
- Do not claim semantic meaning unless there is direct evidence from latent
  visualization, clustering, probing, or downstream performance.

## Week 6 Completion

Week 6 is complete from a learning-work perspective. The remaining possible
extensions, such as formal pytest smoke tests, YAML config files, a broader
latent-space visualization, and latent-size ablations, are useful but not
required before moving to Week 7.
