# Session 1 - VAE Concepts, Objective, and Shape Contract

## Purpose

This session starts Week 7 by making the VAE objective clear before writing
model code. The goal is to understand what changes from a plain autoencoder,
why sampling is involved, and what evidence will be needed before claiming that
the model can generate useful samples.

By the end of this session, you should be able to explain the Week 7 experiment
in one sentence:

```text
Train a small encoder-decoder model that maps each image to a latent Gaussian
distribution, samples from that distribution during training, and learns both
to reconstruct images and keep the latent space close to a standard normal
prior.
```

## Content To Understand

### 1. Deterministic Autoencoder Versus VAE

The Week 6 autoencoder learned:

```text
image -> latent vector -> reconstruction
```

A VAE learns:

```text
image -> latent distribution -> sampled latent vector -> reconstruction
```

The distribution is usually a diagonal Gaussian described by two vectors:

- `mu`: the mean of the latent distribution
- `logvar`: the log variance of the latent distribution

The VAE decoder receives a sampled latent vector `z`, not `mu` directly.

### 2. Why A Latent Distribution Matters

A plain autoencoder can reconstruct images well, but its latent vectors are not
forced to follow a simple shape. Sampling random vectors and decoding them may
produce poor images because the decoder was only trained on the encoder's
private latent layout.

A VAE adds pressure for encoded latent distributions to stay near a standard
normal prior:

```text
z ~ Normal(0, I)
```

That pressure is what makes prior sampling meaningful. It also makes the task
harder, because the model must trade off reconstruction detail against latent
space regularity.

### 3. Reparameterization Trick

The encoder predicts `mu` and `logvar`. During training, sample `z` with:

```text
std = exp(0.5 * logvar)
epsilon ~ Normal(0, I)
z = mu + std * epsilon
```

This keeps the random draw in `epsilon` and lets gradients flow through `mu`
and `logvar`.

### 4. Reconstruction Term

The reconstruction term rewards the decoder for producing an image close to
the original input image. For the first Week 7 pass, use a reconstruction term
that is easy to compare with Week 6:

```text
reconstruction_loss = MSE(reconstruction, original_image)
```

Implementation detail for the measured run should make the reduction explicit.
A common choice is summed pixel error per image, averaged over the batch, while
also reporting per-pixel reconstruction MSE for readability.

### 5. KL Term

The KL term measures how far the encoder's latent distribution is from the
standard normal prior.

For diagonal Gaussian latents:

```text
kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

If KL is too strong, reconstructions may become blurry or generic. If KL is too
weak, the model may behave more like a plain autoencoder and prior samples may
be bad.

### 6. Total Objective

The first VAE training objective is:

```text
total_loss = reconstruction_loss + beta * kl_loss
```

Start with `beta = 1.0`. If that makes learning hard to inspect, record any
beta change as an experiment, not as a hidden fix.

Important comparison rule:

```text
Do not compare VAE total loss directly against Week 6 autoencoder MSE.
```

The total VAE loss includes KL and may use a different scale. Compare
reconstruction MSE, image grids, KL behavior, and generated samples instead.

### 7. Dataset And Split

Use Fashion-MNIST for continuity with Week 6:

- train: 55,000 examples from the official training split
- validation: 5,000 examples from the official training split
- test: official 10,000-example test split

Labels are not used in the VAE objective.

Use validation data for design choices. Use test data only after the run setup
is fixed.

### 8. First Shape Contract

```text
raw image batch:        (batch_size, 1, 28, 28)
flattened input:        (batch_size, 784)
encoder hidden:         (batch_size, 128)
mu:                     (batch_size, 32)
logvar:                 (batch_size, 32)
std:                    (batch_size, 32)
epsilon:                (batch_size, 32)
sampled z:              (batch_size, 32)
decoder hidden:         (batch_size, 128)
decoder output flat:    (batch_size, 784)
reconstruction image:   (batch_size, 1, 28, 28)
```

Most first VAE failures are expected to be shape mistakes, loss-scale mistakes,
or incorrect sampling assumptions.

### 9. What The First Run Should Prove

The first measured run should show:

- validation reconstruction MSE beats the average-image baseline
- reconstructions are recognizable and not blank
- KL is finite and logged separately from reconstruction loss
- generated samples from `z ~ Normal(0, I)` are saved separately from
  reconstructions
- claims stay conservative if samples are blurry or class-ambiguous

The first run does not need to beat the deterministic Week 6 autoencoder on
reconstruction MSE.

## Session 1 Task

Before implementing code, answer these checkpoint questions in your own words:

1. What does the VAE encoder output that the plain autoencoder encoder did not?
2. Why does the decoder receive a sampled `z` instead of only `mu`?
3. What problem does the reparameterization trick solve?
4. What does the KL term encourage?
5. Why is VAE total loss not directly comparable to Week 6 autoencoder MSE?
6. What evidence would show that prior sampling is working at least minimally?

## Definition Of Done

Session 1 is done when the dataset, shape contract, `mu`/`logvar` contract,
reparameterization step, reconstruction term, KL term, first baseline, and
falsification conditions can be stated without guessing.

The next session should only begin after these ideas are clear enough to write
smoke checks before training.

## Completion Note

Session 1 is complete as of 2026-09-07.

The guided discussion established the working VAE mental model:

- The Week 6 autoencoder produced one deterministic latent vector per image.
  The Week 7 VAE encoder predicts a latent distribution through `mu` and
  `logvar`.
- During reconstruction, `z` is sampled from the image-conditioned latent
  distribution predicted by the encoder. During generation, there is no input
  image, so `z` is sampled from the prior `Normal(0, I)` and decoded.
- `epsilon` is the independent random noise source in
  `z = mu + std * epsilon`; `mu` and `logvar` are learned by the encoder.
- The reparameterization trick keeps the random draw separate from the learned
  parameters so gradients can still flow through `mu` and `logvar`.
- The reconstruction term compares the decoder output with the original clean
  input image.
- The KL term is a regularizing pressure from the VAE/ELBO objective. It pushes
  the encoder's latent distributions toward `Normal(0, I)` so prior sampling
  at generation time lands in a region the decoder has learned to use.
- The first Week 7 model should ignore Fashion-MNIST labels. Labels can be
  used later for inspection or for a conditional VAE, but not in the first
  unconditional VAE loss.
- VAE total loss is not directly comparable to Week 6 autoencoder MSE because
  it includes both reconstruction and KL terms, and the reconstruction term may
  use a different reduction scale. Session 2 should log reconstruction loss,
  KL loss, total loss, and per-pixel reconstruction MSE separately.
- Minimal evidence for prior sampling is that random
  `z ~ Normal(0, I)` samples decode into recognizable Fashion-MNIST-like
  images rather than blank, pure-noise, or collapsed average images.

The next session can move to implementation only after smoke checks are written
for `mu`, `logvar`, sampled `z`, reconstruction shape, scalar loss terms, and
prior-sampled decoder output.
