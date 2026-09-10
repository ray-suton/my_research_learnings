# Experiment Plan

## Learning Question

Can a small fully connected variational autoencoder learn a usable latent
distribution for Fashion-MNIST, and what tradeoff appears between
reconstruction quality, KL regularization, and generated samples compared with
the Week 6 plain autoencoder?

## Session 1 Design Target

Start from the Week 6 autoencoder setup, but change the latent representation
from one deterministic vector into a distribution.

The deterministic Week 6 shape was:

```text
image -> encoder -> latent vector -> decoder -> reconstruction
```

The Week 7 VAE target is:

```text
image -> encoder -> mu, logvar -> sample z -> decoder -> reconstruction
```

Use Fashion-MNIST for continuity with Week 6. Keep MNIST as a fallback only if
Fashion-MNIST setup fails.

Use the existing data policy:

```text
AI_ROADMAP_DATA=/home/rui.gao/data/ai_research_roadmap
```

Do not store downloaded datasets, caches, large checkpoints, or bulky generated
artifacts in Git.

## Dataset And Split

- dataset: Fashion-MNIST
- fallback dataset: MNIST if Fashion-MNIST setup fails
- input image shape: `(1, 28, 28)`
- flattened image dimension: `784`
- pixel range: `[0, 1]`
- train split: 55,000 examples from the official training split
- validation split: 5,000 examples from the official training split
- test split: official 10,000-example test split, reserved for final fixed-run
  reporting

Fashion-MNIST labels are not used by the VAE objective. They may be kept only
for optional visual inspection or later latent probes.

## Shape Contract

The first VAE should use this tensor contract:

```text
raw image batch:        (batch_size, 1, 28, 28)
flattened input:        (batch_size, 784)
encoder hidden:         (batch_size, hidden_dim)
mu:                     (batch_size, latent_dim)
logvar:                 (batch_size, latent_dim)
std:                    (batch_size, latent_dim)
epsilon:                (batch_size, latent_dim)
sampled z:              (batch_size, latent_dim)
decoder hidden:         (batch_size, hidden_dim)
decoder output flat:    (batch_size, 784)
reconstruction image:   (batch_size, 1, 28, 28)
reconstruction loss:    scalar
KL loss:                scalar
total loss:             scalar
```

First model target:

- input dimension: 784
- hidden dimension: 128
- latent dimension: 32
- encoder heads: separate linear layers for `mu` and `logvar`
- decoder output activation: sigmoid

## Reparameterization

Sampling directly from `Normal(mu, std)` would block ordinary backpropagation
through the random sampling operation. The VAE should use the
reparameterization trick:

```text
std = exp(0.5 * logvar)
epsilon = random normal noise with the same shape as std
z = mu + std * epsilon
```

This keeps randomness in `epsilon` while leaving `mu` and `logvar` inside a
differentiable computation path.

## Loss Contract

Track the VAE objective as separate terms:

```text
total_loss = reconstruction_loss + beta * kl_loss
```

For the first implementation, use `beta = 1.0` unless training behavior makes
the balance impossible to inspect. If beta is changed, record it as an
experiment choice rather than silently tuning it.

Recommended first reconstruction term:

- mean squared reconstruction error summed per image and averaged per batch
- report per-pixel validation reconstruction MSE separately for comparison
  with Week 6

Session 2 implementation: `vae_loss` receives mean-reduced per-pixel MSE and
the pixel count (784 for the current grayscale images), then computes
`total = 784 * mean_mse + beta * kl`. It returns `(mean_mse, total, kl)`.
Multiplying by 784 already yields the per-image sum averaged across the batch;
do not divide by batch size again. This convention is covered by the saved
gradient and loss checks in `tests/test_session2_smoke.py`.

KL term for a diagonal Gaussian posterior against a standard normal prior:

```text
kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
```

Average this KL term per batch item so it is readable in run logs.

Do not compare VAE total loss directly with Week 6 autoencoder MSE. The fairer
comparison is reconstruction MSE, visual reconstructions, and the presence or
absence of usable prior samples.

## Baselines And Comparisons

Use these comparison anchors:

- Week 6 plain autoencoder, latent dimension 32:
  - validation MSE: 0.014738
  - test MSE: 0.014750
- Week 6 average-image validation baseline:
  - validation MSE: 0.088029

The first VAE does not need to beat the deterministic autoencoder on
reconstruction MSE. It should beat the average-image baseline and produce
recognizable reconstructions before stronger claims are made.

For generation, sample latent vectors from the standard normal prior:

```text
z ~ Normal(0, I)
```

Decode those samples and save them separately from reconstructions.

## Metrics

Track:

- train total loss
- train reconstruction term
- train KL term
- validation total loss
- validation reconstruction term
- validation KL term
- validation per-pixel reconstruction MSE
- final test per-pixel reconstruction MSE after the run is fixed
- generated sample artifact path
- reconstruction artifact path
- runtime
- device
- seed
- beta
- latent dimension

## Planned Config Fields

- `run_name`
- `dataset`
- `data_root`
- `validation_size`
- `model_name`
- `input_dim`
- `hidden_dim`
- `latent_dim`
- `epochs`
- `batch_size`
- `learning_rate`
- `seed`
- `beta`
- `output`
- `metrics_output`
- `device`
- `num_workers`
- `reconstruction_count`
- `sample_count`

## Smoke Tests

Before running a measured training job, add smoke checks for:

- dataset batch shape is `(batch_size, 1, 28, 28)`
- flattened batch shape is `(batch_size, 784)`
- `mu` shape is `(batch_size, latent_dim)`
- `logvar` shape is `(batch_size, latent_dim)`
- sampled `z` shape is `(batch_size, latent_dim)`
- decoder output flat shape is `(batch_size, 784)`
- reconstruction image shape is `(batch_size, 1, 28, 28)`
- reconstruction values are compatible with normalized pixel range
- reconstruction loss, KL loss, and total loss are scalar tensors
- one tiny train step computes gradients and updates parameters
- prior sampling produces image-shaped decoder outputs

## Falsification Conditions

- If `mu`, `logvar`, or sampled `z` shapes are unclear, do not train a measured
  run yet.
- If reconstruction output shape or range does not match the input image
  contract, do not interpret loss.
- If KL becomes non-finite, treat the run as broken.
- If KL is near zero while samples and reconstructions are poor, investigate
  posterior collapse or an ignored latent variable.
- If reconstruction MSE beats the baseline but generated samples are blank or
  average-looking, do not claim useful generation.
- If the average-image baseline is comparable to the VAE reconstruction, record
  that the model has not shown meaningful per-image reconstruction.
- If the test split is used during architecture, beta, or latent-size choices,
  do not treat the final test result as a clean evaluation.

## Expected Output

- documented VAE shape and loss contract
- small fully connected VAE
- reparameterization helper
- VAE loss helper with separate reconstruction and KL terms
- smoke checks for shapes, loss, sampling, and one tiny train step
- one measured VAE run
- reconstruction grid
- generated sample grid
- latent-size ablation table
- Week 7 lab note comparing plain autoencoders and VAEs

## Current Status

All five sessions have completed technical deliverables on 2026-09-08. The
eight Session 2 checks and five experiment checks pass. See `session3.md`,
`session4.md`, `session5.md`, `RESULTS.md`, and `LAB_NOTE.md` for actual outcomes.

The experiment contract was fixed in
`.omx/plans/week07-finish-sessions-20260908.md` at the repository root before
running. Latent sizes 8/16/32/64 used the same five-epoch settings. Minimum
final validation sampled-reconstruction MSE selected size 8; the selection was
saved before final test evaluation. A fresh matched latent-8 AE was used for
the final comparison. The original Week 6 results remain historical anchors.

The basic intuition review completed with guided discussion on 2026-09-10.
`session5.md` records the demonstrated ideas and corrections. Independent ELBO
derivation and a comprehensive theory assessment were not performed.
