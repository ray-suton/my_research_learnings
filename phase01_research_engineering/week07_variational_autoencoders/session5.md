# Session 5 - Plain Autoencoder Versus VAE

## Status

Implementation, experiments, and the final lab note completed on 2026-09-08.
The basic intuition review completed on 2026-09-10 through guided discussion.
This records a practical understanding checkpoint, not mastery of every VAE
derivation or theoretical detail.

## What Was Built

- `compare_autoencoder.py` reuses the Week 6 AutoEncoder architecture and trains
  a fresh matched baseline.
- Both models use dimensions 784/128/8, seed 42, the same 55,000/5,000 split,
  five epochs, batch 128, Adam learning rate 0.001, and CPU.
- The VAE checkpoint was chosen using validation only before this comparison.
- Final evaluation uses all 10,000 official test images. Reconstruction and
  prior-probe figures are saved separately.

## Results

| Model / reconstruction policy | Validation MSE | Test MSE |
| --- | ---: | ---: |
| Matched plain AE | 0.020793 | 0.020632 |
| VAE, one posterior sample per image | 0.024347 | 0.024081 |
| VAE, decode the latent mean mu | 0.020744 | 0.020514 |

The VAE test KL is 7.912745 nats per image. Its sampled reconstruction term is
18.879740 and total objective is 26.792484. These are different quantities
from the per-pixel MSE values in the table.

Decoding `mu` uses the mean latent vector. Because the decoder is nonlinear,
this is not generally the average of decoded posterior samples.

The AE has 203800 parameters and the VAE has 204832. Objectives and the exact
encoder heads differ. This is a controlled learning comparison, not a claim
that the architectures or compute are identical.

## Visual Check

- Reconstructions preserve broad item shapes but lose logos, fine texture, and
  sharp edges. The displayed AE and VAE mean-code outputs are visually close.
- VAE samples drawn from `Normal(0, I)` include varied clothing silhouettes,
  with some malformed or ambiguous examples.
- The AE's arbitrary standard-normal latent probes look much less coherent in
  this figure. The AE never learned a standard-normal prior, so that row is an
  illustrative probe, not a fair measure of an AE sampling distribution.

[Reconstruction comparison](results/completed_20260908/session5_comparison/session5_reconstruction_comparison.png)

[Random-latent comparison](results/completed_20260908/session5_comparison/session5_random_latent_comparison.png)

## Limits

- The small MSE difference between AE and VAE mean-code reconstruction is not
  evidence of general superiority: only one seed and a short run were used.
- No quantitative generation metric or semantic latent probe was run.
- The historical Week 6 AE used latent size 32 and reported test MSE 0.014750.
  Its evaluation also averaged batch means. Those historical numbers are not
  the matched latent-8 baseline in this session.
- A fixed sampling seed makes the VAE estimate reproducible; it does not remove
  Monte Carlo uncertainty.

## Evidence And Next Step

The complete config, epoch history, test values, source/checkpoint hashes, and
command are in
[session5_comparison_metrics.json](results/completed_20260908/session5_comparison/session5_comparison_metrics.json).
The recorded comparison runtime is 15.461 seconds, including evaluation and
figure generation; it is not directly comparable to the VAE training-only
runtime fields.

## Understanding Review - 2026-09-10

The learner correctly connected removing KL with a mismatch between
encoder-produced codes and standard-normal codes used for generation. The
learner also identified the encoder’s training role as producing a distribution
conditioned on each input image.

Guided corrections covered:

- Excessively strong KL can remove image-specific information. With squared
  error, an ambiguous black/white pixel favors gray; the decoder can produce
  an average-looking reconstruction rather than necessarily random noise.
- For fixed weights and the same input, the encoder’s distribution is unchanged;
  different random draws cause reconstruction variation. Reusing the same code
  in the current deterministic decoder gives the same output.
- Reparameterization does not rely on negligible noise. The learner correctly
  predicted that shifting mu by 0.1 shifts z by 0.1, and decreasing sigma moves
  the sample toward mu, then connected this path with reconstruction feedback
  reaching the encoder.
- Generation draws from the standard-normal prior without requiring an input
  image or an encoder pass.

Some answers needed hints and correction. Independent ELBO derivation, detailed
log-variance derivatives, and stochastic-gradient theory were not assessed.
The core intuition checkpoint is complete; those are later depth topics.

Week 8 GANs is next. No Week 8 implementation was started here.
