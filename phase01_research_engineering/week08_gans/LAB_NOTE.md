# Week 8 Lab Note - Generative Adversarial Networks

Date: 2026-09-17. Core Week 8 learning work is closed with guided understanding.
This note is grounded in saved runs and the learner's discussion; it does not
claim independent mastery of points that the instructor explained.

## Goal And Implementation

Learn adversarial image generation, isolated alternating updates, failure
analysis, controlled comparisons, and honest experiment reporting.

The learner implemented an MLP generator/discriminator and then a CNN pair.
The CNN generator uses a learned 7×7 feature grid, nearest-neighbor upsampling,
convolutions, and Tanh. Its discriminator uses three stride-2 convolutions,
LeakyReLU, and a two-layer classifier head with a sigmoid output. This is a
small CNN GAN rather than an exact canonical DCGAN reproduction.

The assistant completed explicitly requested runner/configuration integration,
verification, and execution. G uses non-saturating sigmoid/BCE feedback;
D sums real and fake mean BCE terms. The D update detaches generated images;
the G update differentiates through D while keeping D's parameters fixed.

## Recorded MNIST Experiments

All runs use MNIST 55,000/5,000/10,000 splits, seed 42, batch 128, noise width
64, Adam LR 0.0002/betas (0.5, 0.999), and CPU/two threads. Test images did not
enter measured optimization, scoring, or checkpoint selection. Final epochs
were declared in advance. Monitoring/final noise seeds are 123/456.

| Pair | Epochs | Paired updates | Training seconds | Final D loss | Final G loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP | 5 | 2,150 | 25.523 | 0.394094 | 3.368496 |
| MLP | 20 | 8,600 | 100.959 | 0.525343 | 2.534042 |
| CNN | 5 | 2,150 | 262.733 | 0.201683 | 9.202752 |
| CNN | 20 | 8,600 | 1,098.076 | 0.729749 | 1.967853 |

Training seconds sum completed epochs; the CNN twenty-epoch wall time also
included interruption and verified checkpoint recovery after epoch 12.
These losses are diagnostics, not comparable image-quality scores.

[Final MLP/CNN panels](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_epoch20.png) ·
[Loss curves](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_losses_20epochs.png) ·
[Epoch-5/10/20 progression](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/epoch_05_10_20_comparison.png)

The MLP improves visually with more training but retains speckle and many
1-like outputs. The CNN panels show cleaner, more recognizable and visibly
more varied shapes, especially after epoch 5, while some images remain
ambiguous. These observations support this pairwise experiment, not general
superiority or quantitative class coverage.

## VAE And GAN Conceptual Comparison

| Aspect | VAE | This GAN |
| --- | --- | --- |
| Training input to decoder/generator | Latent sampled from an encoder distribution conditioned on a real image | Noise sampled from a prior |
| Feedback | Reconstruction of the same real image plus KL regularization | A learned discriminator's real/fake feedback |
| Latent organization | KL encourages compatibility with the chosen prior | Learned implicitly through adversarial training |
| Generation | Sample the prior and run the decoder | Sample noise and run the generator |
| Extra network needed at generation | Encoder is unnecessary | Discriminator is unnecessary |
| Interpretation caution | Reconstruction quality does not establish prior-sample quality | Low losses or realistic samples do not establish coverage or convergence |

Week 7 used Fashion-MNIST and Week 8's core GAN runs used MNIST. Their
reconstruction losses, ELBO terms, and adversarial losses cannot establish
numerical superiority of one model family. Even the requested Fashion-MNIST
extension will need a separately matched evaluation before such a claim.

## Failure Analysis And Learning

The discussion established that a changing discriminator can raise G's loss
while images improve. Mode collapse concerns the output distribution's
coverage, independently of the appearance of one sample. The distinction
between D classification accuracy and useful generator gradients needed
explanation, as did why different D losses are not a common comparison scale.

The learner proposed holding the initial generator fixed and inspecting
generated images. Guidance added matched training budgets, complete noise
panels, and independent evaluation of fidelity and variety. Memorization
diagnostics were explained on request; they were not independently proposed
or executed. [session5.md](session5.md) records the assessment boundaries.

## Verification And Remaining Limits

The saved CNN integration passed 25 CPU checks with zero skips, focused lint,
and runner type checks. The longer run restored model, optimizer, and RNG
states; a synthetic resumed-versus-uninterrupted check matched exactly.
Both twenty-epoch runs' forty checkpoint hashes and epoch-5/10/20/final panel
replays were verified. Source/configuration snapshots and full records remain
in each run directory. These are retained results, not new tests in this
documentation closure.

Both G and D change between MLP and CNN, with larger CNN parameter counts
and compute. Equal seeds do not imply identical training-noise trajectories
across architectures. Independent seeds, quantitative generated-class
coverage, convergence, and absence of memorization have not been established.

## Next Step

The learner explicitly confirmed the GAN topic finished on 2026-09-17 after
noticing predominantly sweatshirt-like Fashion-MNIST outputs. This is
consistent with partial mode collapse / mode dropping in the displayed panels;
generated-class coverage was not quantified. The dataset itself is roughly
class-balanced, so the observation cannot simply be attributed to absent
footwear or bag training examples.

Conditional generation and batch-aware diversity methods were discussed,
but no remedy, additional penalty, or extra training was implemented. The
remaining limitations are documented results, not blockers to moving on.

The core Week 8 sessions are complete from a learning-work perspective.
The learner requested a separate CNN-only, twenty-epoch Fashion-MNIST
extension after this documentation. It subsequently completed successfully;
the [extension record](fashion_mnist_extension.md) keeps its settings, results,
verification, and limits separate from the core MNIST experiments above.
The next roadmap topic is Week 9 tokenization and text data.
