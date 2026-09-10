# Week 6 - Autoencoders and Representation Learning

## Purpose

Week 6 moves from supervised prediction and sequence modeling into learned
representations. The goal is to understand how an encoder compresses an input,
how a decoder reconstructs it, and what a bottleneck can reveal about structure
in image data.

This is not meant to build a state-of-the-art image model. The target is a
small, inspectable autoencoder workflow on MNIST or Fashion-MNIST with clear
reconstruction metrics, visual samples, and a short note explaining what
representation learning changes compared with supervised classification.

## Scope

- Study plain autoencoders, bottlenecks, reconstruction loss, latent vectors,
  and denoising autoencoders.
- Reuse the existing `AI_ROADMAP_DATA` policy and choose MNIST or Fashion-MNIST
  from the dataset registry.
- Implement a small fully connected or convolutional autoencoder.
- Train the model to reconstruct input images.
- Visualize original images, reconstructions, and optionally a 2D latent view.
- Add a simple denoising variant that corrupts inputs but reconstructs clean
  targets.
- Record commands, metrics, visual outputs, limitations, and a lab note.

## Workload

### Session 1 - Concepts, Dataset, and Shape Contract

- Read about encoder/decoder structure, bottleneck size, reconstruction loss,
  and why reconstruction is not the same objective as classification.
- Pick MNIST or Fashion-MNIST and confirm the data root rule:
  `AI_ROADMAP_DATA=/home/rui.gao/data/ai_research_roadmap`.
- Define the tensor contract for flattened or image-shaped batches.
- Choose the first metric, likely validation reconstruction loss.

Deliverable: documented dataset choice, input shape, latent shape, loss, and
baseline expectation.

### Session 2 - Plain Autoencoder

- Build the Fashion-MNIST dataset loader.
- Implement the first plain fully connected autoencoder.
- Train with MSE reconstruction loss against the original input image.
- Check batch shape, model forward shape, loss computation, and one small train
  path.
- Compare an original image against its reconstruction.

Deliverable: runnable first training path with manual smoke checks.

### Session 3 - Reconstruction Run and Visuals

- Run one small measured training job.
- Save run config, train/test reconstruction loss, runtime notes, and
  reconstruction images.
- Compare original versus reconstructed samples without overclaiming.
- If useful, add a simple 2D latent visualization using PCA or a two-dimensional
  bottleneck run.

Deliverable: measured plain-autoencoder run plus reconstruction figure.

### Session 4 - Denoising Variant

- Add input corruption, such as Gaussian noise or random pixel masking.
- Train the model on noisy inputs with clean-image targets.
- Compare denoising reconstructions against the plain autoencoder.
- Record whether denoising helps visually or only changes the learning behavior.

Deliverable: denoising run and comparison notes.

### Session 5 - Lab Note and Carry-Forward

- Write the Week 6 lab note.
- Explain what the latent bottleneck is doing.
- Compare supervised learning and representation learning in plain language.
- Record failure modes such as blurry outputs, too-small latent dimensions, or
  memorization.
- Note what should carry into Week 7 VAEs.

Deliverable: final Week 6 results table, visuals, and lab note.

## Working Rules

- Keep the first model small and easy to inspect.
- Prefer one clear dataset over broad dataset comparisons.
- Do not use test data while tuning architecture or latent dimension.
- Save large datasets, checkpoints, and bulky generated outputs outside Git.
- Track small summary images and metrics only when they are useful for the
  learning record.
- Treat blurry but recognizable reconstructions as a useful early result, not a
  failure.

## Folder Guide

- `EXPERIMENT_PLAN.md`: learning question, dataset decision, metric, baseline,
  falsification conditions, and expected outputs.
- `dataloader.py`: dataset loading and transforms.
- `model.py`: encoder, decoder, and autoencoder module.
- `train.py`: CLI-controlled training loop, validation/test evaluation, optional Gaussian-noise corruption, average-image and noisy-input baselines, metrics JSON output, and reconstruction-grid writer.
- `session1.md`: Session 1 notes.
- `session2.md`: Session 2 completion notes.
- `session3.md`: Session 3 measured-run note.
- `session4.md`: Session 4 denoising-run note.
- `LAB_NOTE.md`: final Week 6 learning note and Week 7 carry-forward.
- `configs/`: optional later YAML configs for plain and denoising runs.
- `visualize.py`: optional later reconstruction grid and latent-space helpers.
- `tests/`: optional later smoke tests for data, shapes, loss, and tiny
  training.
- `results/`: small tracked summaries and reconstruction figures.
- `RUN_LOG.md`: commands and outcomes.
- `RESULTS.md`: measured metrics and reconstruction comparisons.

## Current Checkpoint

Week 6 is complete from a learning-work perspective. The current
implementation uses Fashion-MNIST, a fully connected autoencoder, latent
dimension 32, MSE reconstruction loss against clean targets, deterministic
train/validation/test dataloaders, an average-image baseline, a noisy-input
baseline, CLI-controlled Gaussian input noise, metrics JSON output, and
project-local reconstruction grids.

The selected Session 3 and Session 4 output files are:

```text
results/session3_metrics.json
results/session3_reconstructions.png
results/session4_denoising_metrics.json
results/session4_denoising_reconstructions.png
results/session3_vs_session4_metrics.png
results/latent_interpolation_two_real_images.png
LAB_NOTE.md
```

The Session 4 denoising run trained for 5 epochs on CPU with
`--noise-std 0.3`. Final validation reconstruction MSE was 0.018124, test
reconstruction MSE was 0.018131, the noisy-input validation baseline was
0.052527, and the average-image validation baseline was 0.088029.

The final Week 6 lab note records the strongest supported claim: the
32-dimensional bottleneck preserves enough information to reconstruct coarse
Fashion-MNIST image structure better than an average-image baseline, but this
does not prove semantic latent structure. The immediate next roadmap step is
Week 7: variational autoencoders.
