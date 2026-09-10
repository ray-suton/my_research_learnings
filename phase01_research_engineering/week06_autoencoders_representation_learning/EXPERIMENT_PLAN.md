# Experiment Plan

## Learning Question

Can a small fully connected autoencoder learn a useful compressed
representation of Fashion-MNIST images, and do the reconstructions show that
the model has learned image structure rather than only reducing a scalar loss?

## Session 1 Design Decision

Use Fashion-MNIST for the first Week 6 autoencoder experiment unless setup
friction makes MNIST necessary as a fallback.

The first task is reconstruction, not classification:

```text
input image -> encoder -> latent vector -> decoder -> reconstructed image
```

The reconstruction target is the original clean input image. Labels may be kept
for optional visual inspection, but they are not part of the training objective.

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
- current learning pass: official train split plus official test split
- later measured pass: carve validation examples from the official training
  split before tuning architecture or hyperparameters
- test data: official test split

For Session 2, train/test is acceptable because the goal is mechanics, not a
clean model-selection claim. For later reporting, do not tune architecture,
latent size, epoch count, or learning rate on the test split.

## Shape Contract

The first fully connected autoencoder should use this tensor contract:

```text
raw image batch:        (batch_size, 1, 28, 28)
flattened input:        (batch_size, 784)
encoder hidden:         (batch_size, 128)
latent vector:          (batch_size, 32)
decoder hidden:         (batch_size, 128)
decoder output flat:    (batch_size, 784)
reconstruction image:   (batch_size, 1, 28, 28)
loss:                   scalar
```

The implementation should make this contract easy to inspect in smoke tests.
Most early failures are expected to be shape, range, or split mistakes rather
than deep modeling problems.

## Baseline

Compare the first autoencoder against a simple non-learned reconstruction
baseline.

Start with an average-image baseline:

- compute the mean image from the training split
- use that same mean image as the reconstruction for every validation example
- report validation reconstruction MSE
- compare its visual output against the learned autoencoder

The autoencoder should beat this baseline on validation MSE and should produce
reconstructions that preserve more per-image shape than the average image.

## Model Target

Implement one small plain autoencoder first:

- model type: fully connected plain autoencoder
- encoder: `784 -> 128 -> 32`
- latent dimension: `32`
- decoder: `32 -> 128 -> 784`
- hidden activation: ReLU
- output activation: sigmoid
- reconstruction loss: mean squared error

This model is intentionally small. The goal is to understand the mechanics of
compression, reconstruction, validation loss, and visual inspection before
trying convolutional or denoising variants.

## Metrics

Track:

- train reconstruction MSE
- validation reconstruction MSE for measured passes
- final test reconstruction MSE after a fixed measured run
- average-image baseline validation MSE
- noisy-input baseline validation MSE for denoising runs
- runtime
- device
- number of train and evaluation examples

Use validation reconstruction MSE as the main scalar metric before any final
test claim. Always pair scalar loss with image inspection before making quality
claims. For denoising, compare against the corrupted input itself so the model
has to beat the trivial "do nothing" noisy reconstruction.

## Visual Outputs

The first required visual output is a grid comparing:

```text
original image | reconstructed image
```

Use fixed test examples for the current learning pass so visual comparisons are
stable across runs. If a validation split is added later, use fixed validation
examples during tuning.
The report should state whether reconstructions preserve coarse shape, whether
details are blurry, and whether any examples collapse toward an average-looking
image.

Optional later visual output:

- PCA view of higher-dimensional latent vectors
- a separate two-dimensional latent run for direct plotting

Do not make latent-space claims until the visualization exists.

## Planned Config Fields

The config-driven training path should eventually control:

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
- `noise_std`
- `output`
- `metrics_output`
- `device`
- `num_workers`
- `reconstruction_count`

## Smoke Tests

Before running a measured training job, add smoke tests for:

- dataset batch shape is `(batch_size, 1, 28, 28)`
- flattened batch shape is `(batch_size, 784)`
- encoder output shape is `(batch_size, 32)`
- decoder output flat shape is `(batch_size, 784)`
- reconstruction image shape is `(batch_size, 1, 28, 28)`
- reconstruction values are compatible with normalized pixel range
- one tiny train step computes scalar MSE and updates parameters
- for the later measured pass, validation split is carved from train data, not
  from test data
- for the later measured pass, average-image baseline can produce a validation
  MSE

## Session 2 Checkpoint

Session 2 was completed on 2026-09-01 for the learning version.

Completed:

- Fashion-MNIST downloaded and loaded from the roadmap data root.
- Train/test dataloaders produce image batches shaped `(batch_size, 1, 28, 28)`.
- Pixel values are in `[0, 1]`.
- Labels are ignored during autoencoder training.
- Fully connected autoencoder uses `784 -> 128 -> 32 -> 128 -> 784`.
- Decoder output uses `Sigmoid()`.
- Reconstruction is reshaped back to `(batch_size, 1, 28, 28)`.
- Training uses MSE between reconstruction and original input image.
- Manual original-versus-reconstruction comparison works.

Deferred:

- config file
- formal pytest smoke tests
- validation split
- average-image baseline
- saved run table
- polished reconstruction grid

## Session 3 Checkpoint

Session 3 was completed on 2026-09-02 for the first measured plain-autoencoder
run.

Completed:

- CLI-controlled measured run with explicit epoch, input, hidden, latent,
  batch-size, learning-rate, seed, output, and sample-count arguments.
- Deterministic train/validation split carved from the official training split.
- Five-epoch plain autoencoder run on Fashion-MNIST.
- Train, validation, and test reconstruction MSE recorded in `RESULTS.md` and
  `results/session3_metrics.json`.
- Average-image validation baseline recorded.
- Clean original-versus-reconstruction grid saved under `results/`.

Deferred:

- YAML config file
- formal pytest smoke tests
- denoising autoencoder
- latent-space visualization
- latent-size ablation
- final Week 6 lab note

## Session 4 Checkpoint

Session 4 was completed on 2026-09-03 for the first denoising-autoencoder run.

Completed:

- CLI-controlled Gaussian input corruption with `--noise-std`.
- Clean-image reconstruction targets retained while the model receives noisy
  inputs.
- Default Session 4 output paths selected automatically when noise is enabled.
- Metrics JSON output added with train, validation, test, average-image
  baseline, noisy-input baseline, runtime, and run config.
- Five-epoch denoising run on Fashion-MNIST with `--noise-std 0.3`.
- Three-row original/noisy/denoised reconstruction grid saved under
  `results/`.
- Visual inspection confirmed that the denoised outputs remove most visible
  pixel noise while remaining blurrier than the clean originals.

Deferred:

- YAML config file
- formal pytest smoke tests
- latent-space visualization
- latent-size ablation
- final Week 6 lab note

## Falsification Conditions

- If the model output shape does not match the input image shape, do not train
  a measured run yet.
- If the output range is incompatible with normalized image targets, fix that
  before interpreting loss.
- If train MSE decreases but validation MSE worsens, treat the run as
  overfitting rather than progress.
- If validation MSE improves but reconstructions are blank, average-looking, or
  visually collapsed, do not claim useful representation learning.
- If the average-image baseline is comparable to the autoencoder, record that
  the learned model has not yet shown meaningful per-image reconstruction.
- For denoising, if the model cannot beat the noisy-input baseline, do not
  claim that it learned useful denoising behavior.
- If the test split was used during architecture or hyperparameter choices, do
  not treat the final test result as a clean evaluation.

## Expected Output

- dataset loader using the repository data-root policy
- small fully connected autoencoder
- config-driven training entrypoint
- pytest smoke tests for data, shapes, loss, and tiny training
- average-image baseline
- one measured plain-autoencoder run
- reconstruction grid
- results table in `RESULTS.md`
- Week 6 lab note explaining bottlenecks, reconstruction, and representation
  learning limitations
