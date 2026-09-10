# Session 1 - Autoencoder Concepts and Shape Contract

## Purpose

This session is for understanding the basic idea before writing training code.
Do not start by implementing the model. First learn what an autoencoder is,
what it is optimized to do, what the tensor shapes should be, and what the
first experiment is supposed to prove.

By the end of this session, you should be able to explain the Week 6 experiment
in one sentence:

> Train a small neural network to compress an image into a latent vector and
> reconstruct the original image from that compressed representation.

## Content To Understand

### 1. Reconstruction Instead Of Classification

In supervised classification, the model learns:

```text
image -> class label
```

In an autoencoder, the model learns:

```text
image -> reconstructed image
```

The target is the input image itself. Labels are not required for the first
autoencoder experiment.

Understand this distinction:

- a classifier is judged by whether it predicts the right class
- an autoencoder is judged by whether its output resembles the input
- reconstruction quality is not the same thing as semantic understanding

### 2. Encoder

The encoder is the part of the model that compresses the input.

For a 28 by 28 grayscale image, the raw image has:

```text
1 * 28 * 28 = 784 pixel values
```

A simple fully connected encoder might map:

```text
784 -> 256 -> 64 -> latent_dim
```

The encoder's job is to turn the input image into a smaller vector called a
latent vector.

### 3. Bottleneck

The bottleneck is the narrowest part of the model. It is usually smaller than
the input.

Example:

```text
input: 784 values
latent vector: 32 values
```

The bottleneck matters because the model cannot simply copy every pixel through
unchanged. It has to decide what information is most useful for reconstructing
the image.

Understand the tradeoff:

- if the bottleneck is too small, reconstructions lose important details
- if the bottleneck is too large, the model may learn a near-copy shortcut
- a useful bottleneck should preserve enough structure to reconstruct the main
  shape without pretending the model has learned high-level meaning

### 4. Decoder

The decoder expands the latent vector back into the original image shape.

Example:

```text
latent_dim -> 64 -> 256 -> 784
```

For a simple first model, the decoder output can be reshaped back into:

```text
(batch_size, 1, 28, 28)
```

If the input pixels are scaled to the range `[0, 1]`, the final decoder output
should also stay in that range. A sigmoid output layer is a simple first choice.

### 5. Latent Representation

The latent vector is the compressed representation learned by the encoder.

It is tempting to say the latent vector learns "meaning", but that is too
strong for this first experiment. A safer statement is:

```text
The latent vector stores whatever information helps the decoder reconstruct
the image under the chosen bottleneck and loss.
```

Later, you can inspect latent representations by:

- visualizing a two-dimensional latent space
- running PCA on higher-dimensional latent vectors
- checking whether similar images land near each other
- comparing reconstructions from different latent sizes

### 6. Reconstruction Loss

The loss compares the reconstructed image with the original image.

For Session 1, understand these two common choices:

- mean squared error: penalizes pixel-by-pixel squared difference
- binary cross entropy: often used when normalized pixels are treated like
  probabilities

Use mean squared error for the first plan unless you have a clear reason not
to. It is simple to understand:

```text
lower validation reconstruction MSE = closer average pixel reconstruction
```

Do not rely only on the scalar loss. Always inspect reconstruction images too.
A low loss can still hide blurry or uninteresting outputs.

### 7. Dataset Choice

Week 6 should use one small 28 by 28 grayscale dataset:

- MNIST: easier, digit shapes are simple
- Fashion-MNIST: slightly harder, clothing shapes are more varied

Recommended first choice: Fashion-MNIST.

Reason: this roadmap already used Fashion-MNIST in Week 2, so the dataset is
familiar, but the task is now different: reconstruct the image instead of
classifying it.

MNIST is an acceptable fallback if Fashion-MNIST causes dataset friction.

### 8. Data Splits

Use labels only for optional inspection, not for the reconstruction objective.

Keep the split discipline:

- train split: used to update model weights
- validation split: used to compare reconstruction loss while designing
- test split: reserved for final reporting only

Do not tune architecture, latent size, or training length on the test split.

### 9. Tensor Shape Contract

For the first fully connected autoencoder, use this shape contract:

```text
raw image batch:        (batch_size, 1, 28, 28)
flattened input:        (batch_size, 784)
latent vector:          (batch_size, 32)
decoder output flat:    (batch_size, 784)
reconstruction image:   (batch_size, 1, 28, 28)
loss:                   scalar
```

This shape contract is the main thing to understand before coding. Most early
autoencoder bugs are shape bugs.

### 10. Baseline Expectation

The first autoencoder should beat a naive reconstruction baseline, but it does
not need to be impressive.

A useful starting expectation:

- early reconstructions may be blurry
- the model should capture the rough object or digit shape
- validation reconstruction loss should decrease during training
- visual examples should look more like the input than a blank or average image

Do not call the model good just because loss decreases. Check the images.

### 11. Denoising Autoencoder Preview

A denoising autoencoder changes the input-target pair:

```text
noisy image -> clean original image
```

The target is still the clean image. The model must learn to remove corruption
instead of merely copying pixels.

This is not Session 1 implementation work. For now, just understand why it is
different from the plain autoencoder.

### 12. Representation Learning Compared With Supervised Learning

Supervised learning uses labels to learn features that help a prediction task.

Autoencoder-style representation learning uses reconstruction pressure to learn
features without class labels.

The important limitation:

```text
Features that help reconstruction are not always the same features that help
classification or reasoning.
```

## Session 1 Task

Create the Week 6 experiment contract before writing any model code.

Write a short `EXPERIMENT_PLAN.md` draft, or notes that can become that file,
with these decisions:

- Dataset: use Fashion-MNIST unless there is a concrete setup problem.
- Input shape: `(batch_size, 1, 28, 28)`.
- First model type: fully connected plain autoencoder.
- Flattened dimension: `784`.
- First latent dimension: `32`.
- Reconstruction loss: mean squared error.
- Main metric: validation reconstruction MSE.
- First visual output: grid of original images beside reconstructions.
- Rule: do not use the test split while choosing architecture or latent size.

Then answer these checkpoint questions in your own words:

1. What is the target output of an autoencoder?
2. Why does the bottleneck matter?
3. What does the decoder receive as input?
4. Why is validation reconstruction loss not enough by itself?
5. What changes in a denoising autoencoder?

## Definition Of Done

Session 1 is done when you can state the dataset, shape contract, latent size,
loss, metric, and first visual output without guessing.

The next session should only begin after the experiment contract is clear.

## Completion Note

Session 1 is complete as of 2026-08-31.

The accepted Week 6 experiment contract is:

- dataset: Fashion-MNIST, with MNIST as fallback if setup fails
- model type: fully connected plain autoencoder
- encoder shape: `784 -> 128 -> 32`
- latent shape: `(batch_size, 32)`
- decoder shape: `32 -> 128 -> 784`
- reconstruction loss: mean squared error between reconstructed image and
  original image
- main validation metric: validation reconstruction MSE
- first visual output: fixed grid of original images beside reconstructions
- split rule: use train for optimization, validation for design checks, and
  reserve test for final reporting

The quiz showed enough understanding to move to Session 2. The main wording to
keep precise is that denoising changes the input-target pair: the input is
corrupted or noisy, while the target remains the clean original image.
