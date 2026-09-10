# Results

Week 6 has one measured plain fully connected autoencoder run and one measured
denoising run on Fashion-MNIST. The runs are intentionally small; they are
learning exercises for reconstruction loss, bottlenecks, held-out validation,
baseline comparison, denoising behavior, and visual inspection.

## Summary Table

| Run ID | Model | Noise Std | Latent Dim | Epochs | Final Train MSE | Final Val MSE | Test MSE | Average-Image Val MSE | Noisy-Input Val MSE | Runtime | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `session3_plain_autoencoder` | Fully connected autoencoder | 0.0 | 32 | 5 | 0.014930 | 0.014738 | 0.014750 | 0.088029 | n/a | 20.764s | Validation loss decreased every epoch; the learned model clearly beats the average-image baseline on MSE. |
| `session4_denoising_autoencoder` | Fully connected denoising autoencoder | 0.3 | 32 | 5 | 0.018452 | 0.018124 | 0.018131 | 0.088029 | 0.052527 | 21.559s | Validation loss decreased every epoch; the learned model beats both the noisy-input and average-image baselines. |

Artifacts:

```text
results/session3_metrics.json
results/session3_reconstructions.png
results/session4_denoising_metrics.json
results/session4_denoising_reconstructions.png
results/session3_vs_session4_metrics.png
results/latent_interpolation_two_real_images.png
```

## Interpretation

The plain autoencoder learned a useful first reconstruction mapping under the
chosen small bottleneck. Validation MSE fell from 0.022648 after epoch 0 to
0.014738 after epoch 4. The learned reconstruction loss is much lower than the
average-image validation baseline of 0.088029, so the model is reconstructing
per-image structure instead of behaving like a single generic image.

The reconstruction grid supports the scalar result. Shoes, pants, shirts, and
tops remain recognizable. Fine texture and sharp edges are blurred, which is
expected for a small fully connected model with a 32-dimensional bottleneck.

The denoising run is a harder objective than the plain reconstruction run
because the input is corrupted but the target stays clean. Its final validation
MSE of 0.018124 is higher than the plain run's 0.014738, which is expected.
The stronger comparison is against the corrupted input itself: the denoising
model beats the noisy-input validation baseline of 0.052527 by a wide margin.
The three-row grid shows that most visible pixel noise is removed while coarse
Fashion-MNIST silhouettes remain recognizable.

The exploratory latent-interpolation probe encodes two real test images
separately, then decodes linear points between their latent vectors. The saved
figure shows an ankle boot gradually changing into trousers through blurry
intermediate shapes. This is a useful representation-learning probe, but it is
not diffusion-style generation from pure noise.

This result should not be interpreted as evidence that the latent vector has
semantic class meaning. It shows that the bottleneck stores enough information
to reconstruct coarse Fashion-MNIST shapes better than an average-image
baseline.

## Week 6 Completion Note

`LAB_NOTE.md` closes the Week 6 learning loop by recording the bottleneck
interpretation, the difference between supervised classification and
reconstruction-based representation learning, the main failure modes, and the
carry-forward questions for Week 7 VAEs.

## Limitations

- This is one seed and one small architecture.
- No YAML config file exists yet; reproducibility currently comes from CLI
  arguments and seed control.
- No formal pytest smoke tests exist yet.
- Test loss is reported for fixed Session 3 and Session 4 runs; do not tune future
  choices against the test split.
- No broad latent-space visualization or clustering probe exists yet, so global
  representation structure is not directly inspected.
