# Session 3 - First Measured VAE Run and KL Debugging

## Status

Completed on 2026-09-08.

This session turned the working VAE training step into a measured five-epoch
Fashion-MNIST run. The model has 784 input, 128 hidden, and 32 latent dimensions.
The official test set was not evaluated during this baseline run.

## What Was Built

- `train.py` now has a reproducible CLI, training and validation summaries,
  configuration saving, source hashes, and an explicit checkpoint path.
- Epoch summaries weight each batch by its actual image count, including the
  smaller final batch.
- The four loss fields are separate: per-pixel MSE, reconstruction term
  (`784 * MSE`), KL, and total (`reconstruction term + beta * KL`).
- Validation uses one posterior draw per image with an isolated fixed seed.
  Decoding `mu` is recorded separately as `mean_latent_mse`; it is not the same
  calculation as the posterior-sampled reconstruction metric.
- The run saves fixed validation reconstructions, independent prior samples,
  loss curves, and machine-readable metrics.

## Run

Settings: Adam, learning rate 0.001, batch size 128, beta 1, seed 42,
validation seed 123, five epochs, CPU with one thread. The cached dataset uses
55,000 training and 5,000 validation images, with no download.

Exact command from [command.txt](results/completed_20260908/session3_latent32/command.txt):

```bash
/home/rui.gao/anaconda3/bin/python3 /home/rui.gao/Desktop/my_research_learning_roadmap/phase01_research_engineering/week07_variational_autoencoders/train.py --run-name session3_latent32 --latent 32 --epochs 5 --batch 128 --hidden 128 --lr 0.001 --beta 1 --seed 42 --validation-seed 123 --device cpu --threads 1 --output-dir /home/rui.gao/Desktop/my_research_learning_roadmap/phase01_research_engineering/week07_variational_autoencoders/results/completed_20260908/session3_latent32 --checkpoint /tmp/week07_vae_20260908/session3_latent32.pt
```

The actual environment was Python 3.13.5 and PyTorch 2.8.0+cu126; this run used
CPU despite the CUDA-enabled PyTorch build. Full settings are in
[config.json](results/completed_20260908/session3_latent32/config.json).

The checkpoint is `/tmp/week07_vae_20260908/session3_latent32.pt`. It is temporary
and may expire when `/tmp` is cleaned. The repository retains the command,
metrics, and figures, so the run can be repeated if the weights are removed.
Choose fresh output and checkpoint paths for a rerun; existing outputs are
protected from overwrite.

## Results

From [metrics.json](results/completed_20260908/session3_latent32/metrics.json):

| Epoch | Train pixel MSE | Validation pixel MSE | Validation KL | Validation total |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.053096 | 0.035588 | 7.870344 | 35.771591 |
| 2 | 0.032278 | 0.030199 | 7.625436 | 31.301191 |
| 3 | 0.028555 | 0.028061 | 7.307601 | 29.307808 |
| 4 | 0.026858 | 0.026488 | 7.586728 | 28.353562 |
| 5 | 0.025854 | 0.025294 | 7.970769 | 27.801203 |

The table uses posterior-sampled reconstruction. Final validation results also
include:

- Reconstruction term: **19.830434**.
- Deterministic `decode(mu)` pixel MSE: **0.021970**.
- Dimensions with mean KL greater than 0.01: **9 of 32**.
- Runtime: **16.806 seconds** for training and validation; figure generation
  and checkpoint saving are excluded.
- Test result: **not evaluated** (`test_evaluated: false`).

Saved artifacts:

- [Reconstruction grid](results/completed_20260908/session3_latent32/reconstructions.png):
  16 fixed validation images, their `decode(mu)` reconstructions, and posterior
  draws shown separately.
- [Prior-sample grid](results/completed_20260908/session3_latent32/prior_samples.png):
  16 latent vectors drawn from `N(0, I)` and decoded without input images.
- [Loss curves](results/completed_20260908/session3_latent32/loss_curves.png).
- [Console output](results/completed_20260908/session3_latent32/stdout.log).

## Takeaways

- Validation reconstruction MSE and total loss improved across the five epochs.
  KL remained finite throughout the recorded history; it did not have to fall
  every epoch for the combined objective to improve.
- The inspected reconstructions retain broad shoe, shirt, and bag silhouettes,
  but edges and small details are blurry. Posterior draws can change individual
  reconstructions, which is why their metric is separate from `decode(mu)`.
- The inspected prior grid contains different clothing-like shapes and some
  ambiguous outputs. It demonstrates input-free decoding, not a quantitative
  claim about generation quality or coverage of the dataset.
- Many latent dimensions have little KL after this short run. The 0.01 cutoff
  is a diagnostic convention; 9 active dimensions alone does not establish
  posterior collapse or identify the best latent size.
- This is one seed and five epochs. It does not establish convergence or broad
  superiority over another model, and VAE total loss should not be compared
  directly with an AE's per-pixel MSE.

## Next Session

[Session 4](session4.md) compares latent sizes while keeping the split, seed,
training budget, beta, and other settings fixed. Compare validation metrics and
saved reconstructions/prior samples, then choose the VAE configuration using
validation evidence before the final AE comparison and test evaluation.
