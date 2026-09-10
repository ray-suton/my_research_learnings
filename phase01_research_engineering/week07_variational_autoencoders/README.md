# Week 7 - Variational Autoencoders

## Purpose

Week 7 moves from deterministic autoencoders into latent-variable generative
modeling. The goal is to understand how a VAE changes the Week 6 autoencoder
setup by learning a distribution over latent vectors, sampling through the
reparameterization trick, and balancing reconstruction quality against KL
regularization.

This is not meant to build a strong image generator. The target is a small,
inspectable Fashion-MNIST VAE with clear loss accounting, reconstruction
figures, generated samples, and a careful comparison against the Week 6 plain
autoencoder.

## Scope

- Study ELBO, encoder outputs, decoder outputs, `mu`, `logvar`,
  reparameterization, reconstruction loss, and KL divergence.
- Reuse Fashion-MNIST and the existing `AI_ROADMAP_DATA` policy.
- Implement a small fully connected VAE after the shape and loss contract are
  clear.
- Track reconstruction loss and KL loss separately, not only total loss.
- Save reconstructions from held-out images and generated samples from
  prior-sampled latent vectors.
- Run a small latent-size ablation after the first VAE is stable.
- Compare the VAE against the Week 6 plain autoencoder without overclaiming
  generation quality.

## Flexible Session Plan

### Session 1 - Concepts, Objective, and Shape Contract

- Explain the difference between a deterministic autoencoder latent vector and
  a VAE latent distribution.
- Define the encoder output contract: `mu` and `logvar`.
- Define the reparameterization step:

```text
std = exp(0.5 * logvar)
epsilon ~ Normal(0, 1)
z = mu + std * epsilon
```

- Define the first loss contract: reconstruction term plus KL term.
- Choose Fashion-MNIST, latent dimension 32, and the same train/validation/test
  split discipline used in Week 6.

Deliverable: documented dataset, tensor shapes, loss terms, baseline, and
falsification conditions.

### Session 2 - Minimum Trainable VAE

- Build or adapt the Fashion-MNIST dataloader.
- Implement the first fully connected VAE:

```text
image -> flatten -> hidden -> mu/logvar -> z -> hidden -> reconstructed image
```

- Add smoke checks for batch shapes, `mu`, `logvar`, sampled `z`, decoder
  output, scalar loss, and one tiny train step.
- Keep generated samples separate from reconstructions.

Deliverable: runnable VAE model, loss helper, and smoke checks.

Completed on 2026-09-08: all eight tests in `tests/test_session2_smoke.py` pass.
See `session2.md` for the verified contract and reproduction command.

### Session 3 - First Measured Run And KL Debugging

- Run one small measured VAE training job.
- Record train/validation reconstruction term, KL term, total loss, and
  reconstruction MSE.
- Save held-out reconstructions and prior-sampled generated images.
- Check for blank outputs, KL collapse, KL explosion, or a decoder that ignores
  the latent vector.

Deliverable: first measured VAE run with loss components and figures.

### Session 4 - Latent-Size Ablation

- Compare a small set of latent dimensions, likely 8, 16, 32, and 64.
- Keep dataset split, seed, architecture family, optimizer, and training length
  fixed unless the run log records the exception.
- Compare reconstruction quality, KL behavior, generated samples, and runtime.

Deliverable: latent-size comparison table and small visual comparison.

### Session 5 - Lab Note And Autoencoder Comparison

- Write the Week 7 lab note.
- Compare the Week 7 VAE against the Week 6 deterministic autoencoder.
- Explain what sampling from the prior adds, what KL regularization costs, and
  why blurry samples are expected for a small first model.
- Record what should carry forward to Week 8 GANs.

Deliverable: final Week 7 results table, visuals, and lab note.

## Working Rules

- Do not compare VAE total loss directly against Week 6 autoencoder MSE. The
  VAE objective includes a KL term and may use different reduction scale.
- Always report reconstruction and KL terms separately.
- Use validation data for design checks and reserve the test split for final
  fixed-run reporting.
- Treat generated samples from `z ~ Normal(0, I)` as distinct from decoded
  interpolations between real examples.
- Do not claim semantic latent structure without direct evidence.
- Keep datasets, checkpoints, and bulky artifacts outside Git.

## Folder Guide

- `EXPERIMENT_PLAN.md`: learning question, dataset decision, loss contract,
  baseline, metrics, falsification conditions, and expected outputs.
- `session1.md`: Session 1 concept notes and checkpoint questions.
- `session2.md`: completed implementation checkpoint and smoke-test evidence.
- `session3.md`: completed first measured run and KL-debugging results.
- `session4.md`: completed latent-size comparison and validation-only selection.
- `session5.md`: completed matched AE/VAE comparison and understanding-review handoff.
- `RUN_LOG.md`: commands and outcomes after they actually happen.
- `RESULTS.md`: measured results and comparison notes.
- `configs/`: guide to recorded per-run JSON configurations.
- `tests/`: eight Session 2 checks and five experiment checks.
- `results/`: small tracked metrics summaries and figures only.
- `loader.py`: cached Fashion-MNIST data and seeded split helpers.
- `model.py`: VAE encoder, reparameterization, and decoder.
- `loss_helper.py`: mean-MSE scaling, KL, and total-loss helper.
- `train.py`: measured-run CLI, training/evaluation, checkpoint and figure saving.
- `run_sessions.py`: fixed baseline, latent ablation, selection, and comparison sequence.
- `compare_autoencoder.py`: matched plain-AE baseline and final held-out comparison.
- `LAB_NOTE.md`: final Week 7 results, limitations, and conceptual takeaways.

## Current Checkpoint

Sessions 1–5 have completed implementation, experiments, and documentation.
The basic intuition review completed through guided discussion on 2026-09-10.
See `session5.md` for what was covered and the assessment limits. Week 8 GANs
is next; no Week 8 implementation has started.

The initial latent-32 VAE reached validation sampled MSE 0.025294. The fixed
8/16/32/64 latent comparison selected size 8 using validation data only. On the
held-out test set, the matched AE scored 0.020632 MSE; the selected VAE scored
0.024081 using posterior samples and 0.020514 using decode(mu). One short seed
does not establish general superiority. See `RESULTS.md` and `LAB_NOTE.md`.

## Reproduce And Verify

From the repository root, run the two saved check scripts:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 phase01_research_engineering/week07_variational_autoencoders/tests/test_session2_smoke.py
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 phase01_research_engineering/week07_variational_autoencoders/tests/test_experiment_checks.py
```

All 13 checks pass. The full fixed experiment sequence is:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/week07_vae_matplotlib python3 -u phase01_research_engineering/week07_variational_autoencoders/run_sessions.py --output-dir phase01_research_engineering/week07_variational_autoencoders/results/reproduction_run --checkpoint-dir /tmp/week07_vae_reproduction
```

Use unused output/checkpoint paths each time. The completed run is retained at
`results/completed_20260908/`; its exact invocation and individual run commands
are saved there. Checkpoints are in `/tmp/week07_vae_20260908/`, a temporary
writable exception to the external artifact policy. They may expire; the
metrics, figures, source, and commands remain available for regeneration.

Data must already exist at the cache path used by `loader.py`; these commands
never download data. Figures use a headless Matplotlib backend. Measured runs
used CPU; the CUDA path has not been tested.
