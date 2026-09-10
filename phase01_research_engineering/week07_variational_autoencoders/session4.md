# Session 4 - Latent-Size Comparison

## Status

Completed on 2026-09-08. Compared latent sizes 8, 16, 32, and 64 using the same
Fashion-MNIST split, five epochs, hidden size 128, batch size 128, Adam learning
rate 0.001, beta 1, seed 42, and one CPU thread.

## What Was Built

- `run_sessions.py` runs the Session 3 baseline and three additional latent
  sizes, saves the comparison table and figures, and locks a validation-only
  choice before any final test evaluation.
- The latent-32 run is the existing Session 3 baseline, not a duplicate run.
- Each run retains configuration, source hashes, five-epoch history, command,
  reconstruction grid, prior samples, and loss curves.

## Results

All MSE values below are per-pixel validation errors, averaged by image count.

| Latent size | Sampled MSE | Decode(mu) MSE | KL per image | Active dimensions | Parameters | Train + validation seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.024347 | 0.020744 | 7.984677 | 8 | 204832 | 16.337 |
| 16 | 0.025318 | 0.021108 | 8.129920 | 13 | 207920 | 16.462 |
| 32 | 0.025294 | 0.021970 | 7.970769 | 9 | 214096 | 16.806 |
| 64 | 0.026851 | 0.023316 | 7.694735 | 22 | 226448 | 16.740 |

An active dimension here means its average KL contribution exceeds 0.01 nats.
This is a descriptive threshold, not proof of semantic meaning or collapse.

**Selected latent size: 8.** The predeclared rule was the lowest final
validation sampled-reconstruction MSE, with smaller latent size breaking an
exact tie. The choice and checkpoint hash were saved in `selection.json`
before the Session 5 comparison used test data.

## Visual Check

The prior grids contain varied broad clothing silhouettes across all four
sizes. Some samples are blurry or ambiguous. The grid does not establish a
quantitative ranking of generation quality.

- [Comparison table and metrics](results/completed_20260908/session4_latent_ablation_metrics.json)
- [CSV table](results/completed_20260908/session4_latent_ablation.csv)
- [Metric curves](results/completed_20260908/session4_latent_ablation.png)
- [Prior samples by latent size](results/completed_20260908/session4_prior_comparison.png)
- [Locked selection](results/completed_20260908/selection.json)

## Takeaways

- A larger latent vector did not automatically improve reconstruction under
  this five-epoch budget.
- The chosen size was based on validation reconstruction, not test results or
  a favorable-looking sample grid.
- Different sizes also change parameter counts and initialization draws. One
  shared seed is not a repeated-run estimate of uncertainty.
- Runtimes were all about 16–17 seconds; these small differences are not a
  hardware-performance conclusion.

## Reproduce

The exact full invocation is saved in
`results/completed_20260908/command.txt`; individual invocations are in each run
directory. To rerun all experiments, use the command in `README.md` with fresh
output and checkpoint directories. Existing outputs are protected from overwrite.
The recorded checkpoints are under `/tmp/week07_vae_20260908/` and may expire.

## Next Session

Session 5 compares the locked latent-8 VAE with a freshly trained latent-8
plain autoencoder, using matched settings and a final held-out test evaluation.
