# Results

Week 7 Sessions 1–5 have completed implementation and experiment records.
The basic intuitive understanding review completed with guided practice on
2026-09-10; it was not a formal derivation exam.

## Session 3 - First Measured VAE

Five epochs, latent size 32, hidden size 128, batch 128, Adam LR 0.001,
beta 1, seed 42, validation seed 123, CPU.

| Final validation quantity | Value |
| --- | ---: |
| Posterior-sampled per-pixel MSE | 0.025294 |
| Decode(mu) per-pixel MSE | 0.021970 |
| Scaled reconstruction term | 19.830434 |
| KL per image | 7.970769 |
| Total objective | 27.801203 |

Sampled validation MSE fell from 0.035588 to 0.025294. No test data was used
for this baseline run. See [Session 3](session3.md).

## Session 4 - Latent Size

| Latent size | Validation sampled MSE | Validation decode(mu) MSE | Validation KL |
| ---: | ---: | ---: | ---: |
| 8 | 0.024347 | 0.020744 | 7.984677 |
| 16 | 0.025318 | 0.021108 | 8.129920 |
| 32 | 0.025294 | 0.021970 | 7.970769 |
| 64 | 0.026851 | 0.023316 | 7.694735 |

The predeclared final-validation sampled-MSE rule selected latent size 8.
The selection and checkpoint hash were written before final test evaluation.
One short seed does not establish that size 8 is generally optimal.
See [Session 4](session4.md).

## Session 5 - Matched Final Comparison

Both models use latent size 8 and the same split, seed, epoch budget, optimizer,
learning rate, batch size, and CPU setting. Their objectives and parameter
counts differ.

| Reconstruction policy | Validation MSE | Test MSE |
| --- | ---: | ---: |
| Plain AE | 0.020793 | 0.020632 |
| VAE, posterior sample | 0.024347 | 0.024081 |
| VAE, decode(mu) | 0.020744 | 0.020514 |

The selected VAE's test KL is 7.912745, reconstruction term is 18.879740, and
its total objective is 26.792484. Compare per-pixel MSE with per-pixel MSE;
VAE total loss is not on the same scale as AE MSE.

Mean-code VAE reconstruction and sampled reconstruction answer different
questions. The small AE-versus-mean-code difference here does not establish
superiority. The VAE prior grid contains broad clothing silhouettes; the AE's
arbitrary normal-code probes are much less coherent in the displayed sample.
This is a qualitative observation, not a generation benchmark.
See [Session 5](session5.md) and [LAB_NOTE.md](LAB_NOTE.md).

## Historical Week 6 Anchors

The earlier plain AE used latent size 32 and recorded validation/test MSE
0.014738/0.014750. The average-image validation baseline was 0.088029.
Those are historical context: the old evaluator averaged batch means, while
this week's comparison weights each example equally. The fresh matched
latent-8 AE is the comparison used above.

## Evidence

All measured artifacts are under `results/completed_20260908/`:

- Four VAE run directories with configuration, command, epoch history, source
  hashes, reconstruction grids, prior samples, and loss curves.
- `session4_latent_ablation_metrics.json` and `.csv`.
- `session4_latent_ablation.png` and `session4_prior_comparison.png`.
- `selection.json`, locked before the final comparison.
- `session5_comparison/session5_comparison_metrics.json` and two comparison grids.

The eight Session 2 checks and five experiment checks pass. The artifact audit
validated all 11 JSON files and all 16 PNGs, objective identities, split sizes,
selection order, source hashes, and checkpoint identity.

## Remaining Limits

- One seed, one dataset, five epochs, and qualitative generation inspection.
- No confidence intervals, generation metric, or semantic latent probe.
- CPU behavior was tested; CUDA execution was not tested.
- Checkpoints under `/tmp/week07_vae_20260908/` are temporary and may expire.
  Reproduction commands regenerate them in new output directories.
