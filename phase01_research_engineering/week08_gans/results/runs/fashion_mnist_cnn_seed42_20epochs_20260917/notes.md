# Fashion-MNIST CNN GAN - Twenty Epochs

Date: 2026-09-17. Status: completed; all four invocations exited 0.
Scope confirmed by the learner: CNN only, twenty epochs. Session 5 closure
and its documentation were completed before this separate experiment began.

## Setup

The unchanged learner CNN pair was trained from scratch on cached
Fashion-MNIST, using 55,000 training / 5,000 validation / 10,000 test images.
Images are grayscale 28×28, normalized to [-1,1]. Seed 42, batch 128,
64-dimensional Gaussian noise, separate Adam optimizers at LR 0.0002 and
betas (0.5,0.999), sigmoid/mean BCE, CPU/two threads, and no loader workers.
G has 482,049 parameters; D has 355,073. Class labels do not enter either
unconditional training objective. Test images were unused in measured
optimization, scoring, or checkpoint selection.

Twenty epochs and final checkpoint 20 were declared before training. Four
five-epoch invocations restored both networks, both Adam states, torch RNG,
and the training-loader RNG at complete epoch boundaries. The original
configuration remained byte-identical throughout. This was one experiment,
not twenty independent runs or four separate five-epoch trainings.

## Measurements

| Metric | Recorded value |
| --- | ---: |
| Epochs / paired D/G updates | 20 / 8,600 |
| Images processed per epoch | 55,000 |
| Final D real BCE | 0.377924 |
| Final D fake BCE | 0.287542 |
| Final D total BCE | 0.665465 |
| Final G BCE | 1.862303 |
| Completed-epoch training seconds | 1,045.312 |
| Active training/artifact/audit segment seconds | 1,056.616 |
| Peak process RSS, MiB | 955.188 |
| Real-image acceptance: training / validation | 81.63% / 80.86% |

Exact per-epoch terms and real-image class diagnostics are in
[metrics.json](metrics.json) and [discriminator_by_class.csv](discriminator_by_class.csv).
Losses are sample-weighted means; the D total sums real and fake terms.
Active segment timing excludes process startup and gaps between invocations.
Real-only acceptance is not overall binary accuracy or generated-class coverage.

## Images And Interpretation

[Real versus generated](real_vs_generated.png) ·
[Epoch-5/10/20 progression](audit_epoch_05_10_20_progression.png) ·
[D/G losses](audit_d_g_losses.png) ·
[Full loss/score diagnostics](training_curves.png)

The complete panels show clearer garment-like silhouettes with additional
training, particularly between epochs 5 and 10. Texture artifacts and
ambiguous shapes remain. Outputs in the displayed panels are concentrated
on tops, dresses, and trousers, with little obvious footwear or bag variety.
This suggests limited coverage in these samples; no classifier-based
generated-class recall or population-level coverage metric was measured.

All 64 examples in each fixed panel are retained. Monitoring uses noise seed
123; the final independent-noise panel uses seed 456. Real/generated columns
are independent examples, not paired reconstructions. No sample was selected
for being especially attractive.

One seed and one architecture cannot establish convergence, absence of
memorization, or broad generation quality. Neither GAN losses across datasets
nor comparison with the differently trained Week 7 VAE establishes model-family
superiority. No MLP Fashion-MNIST run or additional tuning was performed.

## Verification

- 34 preflight tests passed with zero skips, including dataset identity and
  exact uninterrupted/restored synthetic trajectories for both model pairs.
- Focused lint and runner/loader/train mypy checks passed.
- All seven source snapshots match their recorded and current hashes.
- Raw cached dataset fingerprints, split identities, and garment names match.
- All twenty checkpoints pass SHA-256, embedded-config, finite-weight/optimizer,
  and Adam-step checks, ending at 8,600 updates per optimizer.
- Invocation ranges are 1–5, 6–10, 11–15, 16–20; the first three paused cleanly.
- Epoch-5/10/20 monitoring grids and the final independent panel reproduce
  byte-for-byte from checkpoints. Class CSV/JSON records and counts agree.
- All 28 PNG figures passed integrity checks and final presentation was inspected.

The audit's first float32 loss-sum tolerance was overly strict (1e-9 against
a maximum observed residual of 2.0634e-9). It was aligned with the existing
1e-6 training-test tolerance and passed on rerun; recorded results were unchanged.
Full evidence is in [verification.txt](verification.txt).

## Reproduction And Files

The exact initial command is stored in [config.json](config.json). To reproduce
in a fresh directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-fashion-reproduction OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --dataset fashion-mnist --architecture cnn --device cpu --epochs 20 --chunk-epochs 5 --batch-size 128 --seed 42 --noise-dim 64 --lr 0.0002 --threads 2 --num-workers 0 --out-dir phase01_research_engineering/week08_gans/results/runs/fashion_cnn_reproduction
```

After each clean pause, use the same environment and `--resume --out-dir`
with that run directory. The saved total target and training settings are
loaded; they must not be overridden. Resume is currently supported for
cleanly paused CPU runs, not arbitrary interrupted states or CUDA.

[stdout.txt](stdout.txt) combines four separately retained invocation logs.
[audit_run.py](audit_run.py) reproduces the final verification and two audit
figures from saved checkpoints. Checkpoints live outside Git under
`/home/rui.gao/data/ai_research_roadmap/checkpoints/week08_gans/fashion_mnist_cnn_seed42_20epochs_20260917/`;
their paths/hashes are in metrics. Core training source and test snapshots
are retained under `sources/`. Historical MNIST run bundles remain intact.

The requested extension is complete. The next core roadmap topic is Week 9
tokenization and text data; no further GAN training is implied.
