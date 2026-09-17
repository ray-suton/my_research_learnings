# Session 4 - Controlled CNN Comparison

## Twenty-Epoch Follow-Up - Completed

On 2026-09-16 the learner requested taking both model pairs to twenty epochs.
The existing Session 3 twenty-epoch MLP run has matching settings and verified
epoch-5/10/20 checkpoint hashes, so it was reused. The CNN completed a fresh
twenty-epoch run with the same seed and core source/settings. Its first five
epochs exactly match the retained five-epoch CNN run. This repeats one seed's
training trajectory; it is not independent replication. The initial process
was terminated with exit 143 after twelve saved epochs. The continuation
restores the verified epoch-12 models, both Adam states, and both training/data
RNG states. A synthetic interrupted-versus-uninterrupted check passed exactly.
Core model, training, loader, and runner source remain unchanged; a run-local
continuation helper and its provenance document the recovery.

Both pairs reached the predeclared epoch-20 checkpoint and 8,600 paired
updates. All forty checkpoint hashes passed; both models exactly reproduce
their saved epoch-5/10/20 grids and final panels. Previous runs are retained.
No additional tuning or training extension is implied.

| Pair | Completed-epoch training seconds | Final D loss | Final G loss |
| --- | ---: | ---: | ---: |
| MLP, reused from September 15 | 100.959 | 0.525343 | 2.534042 |
| CNN, completed September 16 | 1,098.076 | 0.729749 | 1.967853 |

[Final side-by-side](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_epoch20.png) ·
[Loss comparison](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_losses_20epochs.png) ·
[Epoch-5/10/20 progression](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/epoch_05_10_20_comparison.png) ·
[Run notes and recovery](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/notes.md)

The CNN panel contains cleaner, more recognizable and visibly more varied
digit shapes than the MLP panel. Some CNN digits remain ambiguous; the MLP
still has many 1-like outputs and residual speckle. This is a qualitative
comparison of two architecture pairs, not quantitative class coverage or
independent-seed evidence. Total CNN wall time was 1,504.219 seconds, including
interruption/recovery; the training-time column includes completed epochs only.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-cnn20-run-20260916 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --architecture cnn --device cpu --epochs 20 --batch-size 128 --seed 42 --noise-dim 64 --lr 0.0002 --threads 2 --num-workers 0 --out-dir phase01_research_engineering/week08_gans/results/runs/session4_mnist_cnn_seed42_20epochs_20260916
```

## Original Five-Epoch Checkpoint

The learner designed the CNN generator and discriminator through guided
discussion on 2026-09-16. They then explicitly requested implementing the
runner/configuration selection and executing the CNN pair for five epochs.
This authorizes the runner integration, supporting verification, and this run.
Normal lessons remain learner-owned coding with instructor guidance.

The five-epoch CNN experiment completed on 2026-09-16. All 25 checks passed:
the existing eleven behavioral checks run against each model pair, plus
three architecture-selection checks. The saved final checkpoint exactly
reproduces both CNN sample panels. The practical comparison is complete;
the subsequent discussion and final closure are recorded in [Session 5](session5.md).

[Full MLP/CNN comparison](results/runs/session4_mnist_cnn_seed42_20260916/mlp_vs_cnn_epoch05.png) ·
[Run notes](results/runs/session4_mnist_cnn_seed42_20260916/notes.md) ·
[Metrics](results/runs/session4_mnist_cnn_seed42_20260916/metrics.json) ·
[Verification](results/runs/session4_mnist_cnn_seed42_20260916/verification.txt)

## Declared Comparison

Compare the preserved five-epoch Session 3 MLP baseline with the learner's
CNN pair. The question is whether this particular convolutional architecture
produces more recognizable and varied MNIST samples within the same update
budget. Retain complete fixed-noise panels and report failures as well as
improvements. Loss values remain training diagnostics, not image-quality ranks.

| Setting | Both five-epoch runs |
| --- | --- |
| Training / validation / test | 55,000 / 5,000 / 10,000 MNIST images |
| Image format | 1 × 28 × 28, normalized to [-1, 1] |
| Training/split seed | 42 |
| Epochs / batch size | 5 / 128; final short batch retained |
| Paired D/G update budget | 2,150 |
| Optimizers | Separate Adam instances, LR 0.0002, betas (0.5, 0.999) |
| Loss | Sigmoid probabilities with mean BCE; D sums real and fake means |
| Noise | 64-dimensional standard normal |
| Monitoring / final-panel seeds | 123 / 456 |
| Device / CPU threads / loader workers | CPU / 2 / 0 |
| Checkpoint selection | Predeclared final epoch 5 |
| Test images | Excluded from measured optimization, scoring, and selection |

## Architecture

Generator: linear 64 → 128×7×7, ReLU, reshape, nearest-neighbor upsampling to
14×14, 3×3 convolution 128 → 64 with padding 1, ReLU, upsampling to 28×28,
3×3 convolution 64 → 1 with padding 1, Tanh.

Discriminator: three 3×3 stride-2, padding-1 convolutions with channels
1 → 32 → 64 → 128 and LeakyReLU (slope 0.01); spatial sizes 28 → 14 → 7 → 4;
flatten 2,048 features; linear 2,048 → 128, LeakyReLU, linear 128 → 1, Sigmoid.
There is no batch normalization or dropout. This is a CNN GAN using
resize-then-convolution, rather than an exact reproduction of the original
DCGAN generator architecture.

| Model | MLP baseline parameters | CNN parameters |
| --- | ---: | ---: |
| Generator | 242,832 | 482,049 |
| Discriminator | 233,985 | 355,073 |

Both networks and their parameter counts change. This comparison cannot
isolate convolution alone or identify which network caused an improvement.
One seed and visual inspection cannot establish quantitative mode coverage,
absence of memorization, or general superiority. Shared noise seeds do not
make individual sample positions semantically paired across architectures.
Model initialization consumes different amounts of the global RNG, so equal
training seeds do not imply identical training-noise draws. Data ordering and
evaluation noise are independently seeded. The update budget is matched;
parameter counts, FLOPs, and elapsed runtime are not matched.

## Command And Artifact Policy

The new argument defaults to `--architecture mlp`, preserving existing CLI
behavior. `--architecture cnn` selects the learner's two CNN classes. The
resolved config includes class names, architecture descriptions, parameter
counts, exact settings, source hashes, and data-split identities.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-cnn-run-20260916 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --architecture cnn --device cpu --epochs 5 --batch-size 128 --seed 42 --noise-dim 64 --lr 0.0002 --threads 2 --num-workers 0 --out-dir phase01_research_engineering/week08_gans/results/runs/session4_mnist_cnn_seed42_20260916
```

Use fresh output and checkpoint directories; keep the Session 3 runs intact.
Checkpoints use the canonical external data root. The final Week 8
[lab note](LAB_NOTE.md) was completed during Session 5.

## Measured Result

The run completed 2,150 paired updates in 262.733 seconds of training
(269.021 seconds including the recorded artifact/audit phase). Final D/G
losses were 0.201683 / 9.202752. These values do not rank image quality
against the MLP losses. Training was about 10.3 times slower than the recorded
25.523-second MLP run; these historical wall-clock measurements are not a
dedicated hardware benchmark.

Inspection of the complete final seed-456 panels shows much cleaner
backgrounds and thicker, more continuous strokes than the five-epoch MLP.
Many CNN outputs are still malformed, incomplete, or ambiguous. This supports
an improvement in local visual structure for this architecture pair and
budget, without establishing full digit coverage or general superiority.

The final discriminator accepted 61.18% of training real images and 61.30%
of validation real images at threshold 0.5. The small gap does not prove
absence of overfitting, and these real-only rates are not overall binary
accuracy or generation-quality metrics. D was highly confident during early
epochs; the late change and remaining malformed images should be discussed
alongside the full loss/score trajectory.

All five checkpoint hashes, five source snapshots, data-split identities,
epoch/sample/update counts, and discriminator CSV/JSON records passed the
artifact audit. Both CNN final panels and the original MLP final panel were
reproduced byte-for-byte from their saved epoch-5 checkpoints.
