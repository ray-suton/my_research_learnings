# Fashion-MNIST CNN GAN Extension

## Scope And Status

Requested on 2026-09-17 after Session 5 closure/documentation. The learner
clarified **CNN only, twenty training epochs**, not twenty independent runs.
Session 5 and the core Week 8 work remain closed; this is a separate extension.

The requested run completed all twenty epochs on 2026-09-17. Preflight passed
34 tests with zero skips, focused lint, and runner/loader/training type checks.
All four five-epoch invocations exited 0; the original configuration remained
byte-identical. Final checkpoint and dataset audits passed, including exact
sample replay. No downloads, MLP run, or extra seeds were needed.

[Run notes](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/notes.md) ·
[Real versus generated](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/real_vs_generated.png) ·
[5/10/20 progression](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/audit_epoch_05_10_20_progression.png) ·
[Loss graph](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/audit_d_g_losses.png)

## Measured Outcome

The CNN completed 8,600 paired updates. Final D/G BCE losses were
0.665465/1.862303; training took 1,045.312 seconds, and active training/artifact/
audit segments totaled 1,056.616 seconds, excluding startup and gaps.
Real-image acceptance was 81.63% on training versus 80.86% on validation;
these are real-only discriminator diagnostics, not generated-class coverage.

The displayed garment shapes become clearer with training, but many outputs
remain noisy or ambiguous and are concentrated on tops, dresses, and trousers.
Little obvious footwear or bag variety is visible in the retained panels.
This is a qualitative coverage concern, not measured class recall. One seed
does not establish broad quality, convergence, or absence of memorization.

All twenty checkpoint hashes/configs/optimizer-step counts, seven source
snapshots, raw data fingerprints/splits, and class CSV/JSON records verified.
Epoch-5/10/20 and independent final sample panels reproduce byte-for-byte.
All 28 PNGs passed integrity checks. Session 5 and this extension are complete;
Week 9 tokenization/text data is the next core roadmap topic.

## Declared Experiment

The following settings describe the completed run, not a request to launch
another experiment.

Use the learner's existing CNN generator and discriminator unchanged. The
question is what garment-like structure this architecture learns on a second
28×28 grayscale dataset under a fixed twenty-epoch budget. This is one seed,
not a claim of robust dataset transfer or a matched VAE/GAN comparison.

| Setting | Value |
| --- | --- |
| Dataset | Fashion-MNIST, explicitly selected and fingerprinted |
| Train / validation / test | 55,000 / 5,000 / 10,000 |
| Image format | 1 × 28 × 28, normalized to [-1, 1] |
| Architecture | Existing CnnGenerator / CNNDiscriminator |
| Parameters | G: 482,049; D: 355,073 |
| Epochs / batch size / seed | 20 / 128 / 42 |
| Paired D/G updates | 8,600 total, 430 per epoch |
| Optimizers | Separate Adam, LR 0.0002, betas (0.5, 0.999) |
| Loss | Sigmoid/mean BCE; D sums real and fake batch means |
| Noise width / monitoring / final seed | 64 / 123 / 456 |
| Device / threads / workers | CPU / 2 / 0 |
| Final checkpoint | Predeclared epoch 20 |

Digit labels are replaced with Fashion-MNIST garment class names in reports.
Labels remain unused by the unconditional GAN objectives. Test images must
not enter measured optimization, scoring, or checkpoint selection.

## Reliable Execution

The total twenty-epoch budget is recorded once in immutable config. Execute
four five-epoch segments, pausing at complete checkpoints and restoring both
model states, both Adam states, torch RNG, and training-loader RNG between
invocations. Resume checks dataset bytes/class/splits, source hashes,
environment, original reference scores, checkpoint identity, and history.
It accepts cleanly paused CPU runs and rejects training-setting overrides.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-fashion-run-20260917 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --dataset fashion-mnist --architecture cnn --device cpu --epochs 20 --chunk-epochs 5 --batch-size 128 --seed 42 --noise-dim 64 --lr 0.0002 --threads 2 --num-workers 0 --out-dir phase01_research_engineering/week08_gans/results/runs/fashion_mnist_cnn_seed42_20epochs_20260917
```

After each clean pause, use the same environment and:

```bash
python3 -u phase01_research_engineering/week08_gans/run_baseline.py --resume --out-dir phase01_research_engineering/week08_gans/results/runs/fashion_mnist_cnn_seed42_20epochs_20260917
```

Retain every checkpoint, epoch-0–20 monitoring grid, final independent-noise
panel, real/generated comparison, loss/score curves, and final real-image
class audit. Save checkpoints under the canonical external data root. Preserve
all earlier MNIST runs. Report measured runtime and visual failures honestly;
do not tune or extend the budget based on intermediate appearance or loss.

## Final Closure - 2026-09-17

The learner identified that generated outputs were predominantly sweatshirt-
or clothing-like and chose to finish GAN work after discussing this limitation.
The observation suggests partial mode collapse / mode dropping, with no
quantitative generated-class coverage claim.

The recorded training classes contain 5,452–5,555 examples each. The saved
real-image audit accepts 99.8–100% of validation footwear at threshold 0.5.
These facts rule out missing footwear training examples and show that D
accepts real footwear; they do not establish why G neglects those shapes or
prove that D supplies useful gradients toward missing modes. See the
[class diagnostics](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/discriminator_by_class.csv).

Conditional GANs and batch-aware discrimination were possible follow-ups,
not tested fixes. No new penalty, class-conditioning change, additional seed,
or longer run is planned. Preserve this completed experiment and continue
the roadmap with Week 9 tokenization/text data.
