# Twenty-Epoch MLP/CNN Comparison

Date: 2026-09-16. Status: CNN completed through epoch 20; continuation exit 0.
The MLP result is reused from the completed 2026-09-15 run, freshly verified.

## Result

Both model pairs completed 8,600 paired D/G updates over 20 epochs, with
55,000 MNIST images per epoch, seed 42, batch 128, Adam LR 0.0002 and betas
(0.5, 0.999), CPU/two threads, and identical data split and evaluation-noise
seeds. Final epoch 20 was declared before this run. Test images were unused
in measured optimization, scoring, or checkpoint selection.

| Pair | G parameters | D parameters | Completed-epoch training seconds | Final D loss | Final G loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| MLP, reused | 242,832 | 233,985 | 100.959 | 0.525343 | 2.534042 |
| CNN, extended | 482,049 | 355,073 | 1,098.076 | 0.729749 | 1.967853 |

[Full final comparison](mlp_vs_cnn_epoch20.png) ·
[Loss comparison](mlp_vs_cnn_losses_20epochs.png) ·
[Epoch-5/10/20 progression](epoch_05_10_20_comparison.png)

The complete final panel shows cleaner backgrounds, more recognizable digits,
and visibly greater shape variety for the CNN pair than the MLP pair. The
MLP still produces many 1-like outputs and residual speckle. CNN images improve
substantially between epochs 5 and 10 and become more consistently digit-like
by epoch 20; some remain ambiguous or malformed. These are qualitative
observations from full panels, not generated-class coverage measurements.

The loss curves show a major CNN transition around epochs 5–6 followed by
slower changes. The cause is not isolated by these records. Each generator
faces a different changing discriminator, so the losses cannot rank image
quality across models. Both networks and their capacities change; equal
update budgets do not match FLOPs or runtime. Initialization also consumes
different amounts of global RNG, so equal seeds do not make training-noise
draws identical across architectures. One seed does not establish general
superiority, convergence, absence of memorization, or the isolated value of
convolution. The CNN generator uses resize-then-convolution, not canonical
DCGAN transposed convolutions.

Final CNN real-image acceptance: 96.66% training and 95.68% validation at
threshold 0.5; mean real probabilities 0.904775 and 0.895347. These real-only
audits are not overall binary accuracy or generation-quality metrics.

## Interruption And Verified Recovery

The initial process was terminated with exit 143 after epoch 12. No training
process remained. The exact cause of termination was not established. Twelve
complete checkpoint records were intact; both Adam optimizers were at step
5,160. The run-local [continuation helper](continue_run.py) restored both
model states, both optimizer states, torch RNG, and training-loader RNG from
the verified epoch-12 checkpoint before completing epochs 13–20.

The helper reconstructed and checked the original reference scores before
restoration. A synthetic interrupted-versus-uninterrupted experiment matched
metrics, parameters, optimizer states, and both RNG states exactly. Original
configuration and core source snapshots remain unchanged; additional execution
provenance and the continuation helper hash are in [continuation.json](continuation.json).

Completed-epoch training took 1,098.076 seconds. The continuation's active
training/artifact/audit phase took 436.542 seconds. Total wall elapsed was
1,504.219 seconds and explicitly includes interruption/recovery delay and any
lost partial-epoch work. It is not directly comparable to the MLP's recorded
active total time; historical CPU training times are not a dedicated benchmark.

## Commands

The exact original command is recorded in [config.json](config.json).
To reproduce uninterrupted, use a fresh output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-cnn20-reproduction OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --architecture cnn --device cpu --epochs 20 --batch-size 128 --seed 42 --noise-dim 64 --lr 0.0002 --threads 2 --num-workers 0 --out-dir phase01_research_engineering/week08_gans/results/runs/session4_cnn20_reproduction
```

The continuation was executed once with `continue_run.py`, using the same
two-thread environment. It intentionally refuses to rerun a completed run
or overwrite existing continuation provenance. [stdout.txt](stdout.txt)
combines both execution segments; original segment outputs are also retained.

The [audit/figure script](audit_and_compare.py) reproduces the final verification
and figures from retained checkpoints. Its `--prefix-only` option verifies
the first five CNN epochs against the earlier run.

## Verification And Artifacts

- All five core implementation/test hashes match the five-epoch CNN run,
  which passed 25 tests with zero skips. Those unchanged-source tests were
  reused; the new restoration equivalence check and helper lint passed.
- The first five CNN epochs exactly match original non-runtime metrics,
  epoch-0–5 grids, model/optimizer states, and both RNG states.
- Both models' twenty checkpoints pass hashes: forty verified checkpoint
  references in total. The final CNN Adam states each record 8,600 updates.
- Both architectures reproduce their epoch-5/10/20 monitoring grids and
  final independent-noise panels byte-for-byte.
- All 29 PNG figures verify. The three comparison figures were visually
  inspected; a progression-title spacing issue was corrected.
- CNN real-image CSV and JSON records agree, including 55,000 training and
  5,000 validation examples. Full output is in [verification.txt](verification.txt).

CNN checkpoints are under
`/home/rui.gao/data/ai_research_roadmap/checkpoints/week08_gans/session4_mnist_cnn_seed42_20epochs_20260916/`.
Paths and hashes are recorded in [metrics.json](metrics.json). The original
five-epoch runs and historical twenty-epoch MLP run were preserved.

The requested execution is complete. Interpreting the dynamics and writing
the final Week 8 lab note remain learning activities, not unfinished training.
