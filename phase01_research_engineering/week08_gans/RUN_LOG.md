# Run Log

## Final GAN Topic Closure - 2026-09-17

The learner requested marking GAN work finished after the Fashion-MNIST
coverage discussion. Updated the session/lab/extension conclusions and roadmap
handoff. Recorded likely mode dropping as a qualitative limitation and
distinguished discussed remedies from executed experiments. No further GAN
training, penalty change, conditional model, or closing quiz is planned.
This closure updates documentation only; saved results and source are preserved.

## Session 5 Closure And Fashion-MNIST Preflight - 2026-09-17

At the learner's request, closed Session 5 through a conceptual discussion
and wrote [session5.md](session5.md) plus [LAB_NOTE.md](LAB_NOTE.md). Recorded
demonstrated reasoning separately from supplied explanations. Five closure
documents passed whitespace/fence and 65 local-link checks before training.
No extra quiz or claim of independent mastery was added.

The learner clarified the next task as **CNN only, twenty epochs on
Fashion-MNIST**. Cached 60,000/10,000 data loaded with downloads disabled;
normalization and class names were checked. Added explicit dataset selection,
data fingerprints, correct class-oriented reporting, and planned five-epoch
pause/resume segments in the existing runner. CNN model and update code
remain unchanged. All 34 tests passed with zero skips in 0.995 seconds;
focused flake8 and three-file mypy checks passed. The
[extension plan](fashion_mnist_extension.md) records the predeclared settings.
The measured run started after this documentation and verification and
subsequently completed, as recorded below.

## Fashion-MNIST CNN Completion - 2026-09-17

Executed the [declared initial command](fashion_mnist_extension.md), followed
by three `--resume --out-dir` invocations. All exited 0. The first three paused
at epochs 5, 10, and 15; the last finalized epoch 20. The config remained
byte-identical, and each resume verified data/source/environment/checkpoint
identity and original reference scores before restoring model/optimizer/RNG
states. All invocation commands and active timings are in metrics.json.

Completed 20 epochs / 8,600 paired updates. Final D/G losses:
0.665465/1.862303. Training: 1,045.312 seconds; active segment total:
1,056.616 seconds, excluding startup and gaps. Twenty external checkpoints
were retained. The final artifact audit verified all identities, optimizer
steps, class CSV/JSON counts, and exact epoch-5/10/20/final sample replay.
All 28 figures passed integrity checks and were visually inspected.

The first audit's 1e-9 tolerance on float32 loss sums was too strict; the
maximum residual was 2.0634e-9. Matching the existing 1e-6 training-test
tolerance resolved the audit without changing training data or results.
Full output is retained in [verification.txt](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/verification.txt).
The [run notes](results/runs/fashion_mnist_cnn_seed42_20epochs_20260917/notes.md)
record clearer garments with remaining texture and coverage limitations.

Record commands and outcomes only after execution. Distinguish documentation
checks, data checks, smoke tests, and measured experiments.

## Initialization - 2026-09-10

- Created the Week 8 documentation scaffold using the Week 7 initialization
  pattern.
- Read-only filesystem check: with `AI_ROADMAP_DATA` unset, the fallback root
  was `/home/rui.gao/data/ai_research_roadmap`; its `raw/torchvision/MNIST`
  directory was absent. The Fashion-MNIST directory was present.
- Selected MNIST as the planned dataset to follow the root roadmap.
- No data preparation, model code, executable tests, training, or sampling
  was run. Documentation checks are recorded in the session report.

## Session 1 Conceptual Discussion - 2026-09-11

- Concluded the guided discussion at the learner's request. Recorded answers,
  corrections, and assessment limits in [session1.md](session1.md#discussion-record---2026-09-11).
- Discussed the generator/discriminator goals, binary targets and separate
  losses, alternating updates, detach, mode collapse, and evaluation.
- The learner identified fixed-noise comparison as a controlled experiment.
  Practical loss, shape, normalization, and gradient checks carry into the
  opening of Session 2.
- No data preparation, model implementation, executable tests, training, or
  sampling occurred. The documentation closure checks are recorded in the
  [session report](../../.omx/reports/task-2026-09-11-215027.md).

## Session 2 Review, Assessment, And Closure - 2026-09-14/15

- The learner implemented model and training drafts and selected sigmoid
  output with plain BCE. Reviewed real/fake targets, detach, and alternating
  updates without implementing on the learner's behalf.
- The September 14 [preflight report](../../.omx/reports/task-2026-09-14-212821.md)
  records passing syntax/import checks and D probability/BCE checks for
  synthetic batches of 1 and 8. G construction failed on undefined `x`;
  static checks also found undefined `model` and `z` in `train.py`.
- No complete D/G update, parameter-isolation check, or repeated cycle passed
  on the current learner draft. Earlier passing checks from an assistant
  implementation that was reverted are not evidence about this code.
- The concepts/research-practice discussion concluded on September 14 with
  guided corrections. The learner distinguished gradients from optimizer
  updates, identified matching update budgets, and proposed seeds/folds.
  The limits of average G loss as a quality measure were explained.
- September 15: documented closure in [session2.md](session2.md). Current
  source hashes match the September 14 preflight; code was not changed and
  model tests or training were not rerun. The
  [closure report](../../.omx/reports/task-2026-09-15-103317.md) records document checks.
- Session 3 begins with the unresolved execution checks and MNIST preparation
  before any measured run. No download or data preparation occurred here.

## Session 3 Understanding And Documentation Closure - 2026-09-15

- The learner corrected G's constructor and batch-size handling. The
  [recheck](../../.omx/reports/task-2026-09-15-112340.md) passed both constructors
  and G-to-D shape/range checks for batches 1 and 8. An actual call to the
  training function stopped at undefined `model`; neither model's parameters
  changed in that attempted call. Other name/data-flow issues remain.
- The understanding/research-practice discussion concluded with guided
  corrections. It covered changing discriminator feedback, fixed-input
  scores, complete reporting trajectories, collapse, digit bias, and the
  evidence needed to distinguish overfitting from other explanations.
- The all-3s example was hypothetical. The requested per-digit diagnostic
  could not be run on a trained model: no MNIST cache, loader, measured run,
  or generated sample panels were available at the closure check.
- Source hashes still match the successful forward/blocked-training recheck.
  No model tests or training were rerun during documentation closure, and
  no source, dataset, or checkpoint was modified.
- [session3.md](session3.md) and the
  [closure report](../../.omx/reports/task-2026-09-15-124416.md) separate the
  completed understanding discussion from pending practical work.

## Session 3 Practical Baseline - 2026-09-15

The learner explicitly requested practical completion: make the baseline
runnable, train it, save the real-versus-generated figures, and finish the
run documentation. Necessary training repairs and supporting loader/runner/
checks were completed for this request. The learner's model.py was unchanged.

Data preparation executed from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/data/prepare_standard_datasets.py --dataset mnist
```

It downloaded/verified 60,000 official training and 10,000 test examples at
/home/rui.gao/data/ai_research_roadmap/raw/torchvision/MNIST.
Manifest: /home/rui.gao/data/ai_research_roadmap/manifests/prepare_standard_datasets_20260915T093506Z.json

Final verification command (11 tests, zero skips, exit 0):

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-mpl-tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/tests/test_session3_baseline.py
```

Measured command (exit 0):

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-mpl-run OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --device cpu --epochs 5 --batch-size 128 --seed 42 --threads 2 --out-dir phase01_research_engineering/week08_gans/results/runs/session3_mnist_mlp_seed42_20260915
```

The run completed all five epochs: 55,000 images and 430 minibatches each,
2,150 paired D/G iterations total. Training took 25.523 s on CPU with two
threads; the recorded training/artifact/audit phase took 31.290 s. CUDA was
unavailable. The predeclared epoch-5 checkpoint was selected without looking
for the lowest G loss or nicest samples.

The [run directory](results/runs/session3_mnist_mlp_seed42_20260915/notes.md) contains immutable configuration, source
snapshots, metrics, actual stdout, eleven PNGs, a per-digit discriminator audit,
and verification evidence. Five checkpoints are stored under:
/home/rui.gao/data/ai_research_roadmap/checkpoints/week08_gans/session3_mnist_mlp_seed42_20260915

All five checkpoint hashes and all executed source hashes were checked.
Reloading epoch 5 reproduced both saved generated-image panels byte-for-byte.
Figures were visually inspected. Final D/G losses: 0.394094 / 3.368496.
Samples have digit-like structures but remain noisy. Real-image acceptance
was 95.80% on training and 95.70% on validation data; these are not overall
binary-classification accuracies or generator-quality metrics.

## Twenty-Epoch Follow-Up - 2026-09-15

The learner requested longer training. Executed the unchanged runner in a
fresh directory; only the epoch budget changed. The initial five-epoch state
was verified identical to the preserved baseline.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-mpl-run OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --device cpu --epochs 20 --batch-size 128 --seed 42 --threads 2 --out-dir phase01_research_engineering/week08_gans/results/runs/session3_mnist_mlp_seed42_20epochs_20260915
```

Exit 0. All 20 epochs completed: 55,000 images/430 batches each, 8,600 paired
D/G iterations; 100.959 s training and 111.343 s
recorded training/artifact/audit phase. Final D/G losses: 0.525343 / 2.534042.
Source/test files were unchanged; no unit-test rerun or new dependency was
needed. Fresh artifact/state/replay checks passed. [Run notes](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/notes.md)
record the measured results and the qualitative diversity concern.

## Final Session 3 Closure - 2026-09-15

The learner confirmed the visible improvement at epoch 20 and requested
final documentation. Closed Session 3 with both the five-epoch baseline
and the same-settings twenty-epoch follow-up preserved. The conclusion credits
clearer individual images while recording the separate diversity concern.
No additional training, checkpoint selection, or source changes occurred
during this documentation closure. Existing metrics, verification records,
images, and checkpoint references support the completion claim.

See [session3.md](session3.md) for the final assessment and
the [closure report](../../.omx/reports/task-2026-09-15-203942.md) for document
validation. Session 4 is next; the final Week 8 lab note remains pending.

## Data Preparation

The following data and Session 3 verification entries retain their original
run scope. Session 4's execution is recorded below.

MNIST is prepared at the canonical external root. The recorded loader split
is 55,000 training / 5,000 validation / 10,000 test, seeded with 42. Data checks
verified disjoint indices, normalization, and independent loader RNG streams.
Downloads are disabled in the loader. The measured run used no test images
for optimization, scoring, or checkpoint selection.

## Smoke Checks

All 11 baseline checks passed. See [tests/README.md](tests/README.md) and the
retained [verification output](results/runs/session3_mnist_mlp_seed42_20260915/verification.txt). This supersedes the
earlier failed draft checks and is tied to the current run's source hashes.

## Measured Runs

| Run | Dataset | Epochs | Seed | Selected checkpoint | Status |
| --- | --- | --- | --- | --- | --- |
| [session3_mnist_mlp_seed42_20260915](results/runs/session3_mnist_mlp_seed42_20260915/notes.md) | MNIST | 5 | 42 | Final epoch 5, predeclared | Completed |
| [session3_mnist_mlp_seed42_20epochs_20260915](results/runs/session3_mnist_mlp_seed42_20epochs_20260915/notes.md) | MNIST | 20 | 42 | Final epoch 20, predeclared | Completed |
| [session4_mnist_cnn_seed42_20260916](results/runs/session4_mnist_cnn_seed42_20260916/notes.md) | MNIST | 5 | 42 | Final epoch 5, predeclared | Completed |
| [session4_mnist_cnn_seed42_20epochs_20260916](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/notes.md) | MNIST | 20 | 42 | Final epoch 20, predeclared | Completed with checkpoint recovery |

Preserve all runs, including imperfect samples and losses, rather than
overwriting them with later attempts.

## Session 4 CNN Comparison - 2026-09-16

The learner explicitly requested configuration/argument integration and a
five-epoch CNN run. Added `--architecture {mlp,cnn}` with MLP as the default;
saved the resolved classes, layers, and counts in new configs. Model code
remains learner-authored. All 25 tests passed, zero skips, in 0.527 seconds;
focused flake8 and three-file mypy checks passed before execution.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-cnn-run-20260916 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/run_baseline.py --architecture cnn --device cpu --epochs 5 --batch-size 128 --seed 42 --noise-dim 64 --lr 0.0002 --threads 2 --num-workers 0 --out-dir phase01_research_engineering/week08_gans/results/runs/session4_mnist_cnn_seed42_20260916
```

Exit 0. Five epochs processed 55,000 images/430 batches each; 2,150 paired
updates. Training took 262.733 seconds; the recorded training/artifact/audit
phase took 269.021 seconds. Final D/G losses were 0.201683/9.202752.
All five checkpoints were saved under the external data root. No test images
were used for measured optimization, scoring, or selection.

The [artifact audit](results/runs/session4_mnist_cnn_seed42_20260916/audit_and_compare.py)
passed source/checkpoint hashes, all 18 matched configuration fields, counts,
CSV/JSON consistency, and exact replay of two CNN panels plus the original
MLP final panel. The [comparison figure](results/runs/session4_mnist_cnn_seed42_20260916/mlp_vs_cnn_epoch05.png)
was visually inspected: cleaner CNN backgrounds and strokes coexist with
malformed/incomplete digits. Saved losses are diagnostics, not a quality rank.
The run changes both architecture and capacity and does not establish
independent-seed superiority. Full records are in [notes.md](results/runs/session4_mnist_cnn_seed42_20260916/notes.md)
and [verification.txt](results/runs/session4_mnist_cnn_seed42_20260916/verification.txt).

## Twenty-Epoch CNN Extension And MLP Reuse - 2026-09-16

At the learner's request, ran the same CNN command with `--epochs 20` and a
fresh `--out-dir` ending in `session4_mnist_cnn_seed42_20epochs_20260916`.
The prior twenty-epoch MLP result was reused after verifying all twenty
checkpoint hashes and exact sample replay. No MLP training was rerun today.
No core source or hyperparameter changes were made for the longer CNN budget.

CNN epochs 1–5 exactly reproduced the saved baseline's non-runtime metrics,
grids, models, optimizers, and both RNG states. The initial process terminated
with exit 143 after epoch 12; no training process remained. The saved checkpoint
was verified, and the run-local `continue_run.py` restored all required states
before finishing epochs 13–20 with exit 0. Its synthetic resumed-versus-
uninterrupted check matched exactly. [continuation.json](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/continuation.json)
records the helper hash, command, and recovery provenance.

Final CNN D/G losses: 0.729749/1.967853; 8,600 paired updates. Completed-epoch
training time: 1,098.076 seconds; continuation active time: 436.542 seconds;
wall elapsed including recovery: 1,504.219 seconds. All twenty CNN checkpoints
and twenty MLP checkpoints verified; both models' epoch-5/10/20 and independent
final panels reproduce exactly. The full [comparison](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_epoch20.png),
[loss plot](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/mlp_vs_cnn_losses_20epochs.png),
and [run notes](results/runs/session4_mnist_cnn_seed42_20epochs_20260916/notes.md)
record cleaner and more varied CNN samples alongside remaining ambiguity and
one-seed/capacity limits. No further training is pending for this request.
