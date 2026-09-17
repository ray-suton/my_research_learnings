# Tests

The Fashion-MNIST extension preflight on 2026-09-17 passed **34 tests with zero
skips**: the 25 model/architecture checks below, five dataset-selection checks,
and four run-control checks. The run-control checks include both architectures'
exact uninterrupted-versus-restored trajectories, immutable config and deferred
finalization, Fashion class labels, and rejection of inconsistent resume state.

Run the full suite from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-fashion-tests-20260917 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -m unittest discover -s phase01_research_engineering/week08_gans/tests -p 'test_*.py' -v
```

`test_dataset_selection.py` verifies actual MNIST/FashionMNIST class selection,
cached splits, normalization, independent RNG streams, and download prohibition.
`test_run_control.py` exercises a two-invocation synthetic lifecycle and both
model pairs' exact optimizer/RNG restoration. Tiny integration fixtures and
checkpoints live in automatically removed temporary directories.

## Retained MNIST Verification

The Session 3 MLP baseline and Session 4 CNN pair are verified by
[test_session3_baseline.py](test_session3_baseline.py), a stdlib unittest suite
that also runs directly. **All 25 tests passed with zero skips on 2026-09-16**:
eleven behavioral checks for each model pair and three architecture-selection
checks. These include default MLP initialization preservation, explicit CNN
selection with a custom noise width, and rejection of an unknown architecture.
The original Session 3 run retains its historical eleven-test verification.
The [Session 4 verification record](../results/runs/session4_mnist_cnn_seed42_20260916/verification.txt)
retains the 25-test output, static checks, and final artifact/replay audit.

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/week08-mpl-tests OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 -u phase01_research_engineering/week08_gans/tests/test_session3_baseline.py
```

Coverage includes actual optimizer-boundary parameter/gradient checks, loss
targets and values, single/regular batches, two consecutive updates, a short
final batch, weighted epoch summaries, empty-loader handling, D flag
restoration including exceptions, monitoring mode/RNG preservation, and
cached split/normalization/loader-RNG checks. Test optimizers use small test
settings; measured-run settings are recorded separately in config.json.

[Session 3 retained output](../results/runs/session3_mnist_mlp_seed42_20260915/verification.txt) includes its final command,
checks, type-check result, and artifact audit. The run's source snapshots
include this test file. The suite never downloads data or runs the measured
experiment. Missing cached MNIST is an explicit skipped prerequisite, not a
passing data check; the recorded baseline validation had no skips.

The following contract remains useful when changing the implementation.
Normal teaching is learner-owned coding; this operational verification suite
was added for the user's explicit request to complete the baseline.

## Synthetic CPU Checks Before Training

- Generator maps `(B, 64)` noise to finite `(B, 1, 28, 28)` images in `[-1, 1]`.
- Discriminator accepts real and generated image layouts and produces finite
  sigmoid probabilities in `[0, 1]` of shape `(B, 1)`, including `B=1` and a final short batch.
- Floating targets match prediction shape/device; real targets are 1, fake targets
  are 0, and generator targets are 1.
- Plain BCE consumes probabilities directly. Increasing a real-image probability
  reduces its loss; increasing a fake-image probability increases its loss. At probability 0.5 each
  mean BCE term equals `log(2)`, and the summed discriminator loss equals
  `2 * log(2)`.
- After cleared gradients, a discriminator-only update changes discriminator
  parameters while leaving generator parameters and gradients unchanged.
- A generator-only update produces nonzero generator gradients and changes its
  parameters while preserving discriminator parameters; frozen discriminator
  parameters must not block gradients through its input.
- Two consecutive complete iterations work without reusing a freed/stale
  graph; both losses and gradients remain finite.
- Monitoring fixed noise preserves parameters, model modes, and the training
  random stream. Equal fixed inputs reproduce equal outputs in evaluation.
- Epoch summaries weight unequal batch sizes correctly.

Use deterministic tiny synthetic batches. Tests must not download datasets,
submit HPC jobs, overwrite results, or launch measured training.

## Cached-Data Checks Before the Measured Run

- Loader honors `AI_ROADMAP_DATA` and the documented fallback.
- Cache loading with downloads disabled returns MNIST image batches in the
  declared range; digit labels do not enter either objective.
- Seeded train/validation subsets contain 55,000/5,000 disjoint examples and
  the official test split remains unused by training updates.

Record missing data as a prerequisite failure, not a model test pass. Add and
record runnable commands only after the corresponding test files exist.
