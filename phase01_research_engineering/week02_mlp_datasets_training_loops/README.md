# Week 2 - MLPs, Datasets, and Training Loops

## Purpose

This project is the first full supervised-learning experiment in the roadmap. The goal is to train and compare small MLP classifiers on Fashion-MNIST while practicing clean dataset handling, train/validation/test separation, metric tracking, artifact saving, and short research notes.

## Current Status

- Dataset choice: Fashion-MNIST.
- Expected HPC data root: `/scratch/rui.gao/ai_research_roadmap_data`; it is not visible in the current 2026-07-13 session.
- Dataset preparation: user reported successful preparation on 2026-07-02, but the command output was not captured and neither the scratch root nor the fallback home root is currently present.
- Code status: training smoke path implemented and the three planned 5-epoch runs are recorded as completed.
- Implemented so far:
  - `dataset.py` loads Fashion-MNIST from the external data root and creates a 50,000-example train split, 10,000-example validation split, and official test loader.
  - `model.py` defines the initial one-hidden-layer MLP classifier.
  - `tests/test_smoke.py` checks one batch, forward pass, logits shape, and finite cross-entropy loss.
  - `train.py` supports full or batch-limited training, validation, and test evaluation; it prints metrics but does not persist structured metrics or checkpoints.
- Known gap:
  - `eval.py` is still empty; final test accuracy values are recorded in `RESULTS.md`, but the exact test-evaluation command/output was not captured.
  - raw variant stdout, structured per-run metrics, exact plot-generation commands, checkpoints, and populated per-run directories were not retained.
- Recorded runtime status: one tiny training smoke run completed, followed by the planned baseline, learning-rate variant, and hidden-size variant runs.
- Best recorded run: hidden-size variant with hidden size 256, learning rate 0.001, validation accuracy `0.8822`, and test accuracy `0.8734`.
- Evidence caveat: the baseline epoch stdout is recorded in `RESULTS.md`. Three loss plots exist, and the two variant plots preserve five-epoch curves and test loss, but raw variant stdout and exact plot commands were not captured.
- Recorded smoke commands that passed on 2026-07-06:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. pytest -q -p no:cacheprovider tests/test_smoke.py
PYTHONDONTWRITEBYTECODE=1 python train.py --epochs 1 --max-train-batches 2 --max-val-batches 2 --batch-size 64
```

These smoke checks passed from `phase01_research_engineering/week02_mlp_datasets_training_loops/` on 2026-07-06. Bare `pytest` without `PYTHONPATH=.` can fail to import local project modules in the current layout.

On 2026-07-13, fresh dataset-backed smoke rechecks against both the unset-variable fallback and the documented scratch root stopped during dataset construction with `RuntimeError: Dataset not found`. The test never reached model construction because the external Fashion-MNIST cache was not visible and downloads are disabled by default. A separate model-only forward recheck passed with finite logits of shape `(64, 10)`. This is a current environment/data-availability issue, not evidence of a model-code regression.

## Documentation Structure

```text
phase01_research_engineering/week02_mlp_datasets_training_loops/
  README.md              # project overview and status
  EXPERIMENT_PLAN.md     # hypothesis, baseline, metrics, splits, planned runs
  RUN_LOG.md             # recorded commands, outcomes, and evidence gaps
  RESULTS.md             # completed run comparison and evidence limitations
  results/
    README.md            # artifact policy for this project
    runs/
      README.md          # expected per-run layout
```

## Current Code Structure

```text
phase01_research_engineering/week02_mlp_datasets_training_loops/
  dataset.py             # dataset loading and split helpers
  model.py               # initial MLP model definition
  train.py               # full or batch-limited train/validation/test path; prints but does not persist metrics
  eval.py                # empty; no reusable checkpoint/test evaluation path yet
  tests/test_smoke.py    # one-batch dataloader/model/loss smoke test
```

## Data Policy

Fashion-MNIST data must stay outside Git under `AI_ROADMAP_DATA`.

Use this data root on the HPC:

```bash
export AI_ROADMAP_DATA="/scratch/$USER/ai_research_roadmap_data"
```

Do not commit raw data, processed caches, checkpoints, large logs, or downloaded dataset files.

The external Fashion-MNIST cache is not visible in the current 2026-07-13 session. Re-prepare it or point `AI_ROADMAP_DATA` to an existing cache only if a Week 2 rerun is needed; this does not block the generated-data basic-ML bridge.

## Done Criteria

Week 2 run-comparison work is complete for the planned three-run scope:

- one baseline MLP run completed
- one learning-rate comparison run completed
- one hidden-size comparison run completed
- final metrics are recorded in `RESULTS.md`
- baseline, learning-rate-variant, and hidden-size-variant loss plots are present

Recorded plot artifacts:

- `results/baseline_losses.png`
- `results/rl_variant_losses.png` (learning-rate variant; filename preserved as written)
- `results/hs_variant_losses.png` (hidden-size variant)

Remaining Week 2 evidence limitations:

- raw variant stdout and exact plot-generation commands were not retained
- structured per-run metrics and populated run directories were not retained
- checkpoint saving is not implemented and remains a recorded limitation

## Follow-Up: Basic ML Mini-Projects

After the first MLP project is complete, add a lightweight basic-ML bridge before moving on to heavier neural-network work. Keep this as one compact follow-up project, not a long detour.

Implement the bridge's model logic with PyTorch tensors and standard Python only; do not use scikit-learn or other ML libraries for algorithm implementations.

Suggested scope:

- linear regression
- logistic regression
- k-nearest neighbors
- k-means
- DBSCAN
- EM / Gaussian mixtures
- SVM
- decision tree, random forest, and boosted trees
- at least two probabilistic models, such as Naive Bayes, Gaussian discriminant analysis, LDA/QDA, or Bayesian ridge regression

Expected output: one short comparison note with metrics tables, a few plots, and the main model-assumption lessons.

## Next Step

Complete the compact pure-PyTorch basic-ML bridge, then move on to Week 3 CNN work. Do not tune the MLP further against the test split because test accuracies have already been recorded.
