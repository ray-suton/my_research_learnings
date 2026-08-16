# Basic ML Rebuild Consolidated Report

Date: 2026-08-16

## Purpose

This report replaces the scattered Basic ML session reports as the durable
handoff for the learner-owned Basic ML rebuild. It records what exists, what
has been checked, what is intentionally deferred, and why the roadmap is moving
forward to Phase 1 Week 3 instead of continuing to expand this bridge.

The goal of the bridge was to revisit classical machine-learning ideas with
small pure-PyTorch implementations before returning to neural-network projects.
That goal is sufficiently served for the current learning path. The next active
area is CNNs and mini-ResNets on CIFAR-10.

## Directory Roles

- `basic_ml_mini_projects/*.py`: learner-owned draft implementations.
- `basic_ml_mini_projects/archive/completed_reference_2026_07_15/`: read-only
  reference copy of the earlier completed bridge.
- `basic_ml_mini_projects/archive/README.md`: archive contract and verification
  command.
- `basic_ml_mini_projects/BASIC_ML_REBUILD_REPORT.md`: this consolidated
  status file.

## Reference Archive Status

The archived bridge remains the strongest verified Basic ML artifact in this
repo. It includes the prior source, tests, notes, metrics, plots, and final
comparison from the completed bridge.

Verified on 2026-08-16:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=basic_ml_mini_projects/archive/completed_reference_2026_07_15 \
./.venv/bin/python -m unittest \
  basic_ml_mini_projects.test_day01_supervised_basic \
  basic_ml_mini_projects.test_day02_clustering_em \
  basic_ml_mini_projects.test_day03_margins_trees -v
```

Result:

```text
Ran 26 tests in 8.043s
OK
```

The archive covers:

- linear regression with gradient descent
- ordinary least squares
- closed-form ridge regression
- logistic regression
- k-nearest neighbors
- Gaussian Naive Bayes
- k-means
- DBSCAN
- diagonal Gaussian-mixture EM
- linear SVM
- CART decision tree
- random forest
- gradient boosting
- regularized LDA
- metrics tables, plots, and a final assumptions/failure-case comparison

## Learner-Owned Draft Inventory

The current root-level learner scripts compile. They are educational drafts,
not library-quality implementations and not yet backed by formal tests.

Verified on 2026-08-16:

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python -c "...compile all basic_ml_mini_projects/*.py..."
```

Result:

```text
compiled 11 learner scripts
```

Current draft files:

| File | Role | Status |
| --- | --- | --- |
| `linear_reg.py` | generated regression data, split helper, gradient-descent linear regression, ordinary least squares | compiles; ordinary least squares had a prior name bug that was later fixed and smoke-checked |
| `log_reg.py` | generated binary data, sigmoid/logistic helpers, BCE loss, gradient-descent training loop | compiles; no formal test |
| `knn.py` | generated classification data, split helper, pairwise distances, kNN prediction | compiles; no formal test |
| `gnb.py` | generated Gaussian class data, split helper, Gaussian Naive Bayes statistics and prediction | compiles; no formal test |
| `kmeans.py` | generated cluster data, centroid initialization/update, k-means training loop | compiles; no formal test |
| `dbscan.py` | generated ring/outlier data, core-neighbor helpers, DBSCAN draft | compiles; no formal test |
| `gmm.py` | generated mixture data, initialization, diagonal Gaussian density, E-step, M-step, EM loop | compiles; previous M-step issue was rechecked as fixed in session evidence |
| `svm.py` | generated separable data, hinge loss, linear SVM objective, SGD trainer | compiles; smoke-checked with train/test accuracy on easy generated data |
| `dec_tree.py` | generated toy data, Gini split helpers, recursive CART-style tree | compiles; no formal test |
| `rand_forest.py` | random-split tree wrappers and forest prediction helpers | compiles; no formal test |
| `gbt.py` | regression tree helpers and gradient-boosting training/prediction drafts | compiles; no formal test |

## Session Evidence Consolidated

Recent session reports before cleanup recorded the following important points:

- The original completed bridge was archived as a dated reference.
- The learner-owned rebuild produced drafts for most planned algorithms.
- `gmm.py` previously had an incomplete M-step, then a later recheck showed the
  EM loop running end to end on a generated smoke case.
- `linear_reg.py` previously had an ordinary-least-squares name mismatch, then
  a later recheck showed noiseless 1D/2D/3D recovery working.
- `svm.py` was replaced with a linear SVM draft and smoke-checked on generated
  linearly separable data.
- The latest algorithm-gap report identified ridge regression and regularized
  LDA as the only clear remaining planned algorithms.

## Intentional Deferrals

Ridge regression and regularized LDA are not needed before Week 3.

Rationale:

- The archived reference already contains verified versions.
- The learner rebuild has covered the main algorithm families needed for the
  bridge: regression, classification, clustering, mixture models, margins,
  trees, and shallow ensembles.
- Continuing to chase every classical-model variant would delay the Phase 1
  neural-network sequence without adding enough near-term learning value.

Deferred, not forgotten:

- Add ridge regression only if regularization comparisons become relevant in a
  later tabular or optimization lesson.
- Add regularized LDA only if discriminant analysis becomes relevant in a
  later probabilistic-model or calibration lesson.

## Known Limitations

- The learner-owned scripts do not yet have formal unit tests.
- There is no learner-owned metrics table or plot set outside the archive.
- Failure-case and model-assumption comparisons are documented in the archive,
  not reauthored for the learner drafts.
- Some drafts are intentionally compact and may omit validation, multiclass
  support, numerical-stability checks, or production-grade error handling.
- The root-level learner scripts should not be treated as a stable API.

## Closure Decision

Close the Basic ML rebuild for now.

This is a learning-roadmap decision, not a claim that the learner scripts are
complete production implementations. The archive remains available as the
complete verified reference. The root-level drafts are sufficient as learning
exposure before returning to neural networks.

## Next Active Project

Proceed to:

```text
phase01_research_engineering/week03_cnns_cifar10/
```

Week 3 focus:

- prepare or verify CIFAR-10 under `AI_ROADMAP_DATA`
- implement a small CNN
- add augmentation, weight decay, learning-rate schedule, and confusion matrix
- implement a mini-ResNet-style model
- compare CNN vs mini-ResNet with metrics and limitations

## If This Rebuild Is Resumed

Use this order:

1. Add formal tests for the existing learner-owned scripts.
2. Fix behavior found by tests before adding algorithms.
3. Add ridge regression only after linear regression tests are stable.
4. Add regularized LDA only after Gaussian Naive Bayes and classification
   metric tests are stable.
5. Write a short comparison note from learner-owned results, not from the
   archive.
