# Day 3 - Margins, Trees, Ensembles, and LDA

## Status and reproduction

Day 3 was implemented and verified on 2026-07-15 using deterministic generated data and pure-PyTorch model logic. Run it from the repository root with:

```bash
.venv/bin/python basic_ml_mini_projects/day03_margins_trees.py
.venv/bin/python -m unittest basic_ml_mini_projects.test_day03_margins_trees -v
```

The script writes one six-row metrics table to `basic_ml_mini_projects/results/day03_metrics.csv`. It deliberately creates no new figure because Days 1 and 2 already use the bridge's four-plot budget. All ten deterministic Day 3 tests pass.

## Scope and data

Day 3 implements five classifier families from scratch:

- a linear SVM trained with mini-batch SGD on hinge loss
- a shallow binary CART decision tree using Gini impurity
- a random forest of bootstrapped shallow CART trees with per-node feature sampling
- logistic gradient boosting with regression stumps fitted to pseudo-residuals
- linear discriminant analysis with a regularized pooled covariance

The margin check reuses Day 1's separable two-dimensional blobs with a deterministic 180/60/60 train/validation/test split. The tabular comparison uses 480 generated rows with six features: three linear signals of different strengths, a nonlinear interaction between features 3 and 4, and one noise feature. Its split is 288/96/96. Both datasets are standardized only from training statistics.

## Measured results

| Dataset | Model | Train accuracy | Validation accuracy | Test accuracy | Train-validation gap |
| --- | --- | ---: | ---: | ---: | ---: |
| Separable blobs | Linear SVM | 0.9944 | 1.0000 | 1.0000 | -0.0056 |
| Mixed-signal tabular | Linear SVM | 0.7639 | 0.7188 | 0.7917 | 0.0451 |
| Mixed-signal tabular | Decision tree | 0.7986 | 0.7604 | 0.8229 | 0.0382 |
| Mixed-signal tabular | Random forest | 0.8507 | 0.7813 | 0.8333 | 0.0694 |
| Mixed-signal tabular | Logistic gradient boosting | 0.7986 | 0.7604 | 0.8229 | 0.0382 |
| Mixed-signal tabular | LDA | 0.7743 | 0.7292 | 0.8125 | 0.0451 |

The SVM hinge objective fell from `1.0000` to `0.0526` on the separable blobs and from `1.0000` to `0.4853` on the mixed-signal data. Boosting accepted all 50 stumps and reduced binary cross-entropy from `0.6923` to `0.4764`; every accepted update was loss-decreasing.

Test-set confusion counts are:

| Model | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: |
| Blob linear SVM | 36 | 0 | 0 | 24 |
| Tabular linear SVM | 43 | 12 | 8 | 33 |
| Decision tree | 45 | 10 | 7 | 34 |
| Random forest | 47 | 8 | 8 | 33 |
| Logistic gradient boosting | 45 | 10 | 7 | 34 |
| LDA | 43 | 12 | 6 | 35 |

## What the comparison shows

- The linear SVM cleanly solves the deliberately separable margin check. Its slightly lower training accuracy than validation/test accuracy is ordinary split variation, not a negative overfitting gap to interpret causally.
- The mixed-signal target contains a nonlinear interaction. The linear SVM and LDA cannot express that interaction directly, but both remain useful baselines because the first three features carry linear signal.
- The random forest has the best held-out accuracy, `0.8333`. Its bootstrap aggregation and feature subsampling expose different trees to different views of the data, while its shallow depth keeps the train-validation gap to `0.0694`.
- The single tree and stump booster both reach `0.8229` test accuracy on this seed. The booster selected feature 0 for 38 stumps and feature 1 for 12, so this shallow additive form did not recover the feature 3-by-4 interaction.
- Raw split counts are diagnostic, not calibrated feature importance. The tree used counts `4|2|1|0|0|0`; the forest used `80|53|34|18|35|36`; boosting used `38|12|0|0|0|0`. A noise feature can receive forest splits because bootstrap samples and random feature subsets create locally useful but unstable partitions.

## Assumptions and failure cases

### Linear SVM

The SVM assumes a useful linear separating direction and optimizes margin violations rather than probability calibration. It can underfit nonlinear boundaries, is sensitive to feature scale and regularization, and its raw scores are not probabilities.

### Decision tree

CART assumes useful decisions can be represented by axis-aligned thresholds. A shallow tree can underfit interactions; a deep tree can memorize small samples. Greedy split selection does not guarantee the globally best tree, and small data changes can alter early splits.

### Random forest

The forest reduces the variance of individual trees through bootstrapping and feature subsampling. It is less interpretable than one tree, costs more prediction work, and raw split frequency can overstate noisy or high-opportunity features.

### Logistic gradient boosting

The implementation adds regression stumps greedily to reduce logistic loss. Stumps create an additive model of one-feature effects, so they need multiple coordinated stages to approximate interactions and may never represent them efficiently. Learning rate, estimator count, leaf size, and noisy residual chasing all affect overfitting.

### LDA

LDA assumes each class is approximately Gaussian with a shared covariance and therefore produces a linear boundary. Correlated features are allowed, unlike Gaussian Naive Bayes, but small or collinear samples can make the pooled covariance singular. Diagonal regularization and Cholesky solves keep this educational implementation finite without forming a raw inverse.

## Verified acceptance criteria

- [x] SVM hinge loss decreases and the separable fixture is classified exactly.
- [x] CART uses deterministic scalar thresholds, leaf-size guards, depth limits, and tie-breaking.
- [x] The forest preserves exact seeded bootstrap samples and samples candidate features independently at each node.
- [x] Boosting records only loss-decreasing stump updates.
- [x] LDA uses a regularized pooled covariance and Cholesky solves.
- [x] Train, validation, test, confusion, loss, and split-count evidence are written deterministically.
- [x] Ten deterministic unit/integration tests pass.

## Artifact

- `results/day03_metrics.csv`
