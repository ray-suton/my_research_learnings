# Basic ML Mini-Projects

This bridge comes after the Week 2 MLP experiment and before Week 3 CNNs. The goal is to revisit core non-neural machine-learning algorithms by implementing small educational versions in pure PyTorch.

## Current Status

All three days are implemented and verified. The bridge is complete and ready to hand off into Week 3 CNN work.

Day 1 includes:

- linear regression with manual gradient descent
- ordinary least squares with the normal equation
- closed-form ridge regression, with no ridge gradient-descent variant
- binary logistic regression with logits-based binary cross-entropy
- k-nearest neighbors using `torch.cdist`
- Gaussian Naive Bayes using log probabilities and variance smoothing
- MSE, MAE, residual summaries, accuracy, log loss, and confusion matrices
- seven deterministic standard-library unit tests
- one CSV metrics table, three plots, and a short comparison note

Day 2 includes:

- k-means with deterministic initialization, empty-cluster recovery, inertia, and silhouette scoring
- an educational `O(n^2)` DBSCAN implementation with core, border, and noise handling
- a diagonal Gaussian mixture trained with expectation-maximization and soft responsibilities
- deterministic isotropic blobs, elongated blobs, and concentric-ring datasets
- a four-setting DBSCAN sensitivity comparison
- nine deterministic standard-library unit/integration tests
- one eleven-row CSV metrics table and one six-panel summary plot

Day 3 includes:

- a linear SVM trained with deterministic mini-batch SGD and hinge loss
- a shallow binary CART decision tree with cumulative Gini split scans
- a random forest with preserved bootstrap samples and per-node feature sampling
- logistic gradient boosting with loss-decreasing regression-stump updates
- regularized LDA using pooled covariance and Cholesky solves
- train/validation/test accuracy, confusion counts, overfitting gaps, losses, and split counts
- ten deterministic standard-library unit/integration tests
- one six-row CSV metrics table and the final three-day comparison note

Scope rules:

- Use PyTorch tensors and standard Python only for model logic.
- Do not use scikit-learn or other ML libraries for algorithm implementations.
- Keep datasets tiny and generated in code.
- Prefer clarity over library-grade completeness.
- Save a small metrics table and 2-4 plots across the whole bridge.

## Three-Day Plan

- Day 1: supervised foundations with regression, logistic regression, k-nearest neighbors, and one probabilistic classifier. **Complete.**
- Day 2: clustering and latent-variable methods with k-means, DBSCAN, and Gaussian mixtures. **Complete.**
- Day 3: margins, trees, shallow ensembles, LDA, and the final comparison note. **Complete.**

## Run and verify all three days

From the repository root:

```bash
.venv/bin/python basic_ml_mini_projects/day01_supervised_basic.py
.venv/bin/python -m unittest basic_ml_mini_projects.test_day01_supervised_basic -v
.venv/bin/python basic_ml_mini_projects/day02_clustering_em.py
.venv/bin/python -m unittest basic_ml_mini_projects.test_day02_clustering_em -v
.venv/bin/python basic_ml_mini_projects/day03_margins_trees.py
.venv/bin/python -m unittest basic_ml_mini_projects.test_day03_margins_trees -v
```

The scripts write three CSV metrics tables and four PNG figures under `basic_ml_mini_projects/results/`. See `day01_results.md`, `day02_clustering_em.md`, and `day03_margins_trees_summary.md` for measured results, interpretation, assumptions, and failure cases.

## Deliverable

The final deliverable is complete in `comparison_note.md`. It explains model assumptions, metrics, failure cases, and what these classical methods teach before moving to CNNs.
