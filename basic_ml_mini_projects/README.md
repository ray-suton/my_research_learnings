# Basic ML Learning Workspace

This directory contains the learner-owned Basic ML rebuild and the archived
reference version of the earlier completed bridge.

The previous completed bridge is preserved under
`archive/completed_reference_2026_07_15/`. Treat it as a read-only answer key:
attempt each concept and implementation first, then use the archive only for
comparison or verification.

## Current status

The rebuild is closed for now so the roadmap can move forward to Phase 1 Week
3 CNN work. The current learner-owned scripts compile and cover the main bridge
topics:

- `linear_reg.py`: generated regression data, gradient-descent linear
  regression, and ordinary least squares
- `log_reg.py`, `knn.py`, `gnb.py`: basic binary/classical classifiers
- `kmeans.py`, `dbscan.py`, `gmm.py`: clustering and mixture-model drafts
- `svm.py`, `dec_tree.py`, `rand_forest.py`, `gbt.py`: margins, trees, and
  shallow ensemble drafts

Ridge regression and regularized LDA are intentionally deferred. The archived
reference still contains complete versions of those ideas if they become useful
later.

See `BASIC_ML_REBUILD_REPORT.md` for the consolidated status, verification
evidence, and handoff notes.

## Learning sequence covered

1. Data foundations: problem definition, generated data, dataset splits,
   preprocessing, leakage checks, metrics, and reproducibility.
2. Regression: gradient-descent linear regression and closed-form least
   squares. Ridge regression is deferred.
3. Classification: logistic regression, k-nearest neighbors, and Gaussian
   Naive Bayes.
4. Unsupervised learning: k-means, DBSCAN, and Gaussian-mixture EM.
5. Margins and trees: linear SVM, CART, random forest, and gradient boosting.
6. Discriminant methods: regularized LDA is deferred.

## Working agreement

- You write the learning implementation.
- Codex teaches with one focused question or review step at a time.
- Use PyTorch tensors and standard Python for algorithm logic.
- Add tests as each behavioral contract becomes clear.
- Keep generated artifacts small and reproducible.
- Do not consult the archived implementation until after making and testing
  your own attempt.

## Next checkpoint

Return to Phase 1 Week 3 in
`../phase01_research_engineering/week03_cnns_cifar10/`.
