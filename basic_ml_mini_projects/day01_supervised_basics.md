# Day 1 - Supervised Basics

## Status

Complete and verified on 2026-07-13. The implementation is in `day01_supervised_basic.py`; ridge uses the closed-form solution only. Seven deterministic tests pass, and the runnable demo writes a metrics CSV and three plots under `results/`.

## Goal

Implement the supervised-learning foundation in pure PyTorch: linear models, nearest-neighbor classification, and one probabilistic classifier.

## Algorithms

1. Linear regression with gradient descent.
2. Linear regression with closed-form normal equation.
3. Ridge regression.
4. Logistic regression, binary first and multiclass if time allows.
5. k-nearest neighbors classifier.
6. One probabilistic model: Gaussian Naive Bayes or Gaussian discriminant analysis.

## Suggested Files

```text
basic_ml_mini_projects/
  day01_supervised_basic.py
  README.md
```

## Data

Use generated toy data only:

- one small 1D or 2D regression dataset with noise
- one small 2D classification dataset with two or three classes

Keep a fixed seed for reproducibility.

## Metrics

- regression: MSE, MAE, residual summary
- classification: accuracy, log loss where applicable, confusion matrix

## Acceptance Criteria

- [x] Each algorithm trains or predicts without external ML libraries.
- [x] Linear regression gradient descent and closed-form regression give similar results on the same toy data.
- [x] Logistic regression improves loss over training.
- [x] k-nearest neighbors predicts from distances computed in PyTorch.
- [x] The probabilistic classifier produces sensible class probabilities or class decisions.

## Notes To Write

- What assumptions do linear regression and logistic regression make?
- When does k-nearest neighbors become sensitive to feature scaling?
- How does a probabilistic classifier differ from a discriminative classifier?

These questions are answered in `day01_results.md`.
