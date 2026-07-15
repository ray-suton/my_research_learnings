# Day 1 Supervised Basics - Results and Comparison

## Status and Reproduction

Day 1 was implemented and verified on 2026-07-13 using deterministic generated data. Run it from the repository root with:

```bash
.venv/bin/python basic_ml_mini_projects/day01_supervised_basic.py
.venv/bin/python -m unittest basic_ml_mini_projects.test_day01_supervised_basic -v
```

The verification run passed seven tests and produced the following held-out metrics:

| Task | Model | MSE | MAE | Accuracy | Log loss |
| --- | --- | ---: | ---: | ---: | ---: |
| Regression | Linear regression, gradient descent | 0.0629 | 0.1930 | - | - |
| Regression | Ordinary least squares | 0.0628 | 0.1928 | - | - |
| Regression | Ridge, lambda = 5 | 0.0680 | 0.2042 | - | - |
| Classification | Logistic regression | - | - | 0.9867 | 0.0460 |
| Classification | k-nearest neighbors, k = 5 | - | - | 1.0000 | - |
| Classification | Gaussian Naive Bayes | - | - | 1.0000 | 0.0144 |

These are educational toy-data results, not benchmark claims. The classification blobs are deliberately easy to separate.

## What the Results Show

- Gradient descent and ordinary least squares learned nearly identical linear-regression weights, which checks the manual optimization against the analytic solution.
- Ridge slightly increased held-out error on this small, low-noise dataset while reducing the weight norm. That is expected when ordinary least squares is already stable: regularization trades some fit for shrinkage rather than guaranteeing better error on every dataset.
- Logistic-regression binary cross-entropy fell from 0.6931 to 0.0469 during training, and the final held-out accuracy was 98.67%.
- kNN and Gaussian Naive Bayes both classified all held-out examples correctly on this seed. That reflects the simple Gaussian blobs and should not be interpreted as universal superiority.

## Model Assumptions

### Linear and Ridge Regression

The conditional mean of the target is assumed to be a linear combination of the features. Independent, roughly constant-variance errors make the usual statistical interpretation stronger, although prediction itself does not require normally distributed errors. Ridge adds the preference that smaller weights are safer; it is particularly useful when features are correlated or ordinary least squares has high variance.

### Logistic Regression

Logistic regression assumes that the log-odds of the positive class are linear in the features. This creates a linear decision boundary in the original feature space. It models the class probability directly and is therefore a discriminative model.

### k-Nearest Neighbors

kNN assumes nearby points should usually have the same label. Distances determine the entire prediction, so feature scale matters: a feature measured in thousands can dominate one measured between zero and one. The implementation standardizes using training-set statistics to prevent that problem and avoid test-data leakage.

### Gaussian Naive Bayes

Gaussian Naive Bayes assumes each feature is Gaussian within each class and that features are conditionally independent after the class is known. The independence assumption is often imperfect, but the model can still work well with little data. Variance smoothing prevents division by zero for nearly constant features.

## Discriminative Versus Probabilistic/Generative Classification

Logistic regression learns the boundary or conditional probability `P(y | x)` directly. Gaussian Naive Bayes learns class priors `P(y)` and feature likelihoods `P(x | y)`, then applies Bayes' rule to obtain `P(y | x)`. The latter can generate a probabilistic explanation of how each class could have produced the features; the former focuses on separating the classes.

kNN is different from both: it stores the training examples and predicts from local distances instead of learning a fixed parametric probability model.

## Failure Cases and Limits

- Linear and logistic regression underfit curved relationships unless nonlinear features are added.
- The normal equation can fail or become numerically unstable for singular or highly collinear design matrices; `torch.linalg.lstsq` or a pseudoinverse is more robust for production use.
- Ridge requires choosing its penalty on validation data. A large penalty can overshrink useful weights.
- Logistic regression can struggle with nonlinear class boundaries and extreme class imbalance.
- kNN is sensitive to feature scaling, irrelevant dimensions, the choice of `k`, and the cost of comparing every test point with all training points.
- Gaussian Naive Bayes can become poorly calibrated when within-class features are strongly dependent or non-Gaussian.

## Artifacts

- `results/day01_metrics.csv`
- `results/day01_regression_predictions.png`
- `results/day01_logistic_loss.png`
- `results/day01_classification_predictions.png`
