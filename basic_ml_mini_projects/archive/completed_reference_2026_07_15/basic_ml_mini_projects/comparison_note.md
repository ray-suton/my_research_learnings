# Basic ML Bridge - Final Comparison

## Scope

This three-day bridge implements classical machine-learning ideas with PyTorch tensors and standard Python on small generated datasets. It is a learning comparison, not a benchmark: results across regression, classification, and clustering use different datasets and metrics and should not be ranked as if they came from one competition.

## Evidence across the three days

| Day | Question | Most useful measured evidence |
| --- | --- | --- |
| 1 | How do linear, local, and generative supervised methods differ? | Ordinary least squares reached test MSE `0.0628`; logistic regression reached accuracy `0.9867`; kNN and Gaussian Naive Bayes each reached `1.0000` on deliberately easy blobs. |
| 2 | How do centroid, density, and mixture assumptions change clustering? | K-means silhouette peaked at `0.8140` for the true `k=3`; DBSCAN at `eps=0.30` recovered two rings and left 12 isolated points as noise; diagonal-GMM log likelihood improved from `-626.6619` to `-626.6369`. |
| 3 | What do margins, recursive partitions, ensembles, boosting, and shared-covariance discrimination add? | Linear SVM reached `1.0000` on separable blobs; on mixed-signal data, random forest led at `0.8333`, tree and boosting reached `0.8229`, LDA `0.8125`, and linear SVM `0.7917`. |

The exact measurements are in `results/day01_metrics.csv`, `results/day02_metrics.csv`, and `results/day03_metrics.csv`. Dataset-specific interpretation is in `day01_results.md`, `day02_clustering_em.md`, and `day03_margins_trees_summary.md`.

## Which assumptions mattered

| Family | Main assumption or bias | Where it helped | Typical failure |
| --- | --- | --- | --- |
| Linear/ridge regression | The conditional mean is linear; ridge prefers smaller weights. | The generated regression target is close to linear, so gradient descent and ordinary least squares agree. | Curved relationships underfit; collinearity destabilizes unregularized closed forms; too much ridge overshrinks. |
| Logistic regression and linear SVM | A linear score separates the classes. | Both perform strongly when blob classes are separated by a line. | Neither directly captures the mixed dataset's feature interaction; SVM scores are not calibrated probabilities. |
| kNN | Nearby standardized points share labels. | The easy blobs have compact local neighborhoods. | Scaling, irrelevant dimensions, choice of `k`, storage, and query cost all matter. |
| Gaussian Naive Bayes | Features are conditionally independent Gaussians within each class. | The blob generator is close to that story and little data is needed. | Dependent or non-Gaussian features can produce poor likelihoods and calibration. |
| LDA | Classes are Gaussian with one shared full covariance. | It models correlations that Naive Bayes discards and remains competitive on the linear signal. | A nonlinear boundary underfits; a small or collinear sample makes covariance estimation fragile. |
| K-means | Clusters are compact around Euclidean centroids. | It finds three separated isotropic blobs and gives a clear silhouette peak. | It divides concentric rings into radial regions and inertia always improves as `k` grows. |
| DBSCAN | Clusters are density-connected under one neighborhood scale. | It follows both non-spherical rings and labels isolated points as noise. | The `eps` sweep moves from all noise to useful clusters to one merged cluster; varying densities are difficult. |
| Gaussian mixture | Data arise from a finite mixture with the chosen covariance form. | Soft responsibilities preserve uncertainty and EM improves likelihood. | EM is local; components can collapse; diagonal covariance misses rotated correlations. |
| Decision tree | Axis-aligned greedy thresholds form a useful partition. | It captures nonlinear regions without manually adding polynomial features. | Deep trees have high variance; shallow trees underfit; greedy early splits are unstable. |
| Random forest | Averaging randomized trees reduces variance. | It gives the best mixed-signal test accuracy and exposes multiple features. | It costs more, is less transparent, and raw split counts can make noise look important. |
| Stump boosting | Many small additive corrections reduce logistic loss. | All 50 accepted stages lower loss and outperform the linear SVM on the mixed data. | One-feature stumps represent interactions inefficiently and can chase residual noise. |

## Metrics must match the question

- Regression uses MSE, MAE, and residual summaries because distance from a continuous target matters.
- Supervised classification uses held-out accuracy, log loss where probabilities exist, confusion matrices, and train-validation gaps. Accuracy alone would hide false-positive versus false-negative behavior and calibration.
- K-means uses inertia for compactness and silhouette for separation. Inertia cannot choose `k` alone because it decreases as centroids are added.
- DBSCAN needs cluster and noise counts because it deliberately leaves some observations unassigned.
- Mixture EM tracks log likelihood and responsibility confidence because soft assignments are part of the model, not an implementation detail.
- Optimization traces are safeguards as well as metrics: gradient losses should fall, k-means inertia should not increase, and accepted EM/boosting steps should improve their objectives.

## What was sensitive or failed

- Ridge increased regression error from the ordinary-least-squares MSE of `0.0628` to `0.0680` on this low-noise seed. Regularization is a tradeoff, not an automatic improvement.
- K-means could not recover ring topology even though it returned a numerical silhouette score.
- DBSCAN was the clearest parameter-sensitivity example: `eps=0.08` marked all 252 points as noise, `0.30` recovered two rings plus 12 noise points, and `1.40` merged everything.
- The diagonal GMM and EM safeguards work on the chosen data, but they do not remove local-optimum or covariance-shape limitations.
- On the mixed-signal data, the SVM and LDA linear boundaries miss the designed interaction. The stump booster used only features 0 and 1, showing that lower training loss does not prove the intended mechanism was learned.
- Forest split counts included 36 uses of the pure-noise feature. Randomization can improve predictions while making naive feature-count stories unreliable.

## Lessons before CNNs

1. **Inductive bias decides what is easy to learn.** Linear scores, local distances, density connectivity, Gaussian likelihoods, and axis-aligned partitions each make different structure simple.
2. **Use held-out evidence and prevent leakage.** Every supervised split is standardized from training data only; validation guides comparison and test data remain a final check.
3. **The training objective and evaluation metric are related but not interchangeable.** Lower hinge loss, cross-entropy, inertia, or negative log likelihood does not automatically imply the best task-level behavior.
4. **Optimization needs invariants.** Deterministic seeds, finite checks, monotonic accepted updates, variance floors, and Cholesky solves turn numerical behavior into evidence that can be tested.
5. **Ensembles trade simplicity for stability.** A forest is harder to inspect than one tree but reduces variance; boosting turns many weak rules into a stronger additive model.
6. **Interpretability claims need restraint.** Coefficients, centroids, responsibilities, and split counts describe the fitted model, but none automatically establish causal importance.

These same habits transfer directly to CNN work: define the data contract, choose an architecture whose bias matches the problem, separate optimization diagnostics from held-out metrics, and keep every comparison reproducible.
