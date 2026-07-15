# Day 2 - Clustering and EM

## Status and reproduction

Day 2 was implemented and verified on 2026-07-15 using deterministic generated data and pure-PyTorch model logic. Run it from the repository root with:

```bash
.venv/bin/python basic_ml_mini_projects/day02_clustering_em.py
.venv/bin/python -m unittest basic_ml_mini_projects.test_day02_clustering_em -v
```

The script writes an eleven-row metrics table and one six-panel summary figure under `basic_ml_mini_projects/results/`. All nine deterministic Day 2 tests pass.

## Scope

Day 2 compares three kinds of unsupervised assignment:

- k-means uses hard nearest-centroid assignments
- DBSCAN expands density-connected neighborhoods and can label noise
- a diagonal Gaussian mixture uses expectation-maximization and soft responsibilities

The implementations use PyTorch tensors and standard Python for model logic. No scikit-learn algorithms are used. PCA was optional and unnecessary because every generated dataset is already two-dimensional.

## Generated datasets

- **Isotropic blobs:** 240 points from three spherical clusters, used for the k-means model-size sweep.
- **Concentric rings:** two noisy rings with 120 points each plus 12 isolated points, used to compare shape assumptions and DBSCAN sensitivity.
- **Elongated blobs:** 240 points from three axis-aligned Gaussian clusters with unequal spreads, used to compare k-means with a diagonal GMM.

All generators use fixed seeds and are covered by reproducibility tests.

## K-means results

| k | Final inertia | Silhouette | Updates |
| ---: | ---: | ---: | ---: |
| 2 | 1166.9319 | 0.5863 | 6 |
| 3 | 133.9730 | 0.8140 | 2 |
| 4 | 116.6636 | 0.6529 | 4 |
| 5 | 96.8800 | 0.5058 | 8 |

Inertia decreases whenever more centroids are added, so inertia alone would keep favoring larger `k`. The silhouette score peaks at `k=3`, which matches the three generating blobs. This illustrates why an elbow-style fit measure and a separation measure should be interpreted together.

K-means assumes that Euclidean distance to a centroid represents cluster membership. It works best for compact, roughly spherical groups with comparable scale. On the concentric rings, `k=2` achieved silhouette `0.3263` but divided both rings into radial wedges rather than recovering the two ring-shaped groups.

## DBSCAN results

The sensitivity sweep kept `min_samples=5` fixed:

| eps | Clusters | Noise points | Interpretation |
| ---: | ---: | ---: | --- |
| 0.08 | 0 | 252 | Every point is treated as noise. |
| 0.20 | 2 | 127 | Two dense regions appear, but much of the outer ring is still noise. |
| 0.30 | 2 | 12 | Both rings are recovered and the 12 isolated points remain noise. |
| 1.40 | 1 | 0 | The neighborhoods are broad enough to merge everything. |

DBSCAN helps more than k-means when cluster identity follows connected density rather than distance to a center. The `eps=0.30` result follows both non-spherical rings and rejects the intentionally isolated points. The sweep also shows its main weakness: small parameter changes can move the result from all noise, through fragmented coverage, to one merged cluster.

## Gaussian-mixture EM results

On the elongated three-cluster dataset:

| Model | Inertia | Silhouette | Initial log likelihood | Final log likelihood | Updates |
| --- | ---: | ---: | ---: | ---: | ---: |
| k-means, k=3 | 183.3686 | 0.6733 | - | - | 4 |
| Diagonal GMM, 3 components | - | 0.6733 | -626.6619 | -626.6369 | 4 |

The GMM component weights were `0.3343`, `0.3332`, and `0.3325`; the mean maximum responsibility was `0.9952`. The hard GMM labels match the same strong three-group structure found by k-means on this easy deterministic dataset, while the responsibilities retain a probability for every component instead of discarding alternatives.

EM alternates between estimating soft component responsibilities (the E-step) and updating weights, means, and diagonal variances (the M-step). Each accepted update improved total log likelihood, which is the objective EM optimizes. The result is still local rather than globally guaranteed, and diagonal covariance assumes feature independence within each component. Variance floors, positive weight floors, log-space probabilities, and collapsed-component recovery keep the educational implementation numerically finite.

## Failure cases and limits

- K-means is sensitive to scale, initialization, outliers, and non-spherical clusters.
- Silhouette scores can favor separated compact groups and are not a universal choice of `k`.
- DBSCAN depends strongly on `eps` and `min_samples`; one global density threshold may fail when clusters have different densities.
- The `O(n^2)` distance matrices used by DBSCAN and silhouette scoring are intentionally educational and do not scale to large datasets.
- GMM can converge to a local optimum or a collapsed component. The diagonal model cannot represent rotated covariance directly.
- The generated datasets make the expected behavior visible; these numbers are learning evidence, not benchmark claims.

## Verified acceptance criteria

- [x] K-means assignments and centroids update iteratively with non-increasing tested inertia.
- [x] Inertia and silhouette are recorded for `k=2,3,4,5`.
- [x] DBSCAN finds both dense rings and labels the isolated points as noise.
- [x] Four `eps` values demonstrate parameter sensitivity.
- [x] GMM likelihood improves across accepted EM updates and responsibilities sum to one.
- [x] One figure visibly contrasts k-means, DBSCAN, and soft GMM assignment.
- [x] Nine deterministic unit/integration tests pass.

## Artifacts

- `results/day02_metrics.csv`
- `results/day02_clustering_em_summary.png`
