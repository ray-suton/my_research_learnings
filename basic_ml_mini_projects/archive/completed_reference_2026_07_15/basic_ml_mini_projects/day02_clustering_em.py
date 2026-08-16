"""Day 2 clustering algorithms implemented with PyTorch tensors.

The implementations are intentionally small and explicit: k-means exposes
its assignment inertia after every centroid update, DBSCAN uses one full
pairwise-distance matrix, and the diagonal Gaussian mixture records total log
likelihood after every expectation-maximization update.  Running this file
reproduces the deterministic Day 2 metrics table and six-panel summary plot.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "ml-roadmap-matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


SEED = 42
CSV_FIELDS = [
    "section",
    "dataset",
    "model",
    "setting",
    "n_clusters",
    "inertia",
    "silhouette",
    "noise_points",
    "initial_log_likelihood",
    "final_log_likelihood",
    "mean_max_responsibility",
    "component_weights",
    "iterations",
]


@dataclass(frozen=True)
class KMeansResult:
    """Final k-means state and assignment inertia at every state boundary."""

    centroids: torch.Tensor
    labels: torch.Tensor
    inertia_history: list[float]


@dataclass(frozen=True)
class DBSCANResult:
    """Density-based labels, core-point membership, and summary counts."""

    labels: torch.Tensor
    core_sample_mask: torch.Tensor
    n_clusters: int
    n_noise: int


@dataclass(frozen=True)
class GaussianMixtureResult:
    """Parameters and state history for a diagonal Gaussian mixture."""

    weights: torch.Tensor
    means: torch.Tensor
    variances: torch.Tensor
    responsibilities: torch.Tensor
    log_likelihood_history: list[float]


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_float(value: float, name: str) -> None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite non-negative number") from error
    if not math.isfinite(numeric_value) or numeric_value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _validated_features(x: torch.Tensor, name: str = "x") -> torch.Tensor:
    """Validate a sample-by-feature tensor and make real integers usable."""

    if not isinstance(x, torch.Tensor):
        raise ValueError(f"{name} must be a PyTorch tensor")
    if x.ndim != 2:
        raise ValueError(f"{name} must be a 2D sample-by-feature tensor")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must contain samples and features")
    if x.is_complex() or x.dtype == torch.bool:
        raise ValueError(f"{name} must contain real numeric values")
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must contain only finite values")
    if not x.is_floating_point():
        return x.to(dtype=torch.get_default_dtype())
    return x


def _cluster_counts(n_samples: int) -> list[int]:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValueError("n_samples must be an integer")
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3")
    quotient, remainder = divmod(n_samples, 3)
    return [quotient + int(index < remainder) for index in range(3)]


def generate_isotropic_blobs(
    n_samples: int = 240,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate three shuffled spherical clusters with equal feature spread."""

    counts = _cluster_counts(n_samples)
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [[-3.0, -2.0], [0.0, 2.5], [3.0, -1.5]],
        dtype=torch.float32,
    )
    feature_blocks = []
    label_blocks = []
    for cluster_index, count in enumerate(counts):
        noise = 0.55 * torch.randn((count, 2), generator=generator)
        feature_blocks.append(centers[cluster_index] + noise)
        label_blocks.append(
            torch.full((count,), cluster_index, dtype=torch.int64)
        )

    features = torch.cat(feature_blocks, dim=0)
    labels = torch.cat(label_blocks, dim=0)
    permutation = torch.randperm(n_samples, generator=generator)
    return features[permutation], labels[permutation]


def generate_elongated_blobs(
    n_samples: int = 240,
    seed: int = 52,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate three shuffled axis-aligned clusters with unequal spreads."""

    counts = _cluster_counts(n_samples)
    generator = torch.Generator().manual_seed(seed)
    centers = torch.tensor(
        [[-2.5, -1.0], [0.0, 1.5], [2.5, -0.5]],
        dtype=torch.float32,
    )
    standard_deviations = torch.tensor(
        [[0.8, 0.3], [0.35, 0.9], [0.7, 0.4]],
        dtype=torch.float32,
    )
    feature_blocks = []
    label_blocks = []
    for cluster_index, count in enumerate(counts):
        noise = torch.randn((count, 2), generator=generator)
        feature_blocks.append(
            centers[cluster_index] + noise * standard_deviations[cluster_index]
        )
        label_blocks.append(
            torch.full((count,), cluster_index, dtype=torch.int64)
        )

    features = torch.cat(feature_blocks, dim=0)
    labels = torch.cat(label_blocks, dim=0)
    permutation = torch.randperm(n_samples, generator=generator)
    return features[permutation], labels[permutation]


def generate_concentric_rings(
    points_per_ring: int = 120,
    n_noise: int = 12,
    seed: int = 62,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two noisy rings and evenly spaced isolated outer points."""

    _validate_positive_integer(points_per_ring, "points_per_ring")
    if isinstance(n_noise, bool) or not isinstance(n_noise, int) or n_noise < 0:
        raise ValueError("n_noise must be a non-negative integer")

    generator = torch.Generator().manual_seed(seed)
    base_angles = (
        torch.arange(points_per_ring, dtype=torch.float32)
        * (2.0 * math.pi / points_per_ring)
    )
    feature_blocks = []
    label_blocks = []
    for ring_index, radius in enumerate((1.0, 2.3)):
        angles = base_angles + 0.01 * torch.randn(
            points_per_ring, generator=generator
        )
        noisy_radii = radius + 0.03 * torch.randn(
            points_per_ring, generator=generator
        )
        points = torch.stack(
            (noisy_radii * angles.cos(), noisy_radii * angles.sin()), dim=1
        )
        feature_blocks.append(points)
        label_blocks.append(
            torch.full((points_per_ring,), ring_index, dtype=torch.int64)
        )

    if n_noise:
        noise_angles = (
            torch.arange(n_noise, dtype=torch.float32)
            * (2.0 * math.pi / n_noise)
        )
        isolated_points = 3.6 * torch.stack(
            (noise_angles.cos(), noise_angles.sin()), dim=1
        )
        feature_blocks.append(isolated_points)
        label_blocks.append(torch.full((n_noise,), -1, dtype=torch.int64))

    features = torch.cat(feature_blocks, dim=0)
    labels = torch.cat(label_blocks, dim=0)
    permutation = torch.randperm(features.shape[0], generator=generator)
    return features[permutation], labels[permutation]


def _squared_distances(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    return (x[:, None, :] - centroids[None, :, :]).square().sum(dim=2)


def _assign_to_centroids(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    squared_distances = _squared_distances(x, centroids)
    nearest_squared_distances, labels = squared_distances.min(dim=1)
    return labels, nearest_squared_distances, nearest_squared_distances.sum().item()


@torch.no_grad()
def fit_kmeans(
    x: torch.Tensor,
    n_clusters: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    seed: int = 42,
) -> KMeansResult:
    """Fit k-means with seeded farthest-first centroid initialization."""

    features = _validated_features(x)
    _validate_positive_integer(n_clusters, "n_clusters")
    _validate_positive_integer(max_iterations, "max_iterations")
    _validate_nonnegative_float(tolerance, "tolerance")
    if n_clusters > features.shape[0]:
        raise ValueError("n_clusters cannot exceed the number of samples")

    generator = torch.Generator(device=features.device).manual_seed(seed)
    first_index = torch.randperm(
        features.shape[0], generator=generator, device=features.device
    )[0]
    chosen_indices = [int(first_index.item())]
    centroids = features[first_index].reshape(1, -1).clone()

    while centroids.shape[0] < n_clusters:
        nearest_squared_distances = _squared_distances(
            features, centroids
        ).min(dim=1).values
        nearest_squared_distances[chosen_indices] = -1.0
        next_index = int(nearest_squared_distances.argmax().item())
        chosen_indices.append(next_index)
        centroids = torch.cat(
            (centroids, features[next_index].reshape(1, -1)), dim=0
        )

    labels, nearest_squared_distances, initial_inertia = _assign_to_centroids(
        features, centroids
    )
    inertia_history = [initial_inertia]

    for _ in range(max_iterations):
        updated_centroids = centroids.clone()
        empty_cluster_indices = []
        for cluster_index in range(n_clusters):
            members = features[labels == cluster_index]
            if members.shape[0]:
                updated_centroids[cluster_index] = members.mean(dim=0)
            else:
                empty_cluster_indices.append(cluster_index)

        if empty_cluster_indices:
            candidate_indices = torch.argsort(
                nearest_squared_distances, descending=True
            ).tolist()
            used_candidates: set[int] = set()
            for cluster_index in empty_cluster_indices:
                candidate_index = next(
                    index
                    for index in candidate_indices
                    if index not in used_candidates
                )
                used_candidates.add(candidate_index)
                updated_centroids[cluster_index] = features[candidate_index]

        maximum_shift = torch.linalg.vector_norm(
            updated_centroids - centroids, dim=1
        ).max().item()
        centroids = updated_centroids
        labels, nearest_squared_distances, inertia = _assign_to_centroids(
            features, centroids
        )
        inertia_history.append(inertia)
        if maximum_shift <= float(tolerance):
            break

    return KMeansResult(
        centroids=centroids,
        labels=labels.to(torch.int64),
        inertia_history=inertia_history,
    )


@torch.no_grad()
def predict_kmeans(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """Assign each sample to its nearest fitted centroid."""

    features = _validated_features(x)
    centers = _validated_features(centroids, "centroids")
    if features.shape[1] != centers.shape[1]:
        raise ValueError("x and centroids must have equal feature dimensions")
    return _squared_distances(features, centers).argmin(dim=1).to(torch.int64)


@torch.no_grad()
def silhouette_score(x: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the mean sample silhouette using one pairwise distance matrix."""

    features = _validated_features(x)
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a one-dimensional PyTorch tensor")
    if labels.shape[0] != features.shape[0]:
        raise ValueError("x and labels must have equal sample counts")
    if labels.is_complex() or not bool(torch.isfinite(labels).all()):
        raise ValueError("labels must contain finite real values")

    unique_labels = torch.unique(labels, sorted=True)
    if unique_labels.numel() < 2:
        raise ValueError("silhouette score requires at least two clusters")
    if unique_labels.numel() >= features.shape[0]:
        raise ValueError("silhouette score requires fewer clusters than samples")

    distances = torch.cdist(features, features)
    silhouettes = torch.zeros(
        features.shape[0], dtype=features.dtype, device=features.device
    )
    for sample_index in range(features.shape[0]):
        own_label = labels[sample_index]
        own_mask = labels == own_label
        own_count = int(own_mask.sum().item())
        if own_count == 1:
            continue

        intra_cluster_distance = (
            distances[sample_index, own_mask].sum() / (own_count - 1)
        )
        nearest_other_distance = None
        for other_label in unique_labels:
            if bool(other_label == own_label):
                continue
            candidate_distance = distances[
                sample_index, labels == other_label
            ].mean()
            if (
                nearest_other_distance is None
                or candidate_distance < nearest_other_distance
            ):
                nearest_other_distance = candidate_distance

        if nearest_other_distance is None:
            raise RuntimeError("unable to find another cluster")
        denominator = torch.maximum(
            intra_cluster_distance, nearest_other_distance
        )
        if denominator > 0:
            silhouettes[sample_index] = (
                nearest_other_distance - intra_cluster_distance
            ) / denominator

    return silhouettes.mean().clamp(-1.0, 1.0).item()


@torch.no_grad()
def fit_dbscan(
    x: torch.Tensor,
    eps: float,
    min_samples: int,
) -> DBSCANResult:
    """Fit educational DBSCAN with deterministic density expansion."""

    features = _validated_features(x)
    try:
        numeric_eps = float(eps)
    except (TypeError, ValueError) as error:
        raise ValueError("eps must be a finite positive number") from error
    if not math.isfinite(numeric_eps) or numeric_eps <= 0.0:
        raise ValueError("eps must be a finite positive number")
    _validate_positive_integer(min_samples, "min_samples")

    distances = torch.cdist(features, features)
    neighborhood_mask = distances <= numeric_eps
    core_sample_mask = neighborhood_mask.sum(dim=1) >= min_samples
    labels = torch.full(
        (features.shape[0],), -1, dtype=torch.int64, device=features.device
    )
    visited = torch.zeros(
        features.shape[0], dtype=torch.bool, device=features.device
    )
    next_cluster_label = 0

    for sample_index in range(features.shape[0]):
        if bool(visited[sample_index]):
            continue
        visited[sample_index] = True
        if not bool(core_sample_mask[sample_index]):
            continue

        labels[sample_index] = next_cluster_label
        initial_neighbors = torch.nonzero(
            neighborhood_mask[sample_index], as_tuple=False
        ).reshape(-1).tolist()
        queue = deque(initial_neighbors)
        queued = set(initial_neighbors)

        while queue:
            neighbor_index = queue.popleft()
            if not bool(visited[neighbor_index]):
                visited[neighbor_index] = True
                if bool(core_sample_mask[neighbor_index]):
                    expanded_neighbors = torch.nonzero(
                        neighborhood_mask[neighbor_index], as_tuple=False
                    ).reshape(-1).tolist()
                    for expanded_index in expanded_neighbors:
                        if expanded_index not in queued:
                            queue.append(expanded_index)
                            queued.add(expanded_index)

            if int(labels[neighbor_index].item()) == -1:
                labels[neighbor_index] = next_cluster_label

        next_cluster_label += 1

    return DBSCANResult(
        labels=labels,
        core_sample_mask=core_sample_mask,
        n_clusters=next_cluster_label,
        n_noise=int((labels == -1).sum().item()),
    )


def _validate_gmm_parameters(
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
) -> None:
    if not all(
        isinstance(value, torch.Tensor)
        for value in (weights, means, variances)
    ):
        raise ValueError("GMM parameters must be PyTorch tensors")
    if weights.ndim != 1 or means.ndim != 2 or variances.ndim != 2:
        raise ValueError("GMM parameters have invalid dimensions")
    if means.shape != variances.shape or weights.shape[0] != means.shape[0]:
        raise ValueError("GMM parameter shapes do not agree")
    if weights.numel() == 0 or means.shape[1] == 0:
        raise ValueError("GMM parameters cannot be empty")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (weights, means, variances)
    ):
        raise ValueError("GMM parameters must be finite")
    if bool((weights <= 0.0).any()) or bool((variances <= 0.0).any()):
        raise ValueError("GMM weights and variances must be positive")


def _expectation_step(
    x: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    differences = x[:, None, :] - means[None, :, :]
    log_component_densities = -0.5 * (
        math.log(2.0 * math.pi)
        + torch.log(variances)[None, :, :]
        + differences.square() / variances[None, :, :]
    ).sum(dim=2)
    weighted_log_probabilities = (
        torch.log(weights)[None, :] + log_component_densities
    )
    sample_log_likelihoods = torch.logsumexp(
        weighted_log_probabilities, dim=1
    )
    responsibilities = torch.exp(
        weighted_log_probabilities - sample_log_likelihoods[:, None]
    )
    return responsibilities, sample_log_likelihoods.sum().item()


@torch.no_grad()
def fit_gaussian_mixture(
    x: torch.Tensor,
    n_components: int,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    variance_floor: float = 1e-6,
    seed: int = 42,
) -> GaussianMixtureResult:
    """Fit a diagonal Gaussian mixture with full EM updates in log space."""

    features = _validated_features(x)
    _validate_positive_integer(n_components, "n_components")
    _validate_positive_integer(max_iterations, "max_iterations")
    _validate_nonnegative_float(tolerance, "tolerance")
    try:
        numeric_variance_floor = float(variance_floor)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "variance_floor must be a finite positive number"
        ) from error
    if (
        not math.isfinite(numeric_variance_floor)
        or numeric_variance_floor <= 0.0
    ):
        raise ValueError("variance_floor must be a finite positive number")
    if n_components > features.shape[0]:
        raise ValueError("n_components cannot exceed the number of samples")

    kmeans_result = fit_kmeans(
        features,
        n_clusters=n_components,
        max_iterations=max_iterations,
        tolerance=tolerance,
        seed=seed,
    )
    means = kmeans_result.centroids.clone()
    global_variance = features.var(dim=0, unbiased=False).clamp_min(
        numeric_variance_floor
    )
    weights = torch.empty(
        n_components, dtype=features.dtype, device=features.device
    )
    variances = torch.empty_like(means)
    for component_index in range(n_components):
        component_samples = features[
            kmeans_result.labels == component_index
        ]
        if component_samples.shape[0]:
            weights[component_index] = (
                component_samples.shape[0] / features.shape[0]
            )
            variances[component_index] = component_samples.var(
                dim=0, unbiased=False
            ).clamp_min(numeric_variance_floor)
        else:
            weights[component_index] = 1.0 / features.shape[0]
            variances[component_index] = global_variance
    weights = weights.clamp_min(1e-8)
    weights = weights / weights.sum()

    responsibilities, initial_log_likelihood = _expectation_step(
        features, weights, means, variances
    )
    log_likelihood_history = [initial_log_likelihood]

    for _ in range(max_iterations):
        effective_counts = responsibilities.sum(dim=0)
        safe_counts = effective_counts.clamp_min(1e-12)
        candidate_means = (
            responsibilities.T @ features
        ) / safe_counts[:, None]
        differences = features[:, None, :] - candidate_means[None, :, :]
        candidate_variances = (
            responsibilities[:, :, None] * differences.square()
        ).sum(dim=0) / safe_counts[:, None]
        candidate_variances = candidate_variances.clamp_min(
            numeric_variance_floor
        )
        candidate_weights = effective_counts / features.shape[0]

        collapsed_indices = torch.nonzero(
            effective_counts < 1.0, as_tuple=False
        ).reshape(-1).tolist()
        if collapsed_indices:
            explanation_confidence = responsibilities.max(dim=1).values
            reinitialization_order = torch.argsort(
                explanation_confidence, descending=False
            ).tolist()
            used_sample_indices: set[int] = set()
            for component_index in collapsed_indices:
                sample_index = next(
                    index
                    for index in reinitialization_order
                    if index not in used_sample_indices
                )
                used_sample_indices.add(sample_index)
                candidate_means[component_index] = features[sample_index]
                candidate_variances[component_index] = global_variance
                candidate_weights[component_index] = 1.0 / features.shape[0]

        candidate_weights = candidate_weights.clamp_min(1e-8)
        candidate_weights = candidate_weights / candidate_weights.sum()
        candidate_responsibilities, candidate_log_likelihood = (
            _expectation_step(
                features,
                candidate_weights,
                candidate_means,
                candidate_variances,
            )
        )
        improvement = candidate_log_likelihood - log_likelihood_history[-1]
        if improvement < -1e-6:
            break

        weights = candidate_weights
        means = candidate_means
        variances = candidate_variances
        responsibilities = candidate_responsibilities
        log_likelihood_history.append(candidate_log_likelihood)
        if abs(improvement) <= float(tolerance):
            break

    responsibilities, _ = _expectation_step(
        features, weights, means, variances
    )
    return GaussianMixtureResult(
        weights=weights,
        means=means,
        variances=variances,
        responsibilities=responsibilities,
        log_likelihood_history=log_likelihood_history,
    )


@torch.no_grad()
def predict_gaussian_mixture_probabilities(
    x: torch.Tensor,
    model: GaussianMixtureResult,
) -> torch.Tensor:
    """Return normalized posterior component probabilities for each sample."""

    features = _validated_features(x)
    if not isinstance(model, GaussianMixtureResult):
        raise ValueError("model must be a GaussianMixtureResult")
    _validate_gmm_parameters(model.weights, model.means, model.variances)
    if features.shape[1] != model.means.shape[1]:
        raise ValueError("x and model means must have equal feature dimensions")
    responsibilities, _ = _expectation_step(
        features, model.weights, model.means, model.variances
    )
    return responsibilities


@torch.no_grad()
def predict_gaussian_mixture_labels(
    x: torch.Tensor,
    model: GaussianMixtureResult,
) -> torch.Tensor:
    """Return the maximum-posterior component label for each sample."""

    return predict_gaussian_mixture_probabilities(x, model).argmax(dim=1)


def _format_csv_value(value: str | int | float) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return format(value, ".10g")
    if isinstance(value, int):
        return str(value)
    return value


def _write_metrics_table(
    rows: list[dict[str, str | int | float]],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _format_csv_value(row.get(field, ""))
                    for field in CSV_FIELDS
                }
            )


def _cluster_colors(labels: torch.Tensor) -> list[tuple[float, float, float, float]]:
    color_map = plt.get_cmap("tab10")
    colors = []
    for label in labels.detach().cpu().tolist():
        if label == -1:
            colors.append((0.15, 0.15, 0.15, 0.8))
        else:
            colors.append(color_map(int(label) % 10))
    return colors


def _save_summary_plot(
    k_values: list[int],
    kmeans_sweep: list[KMeansResult],
    kmeans_silhouettes: list[float],
    ring_features: torch.Tensor,
    ring_kmeans: KMeansResult,
    ring_dbscan: DBSCANResult,
    eps_values: list[float],
    dbscan_results: list[DBSCANResult],
    elongated_features: torch.Tensor,
    gmm_model: GaussianMixtureResult,
    output_path: Path,
) -> None:
    """Render the complete Day 2 comparison in one six-panel figure."""

    figure, axes = plt.subplots(3, 2, figsize=(14, 16))

    inertia_axis = axes[0, 0]
    silhouette_axis = inertia_axis.twinx()
    inertia_axis.plot(
        k_values,
        [result.inertia_history[-1] for result in kmeans_sweep],
        marker="o",
        color="tab:blue",
        label="inertia",
    )
    silhouette_axis.plot(
        k_values,
        kmeans_silhouettes,
        marker="s",
        color="tab:orange",
        label="silhouette",
    )
    inertia_axis.set_title("K-means model-size sweep")
    inertia_axis.set_xlabel("number of clusters (k)")
    inertia_axis.set_ylabel("inertia", color="tab:blue")
    silhouette_axis.set_ylabel("silhouette", color="tab:orange")
    inertia_axis.set_xticks(k_values)
    inertia_axis.grid(alpha=0.25)
    lines = inertia_axis.lines + silhouette_axis.lines
    inertia_axis.legend(lines, [line.get_label() for line in lines])

    ring_points = ring_features.detach().cpu()
    axes[0, 1].scatter(
        ring_points[:, 0].tolist(),
        ring_points[:, 1].tolist(),
        c=_cluster_colors(ring_kmeans.labels),
        s=20,
    )
    axes[0, 1].set_title("K-means forces radial rings into wedges")
    axes[0, 1].set_xlabel("feature 1")
    axes[0, 1].set_ylabel("feature 2")
    axes[0, 1].set_aspect("equal", adjustable="box")

    axes[1, 0].scatter(
        ring_points[:, 0].tolist(),
        ring_points[:, 1].tolist(),
        c=_cluster_colors(ring_dbscan.labels),
        s=20,
    )
    axes[1, 0].set_title(
        f"DBSCAN follows density ({ring_dbscan.n_noise} noise points)"
    )
    axes[1, 0].set_xlabel("feature 1")
    axes[1, 0].set_ylabel("feature 2")
    axes[1, 0].set_aspect("equal", adjustable="box")

    cluster_axis = axes[1, 1]
    noise_axis = cluster_axis.twinx()
    cluster_axis.plot(
        eps_values,
        [result.n_clusters for result in dbscan_results],
        marker="o",
        color="tab:green",
        label="clusters",
    )
    noise_axis.plot(
        eps_values,
        [result.n_noise for result in dbscan_results],
        marker="s",
        color="tab:red",
        label="noise points",
    )
    cluster_axis.set_title("DBSCAN parameter sensitivity")
    cluster_axis.set_xlabel("eps")
    cluster_axis.set_ylabel("clusters", color="tab:green")
    noise_axis.set_ylabel("noise points", color="tab:red")
    cluster_axis.set_xticks(eps_values)
    cluster_axis.grid(alpha=0.25)
    lines = cluster_axis.lines + noise_axis.lines
    cluster_axis.legend(lines, [line.get_label() for line in lines])

    gmm_labels = gmm_model.responsibilities.argmax(dim=1)
    confidence = gmm_model.responsibilities.max(dim=1).values
    color_map = plt.get_cmap("tab10")
    confidence_colors = []
    for label, opacity in zip(
        gmm_labels.detach().cpu().tolist(),
        confidence.detach().cpu().tolist(),
    ):
        base_color = color_map(int(label) % 10)
        confidence_colors.append(
            (base_color[0], base_color[1], base_color[2], float(opacity))
        )
    elongated_points = elongated_features.detach().cpu()
    axes[2, 0].scatter(
        elongated_points[:, 0].tolist(),
        elongated_points[:, 1].tolist(),
        c=confidence_colors,
        s=24,
    )
    axes[2, 0].scatter(
        gmm_model.means[:, 0].detach().cpu().tolist(),
        gmm_model.means[:, 1].detach().cpu().tolist(),
        marker="X",
        c="black",
        s=110,
        label="component means",
    )
    axes[2, 0].set_title("GMM labels; opacity is assignment confidence")
    axes[2, 0].set_xlabel("feature 1")
    axes[2, 0].set_ylabel("feature 2")
    axes[2, 0].legend()

    axes[2, 1].plot(
        range(len(gmm_model.log_likelihood_history)),
        gmm_model.log_likelihood_history,
        marker="o",
        markersize=3,
        color="tab:purple",
    )
    axes[2, 1].set_title("Gaussian-mixture EM progress")
    axes[2, 1].set_xlabel("completed EM updates")
    axes[2, 1].set_ylabel("total log likelihood")
    axes[2, 1].ticklabel_format(
        axis="y", style="plain", useOffset=False
    )
    axes[2, 1].grid(alpha=0.25)

    figure.suptitle("Day 2: hard, density-based, and soft clustering", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _base_metric_row(
    section: str,
    dataset: str,
    model: str,
    setting: str,
) -> dict[str, str | int | float]:
    return {
        field: "" for field in CSV_FIELDS
    } | {
        "section": section,
        "dataset": dataset,
        "model": model,
        "setting": setting,
    }


def run_day02_demo(
    output_dir: Path | None = None,
) -> list[dict[str, str | int | float]]:
    """Run the frozen Day 2 experiments and save their two artifacts."""

    if output_dir is None:
        output_dir = Path(__file__).with_name("results")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    isotropic_features, _ = generate_isotropic_blobs()
    k_values = [2, 3, 4, 5]
    kmeans_sweep = [
        fit_kmeans(isotropic_features, n_clusters=k, seed=42)
        for k in k_values
    ]
    kmeans_silhouettes = [
        silhouette_score(isotropic_features, result.labels)
        for result in kmeans_sweep
    ]

    rows: list[dict[str, str | int | float]] = []
    for k, result, score in zip(
        k_values, kmeans_sweep, kmeans_silhouettes
    ):
        row = _base_metric_row(
            "kmeans_sweep", "isotropic_blobs", "k-means", f"k={k}"
        )
        row.update(
            {
                "n_clusters": int(torch.unique(result.labels).numel()),
                "inertia": result.inertia_history[-1],
                "silhouette": score,
                "iterations": len(result.inertia_history) - 1,
            }
        )
        rows.append(row)

    ring_features, _ = generate_concentric_rings()
    eps_values = [0.08, 0.20, 0.30, 1.40]
    dbscan_results = [
        fit_dbscan(ring_features, eps=eps, min_samples=5)
        for eps in eps_values
    ]
    for eps, result in zip(eps_values, dbscan_results):
        row = _base_metric_row(
            "dbscan_sensitivity",
            "concentric_rings",
            "DBSCAN",
            f"eps={eps:.2f};min_samples=5",
        )
        row.update(
            {
                "n_clusters": result.n_clusters,
                "noise_points": result.n_noise,
            }
        )
        rows.append(row)

    ring_kmeans = fit_kmeans(ring_features, n_clusters=2, seed=42)
    ring_kmeans_row = _base_metric_row(
        "shape_comparison", "concentric_rings", "k-means", "k=2"
    )
    ring_kmeans_row.update(
        {
            "n_clusters": int(torch.unique(ring_kmeans.labels).numel()),
            "inertia": ring_kmeans.inertia_history[-1],
            "silhouette": silhouette_score(
                ring_features, ring_kmeans.labels
            ),
            "iterations": len(ring_kmeans.inertia_history) - 1,
        }
    )
    rows.append(ring_kmeans_row)

    elongated_features, _ = generate_elongated_blobs()
    elongated_kmeans = fit_kmeans(
        elongated_features, n_clusters=3, seed=42
    )
    elongated_kmeans_row = _base_metric_row(
        "gmm_comparison", "elongated_blobs", "k-means", "k=3"
    )
    elongated_kmeans_row.update(
        {
            "n_clusters": int(torch.unique(elongated_kmeans.labels).numel()),
            "inertia": elongated_kmeans.inertia_history[-1],
            "silhouette": silhouette_score(
                elongated_features, elongated_kmeans.labels
            ),
            "iterations": len(elongated_kmeans.inertia_history) - 1,
        }
    )
    rows.append(elongated_kmeans_row)

    gmm_model = fit_gaussian_mixture(
        elongated_features, n_components=3, seed=42
    )
    gmm_labels = predict_gaussian_mixture_labels(
        elongated_features, gmm_model
    )
    gmm_row = _base_metric_row(
        "gmm_comparison",
        "elongated_blobs",
        "diagonal GMM",
        "n_components=3",
    )
    gmm_row.update(
        {
            "n_clusters": int(torch.unique(gmm_labels).numel()),
            "silhouette": silhouette_score(elongated_features, gmm_labels),
            "initial_log_likelihood": gmm_model.log_likelihood_history[0],
            "final_log_likelihood": gmm_model.log_likelihood_history[-1],
            "mean_max_responsibility": gmm_model.responsibilities.max(
                dim=1
            ).values.mean().item(),
            "component_weights": "|".join(
                format(weight, ".10g")
                for weight in gmm_model.weights.detach().cpu().tolist()
            ),
            "iterations": len(gmm_model.log_likelihood_history) - 1,
        }
    )
    rows.append(gmm_row)

    _write_metrics_table(rows, output_dir / "day02_metrics.csv")
    _save_summary_plot(
        k_values=k_values,
        kmeans_sweep=kmeans_sweep,
        kmeans_silhouettes=kmeans_silhouettes,
        ring_features=ring_features,
        ring_kmeans=ring_kmeans,
        ring_dbscan=dbscan_results[2],
        eps_values=eps_values,
        dbscan_results=dbscan_results,
        elongated_features=elongated_features,
        gmm_model=gmm_model,
        output_path=output_dir / "day02_clustering_em_summary.png",
    )

    print(f"Saved Day 2 metrics and plot to: {output_dir}")
    return rows


if __name__ == "__main__":
    run_day02_demo()
