"""Day 3: educational margin, tree, ensemble, and LDA classifiers.

The implementations deliberately favor explicit PyTorch tensor operations over
library-grade abstractions.  Datasets are generated locally, splits are
deterministic, and every reported metric comes from a held-out split.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__:
    from .day01_supervised_basic import (
        confusion_matrix,
        generate_binary_classification_data,
        standardize_from_train,
    )
else:
    from day01_supervised_basic import (  # type: ignore[import-not-found]
        confusion_matrix,
        generate_binary_classification_data,
        standardize_from_train,
    )


SEED = 42
EPSILON = 1e-8

CSV_FIELDS = (
    "section",
    "dataset",
    "model",
    "setting",
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "train_validation_gap",
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "initial_loss",
    "final_loss",
    "training_steps",
    "feature_split_counts",
)

_FLOAT_CSV_FIELDS = {
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "train_validation_gap",
    "initial_loss",
    "final_loss",
}
_INTEGER_CSV_FIELDS = {
    "test_tn",
    "test_fp",
    "test_fn",
    "test_tp",
    "training_steps",
}


@dataclass(frozen=True)
class ClassificationSplits:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_validation: torch.Tensor
    y_validation: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    feature_mean: torch.Tensor
    feature_std: torch.Tensor


@dataclass(frozen=True)
class LinearSVMResult:
    weights: torch.Tensor
    bias: torch.Tensor
    loss_history: tuple[float, ...]


@dataclass(frozen=True)
class TreeNode:
    prediction: int
    n_samples: int
    class_counts: tuple[int, int]
    impurity: float
    feature_index: int | None = None
    threshold: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None


@dataclass(frozen=True)
class DecisionTreeModel:
    root: TreeNode
    n_features: int
    split_counts: tuple[int, ...]
    max_depth: int


@dataclass(frozen=True)
class RandomForestModel:
    trees: tuple[DecisionTreeModel, ...]
    bootstrap_indices: tuple[torch.Tensor, ...]
    split_counts: tuple[int, ...]


@dataclass(frozen=True)
class RegressionStump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float
    step_size: float


@dataclass(frozen=True)
class GradientBoostingModel:
    base_logit: float
    stumps: tuple[RegressionStump, ...]
    loss_history: tuple[float, ...]
    split_counts: tuple[int, ...]


@dataclass(frozen=True)
class LDAModel:
    classes: torch.Tensor
    class_priors: torch.Tensor
    class_means: torch.Tensor
    covariance: torch.Tensor
    cholesky_factor: torch.Tensor
    regularization: float


def _validate_integer(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _validate_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    strictly_greater: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        invalid = converted <= minimum if strictly_greater else converted < minimum
        if invalid:
            relation = "greater than" if strictly_greater else "at least"
            raise ValueError(f"{name} must be {relation} {minimum}")
    return converted


def _validate_fraction(name: str, value: object) -> float:
    converted = _validate_real(name, value)
    if not 0.0 < converted < 1.0:
        raise ValueError(f"{name} must be strictly between zero and one")
    return converted


def _as_feature_tensor(x: object, *, name: str = "x") -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (n_samples, n_features)")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError(f"{name} must contain samples and features")
    if x.is_complex() or x.dtype == torch.bool:
        raise TypeError(f"{name} must contain real numeric values")
    converted = x.detach()
    if not converted.is_floating_point():
        converted = converted.to(torch.get_default_dtype())
    if not torch.isfinite(converted).all().item():
        raise ValueError(f"{name} must contain only finite values")
    return converted


def _as_binary_targets(
    y: object,
    n_samples: int,
    *,
    require_both_classes: bool,
) -> torch.Tensor:
    if not isinstance(y, torch.Tensor):
        raise TypeError("y must be a torch.Tensor")
    if y.ndim != 1:
        raise ValueError("y must have shape (n_samples,)")
    if y.shape[0] != n_samples:
        raise ValueError("x and y must contain the same number of samples")
    if y.is_complex():
        raise TypeError("y must contain real binary labels")
    values = y.detach()
    if values.is_floating_point() and not torch.isfinite(values).all().item():
        raise ValueError("y must contain only finite values")
    if not torch.all((values == 0) | (values == 1)).item():
        raise ValueError("y labels must be zero or one")
    labels = values.to(torch.long)
    if require_both_classes and torch.unique(labels).numel() != 2:
        raise ValueError("both binary classes must be present")
    return labels


def _validate_fit_data(
    x: object,
    y: object,
    *,
    require_both_classes: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = _as_feature_tensor(x)
    labels = _as_binary_targets(
        y,
        features.shape[0],
        require_both_classes=require_both_classes,
    )
    return features, labels


def _validate_prediction_features(
    x: object,
    n_features: int,
) -> torch.Tensor:
    features = _as_feature_tensor(x)
    if features.shape[1] != n_features:
        raise ValueError(
            f"x has {features.shape[1]} features; the model expects {n_features}"
        )
    return features


def _safe_midpoint(lower: float, upper: float) -> float:
    if lower < 0.0 < upper:
        midpoint = lower / 2.0 + upper / 2.0
    else:
        midpoint = lower + (upper - lower) / 2.0
    if not math.isfinite(midpoint):
        raise ValueError("a candidate threshold was not finite")
    if not lower <= midpoint < upper:
        midpoint = lower
    return midpoint


def _threshold_mask(values: torch.Tensor, threshold: float) -> torch.Tensor:
    """Compare without rounding a Python threshold back to a narrower dtype."""

    comparison_values = values.to(torch.float64)
    return comparison_values <= threshold


def generate_mixed_signal_data(
    n_samples: int = 480,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a six-feature binary dataset with linear and interaction signal."""

    n_samples = _validate_integer("n_samples", n_samples, minimum=1)
    seed = _validate_integer("seed", seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((n_samples, 6), generator=generator)
    noise = torch.randn(n_samples, generator=generator)
    interaction = torch.where(x[:, 3] * x[:, 4] >= 0, 1.0, -1.0)
    score = (
        1.50 * x[:, 0]
        + 0.75 * x[:, 1]
        - 0.25 * x[:, 2]
        + 1.25 * interaction
        + 0.35 * noise
    )
    y = (score >= 0).to(torch.long)
    return x.detach(), y.detach()


def _standardize_three_splits(
    raw_train: torch.Tensor,
    raw_validation: torch.Tensor,
    raw_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train, validation, mean, std = standardize_from_train(
        raw_train,
        raw_validation,
    )
    test = (raw_test - mean) / std
    outputs = (train, validation, test, mean, std)
    if not all(torch.isfinite(value).all().item() for value in outputs):
        raise RuntimeError("standardization produced non-finite values")
    return tuple(value.detach() for value in outputs)  # type: ignore[return-value]


def make_margin_splits(
    n_samples: int = 300,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.20,
    seed: int = SEED,
) -> ClassificationSplits:
    """Reuse Day 1 blobs and create deterministic train/validation/test splits."""

    n_samples = _validate_integer("n_samples", n_samples, minimum=8)
    validation_fraction = _validate_fraction(
        "validation_fraction", validation_fraction
    )
    test_fraction = _validate_fraction("test_fraction", test_fraction)
    seed = _validate_integer("seed", seed)
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one")

    test_count = max(1, math.floor(test_fraction * n_samples))
    validation_count = max(1, math.floor(validation_fraction * n_samples))
    train_count = n_samples - test_count - validation_count
    if train_count < 2:
        raise ValueError("the requested split must leave at least two training rows")

    pool_x, raw_test_x, pool_y, test_y = generate_binary_classification_data(
        n_samples=n_samples,
        test_fraction=test_fraction,
        seed=seed,
    )
    permutation = torch.randperm(
        pool_x.shape[0],
        generator=torch.Generator().manual_seed(seed + 1),
    )
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    raw_train_x = pool_x[train_indices]
    raw_validation_x = pool_x[validation_indices]
    train_x, validation_x, test_x, mean, std = _standardize_three_splits(
        raw_train_x,
        raw_validation_x,
        raw_test_x,
    )
    return ClassificationSplits(
        x_train=train_x,
        y_train=pool_y[train_indices].reshape(-1).to(torch.long).detach(),
        x_validation=validation_x,
        y_validation=pool_y[validation_indices]
        .reshape(-1)
        .to(torch.long)
        .detach(),
        x_test=test_x,
        y_test=test_y.reshape(-1).to(torch.long).detach(),
        feature_mean=mean,
        feature_std=std,
    )


def make_mixed_signal_splits(
    n_samples: int = 480,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    seed: int = SEED,
) -> ClassificationSplits:
    """Create deterministic splits of the mixed-signal tabular dataset."""

    n_samples = _validate_integer("n_samples", n_samples, minimum=1)
    train_fraction = _validate_fraction("train_fraction", train_fraction)
    validation_fraction = _validate_fraction(
        "validation_fraction", validation_fraction
    )
    seed = _validate_integer("seed", seed)
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below one")

    train_count = math.floor(train_fraction * n_samples)
    validation_count = math.floor(validation_fraction * n_samples)
    test_count = n_samples - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("train, validation, and test splits must all be nonempty")

    x, y = generate_mixed_signal_data(n_samples=n_samples, seed=seed)
    permutation = torch.randperm(
        n_samples,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    train_indices = permutation[:train_count]
    validation_indices = permutation[train_count : train_count + validation_count]
    test_indices = permutation[train_count + validation_count :]
    train_x, validation_x, test_x, mean, std = _standardize_three_splits(
        x[train_indices],
        x[validation_indices],
        x[test_indices],
    )
    return ClassificationSplits(
        x_train=train_x,
        y_train=y[train_indices].detach(),
        x_validation=validation_x,
        y_validation=y[validation_indices].detach(),
        x_test=test_x,
        y_test=y[test_indices].detach(),
        feature_mean=mean,
        feature_std=std,
    )


def _svm_objective(
    x: torch.Tensor,
    signed_y: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    l2_penalty: float,
) -> torch.Tensor:
    margins = 1.0 - signed_y * (x @ weights + bias)
    return torch.relu(margins).mean() + 0.5 * l2_penalty * weights.square().sum()


def fit_linear_svm(
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 300,
    learning_rate: float = 0.05,
    batch_size: int = 32,
    l2_penalty: float = 0.01,
    seed: int = SEED,
) -> LinearSVMResult:
    """Fit a deterministic linear soft-margin SVM with mini-batch SGD."""

    features, labels = _validate_fit_data(
        x,
        y,
        require_both_classes=True,
    )
    epochs = _validate_integer("epochs", epochs, minimum=1)
    learning_rate = _validate_real(
        "learning_rate", learning_rate, minimum=0.0, strictly_greater=True
    )
    batch_size = _validate_integer("batch_size", batch_size, minimum=1)
    l2_penalty = _validate_real("l2_penalty", l2_penalty, minimum=0.0)
    seed = _validate_integer("seed", seed)

    signed_y = labels.to(features.dtype).mul(2.0).sub(1.0)
    weights = torch.zeros(
        features.shape[1],
        dtype=features.dtype,
        device=features.device,
        requires_grad=True,
    )
    bias = torch.zeros(
        (),
        dtype=features.dtype,
        device=features.device,
        requires_grad=True,
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        initial_loss = float(
            _svm_objective(features, signed_y, weights, bias, l2_penalty).item()
        )
    losses = [initial_loss]

    for _ in range(epochs):
        permutation = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            batch_loss = _svm_objective(
                features[indices],
                signed_y[indices],
                weights,
                bias,
                l2_penalty,
            )
            if not torch.isfinite(batch_loss).item():
                raise RuntimeError("SVM training produced a non-finite loss")
            batch_loss.backward()
            if weights.grad is None or bias.grad is None:
                raise RuntimeError("SVM gradients were not calculated")
            with torch.no_grad():
                weights -= learning_rate * weights.grad
                bias -= learning_rate * bias.grad
            weights.grad = None
            bias.grad = None

        with torch.no_grad():
            full_loss = float(
                _svm_objective(features, signed_y, weights, bias, l2_penalty).item()
            )
        if not math.isfinite(full_loss):
            raise RuntimeError("SVM training produced non-finite parameters")
        losses.append(full_loss)

    detached_weights = weights.detach()
    detached_bias = bias.detach()
    if not (
        torch.isfinite(detached_weights).all().item()
        and torch.isfinite(detached_bias).all().item()
    ):
        raise RuntimeError("SVM training produced non-finite parameters")
    return LinearSVMResult(
        weights=detached_weights,
        bias=detached_bias,
        loss_history=tuple(losses),
    )


@torch.no_grad()
def decision_function_linear_svm(
    x: torch.Tensor,
    model: LinearSVMResult,
) -> torch.Tensor:
    if not isinstance(model, LinearSVMResult):
        raise TypeError("model must be a LinearSVMResult")
    if model.weights.ndim != 1 or model.bias.numel() != 1:
        raise ValueError("model contains invalid SVM parameters")
    if not (
        torch.isfinite(model.weights).all().item()
        and torch.isfinite(model.bias).all().item()
    ):
        raise ValueError("model contains non-finite SVM parameters")
    features = _validate_prediction_features(x, model.weights.shape[0])
    scores = features @ model.weights.to(features) + model.bias.to(features)
    if not torch.isfinite(scores).all().item():
        raise RuntimeError("SVM prediction produced non-finite scores")
    return scores.detach().reshape(-1)


@torch.no_grad()
def predict_linear_svm(
    x: torch.Tensor,
    model: LinearSVMResult,
) -> torch.Tensor:
    return (decision_function_linear_svm(x, model) >= 0.0).to(torch.long)


def _gini_from_counts(class_zero: int, class_one: int) -> float:
    total = class_zero + class_one
    if total == 0:
        return 0.0
    probability_zero = class_zero / total
    probability_one = class_one / total
    return 1.0 - probability_zero**2 - probability_one**2


def _best_classification_split(
    x: torch.Tensor,
    y: torch.Tensor,
    candidate_features: list[int],
    min_samples_leaf: int,
) -> tuple[int, float, float] | None:
    n_samples = x.shape[0]
    total_positive = int(y.sum().item())
    best: tuple[int, float, float] | None = None
    tie_tolerance = 1e-12

    for feature_index in candidate_features:
        sorted_values, order = torch.sort(x[:, feature_index], stable=True)
        sorted_labels = y[order]
        cumulative_positive = torch.cumsum(sorted_labels, dim=0)
        for split_position in range(min_samples_leaf, n_samples - min_samples_leaf + 1):
            left_value = float(sorted_values[split_position - 1].item())
            right_value = float(sorted_values[split_position].item())
            if left_value == right_value:
                continue

            left_count = split_position
            right_count = n_samples - split_position
            left_positive = int(cumulative_positive[split_position - 1].item())
            right_positive = total_positive - left_positive
            left_negative = left_count - left_positive
            right_negative = right_count - right_positive
            weighted_impurity = (
                left_count * _gini_from_counts(left_negative, left_positive)
                + right_count * _gini_from_counts(right_negative, right_positive)
            ) / n_samples
            threshold = _safe_midpoint(left_value, right_value)

            if best is None or weighted_impurity < best[2] - tie_tolerance:
                best = (feature_index, threshold, weighted_impurity)
    return best


def _build_tree(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    depth: int,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: int | None,
    feature_generator: torch.Generator,
    split_counts: list[int],
) -> TreeNode:
    class_one = int(y.sum().item())
    class_zero = y.numel() - class_one
    prediction = 1 if class_one > class_zero else 0
    impurity = _gini_from_counts(class_zero, class_one)

    if max_features is None:
        candidate_features = list(range(x.shape[1]))
    else:
        candidate_features = (
            torch.randperm(x.shape[1], generator=feature_generator)[:max_features]
            .sort()
            .values.tolist()
        )

    leaf = TreeNode(
        prediction=prediction,
        n_samples=y.numel(),
        class_counts=(class_zero, class_one),
        impurity=impurity,
    )
    if (
        impurity == 0.0
        or depth >= max_depth
        or y.numel() < min_samples_split
        or y.numel() < 2 * min_samples_leaf
    ):
        return leaf

    best = _best_classification_split(
        x,
        y,
        candidate_features,
        min_samples_leaf,
    )
    if best is None:
        return leaf
    feature_index, threshold, weighted_impurity = best
    if impurity - weighted_impurity <= 1e-7:
        return leaf

    left_mask = _threshold_mask(x[:, feature_index], threshold)
    if (
        int(left_mask.sum().item()) < min_samples_leaf
        or int((~left_mask).sum().item()) < min_samples_leaf
    ):
        return leaf
    split_counts[feature_index] += 1
    left = _build_tree(
        x[left_mask],
        y[left_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        feature_generator=feature_generator,
        split_counts=split_counts,
    )
    right = _build_tree(
        x[~left_mask],
        y[~left_mask],
        depth=depth + 1,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        feature_generator=feature_generator,
        split_counts=split_counts,
    )
    return TreeNode(
        prediction=prediction,
        n_samples=y.numel(),
        class_counts=(class_zero, class_one),
        impurity=impurity,
        feature_index=feature_index,
        threshold=threshold,
        left=left,
        right=right,
    )


def fit_decision_tree(
    x: torch.Tensor,
    y: torch.Tensor,
    max_depth: int = 3,
    min_samples_split: int = 12,
    min_samples_leaf: int = 6,
    max_features: int | None = None,
    seed: int = SEED,
) -> DecisionTreeModel:
    """Fit a small deterministic binary CART classifier."""

    features, labels = _validate_fit_data(
        x,
        y,
        require_both_classes=False,
    )
    max_depth = _validate_integer("max_depth", max_depth, minimum=0)
    min_samples_split = _validate_integer(
        "min_samples_split", min_samples_split, minimum=2
    )
    min_samples_leaf = _validate_integer(
        "min_samples_leaf", min_samples_leaf, minimum=1
    )
    seed = _validate_integer("seed", seed)
    if max_features is not None:
        max_features = _validate_integer("max_features", max_features, minimum=1)
        if max_features > features.shape[1]:
            raise ValueError("max_features cannot exceed the feature count")

    split_counts = [0] * features.shape[1]
    root = _build_tree(
        features,
        labels,
        depth=0,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        feature_generator=torch.Generator().manual_seed(seed),
        split_counts=split_counts,
    )
    return DecisionTreeModel(
        root=root,
        n_features=features.shape[1],
        split_counts=tuple(split_counts),
        max_depth=max_depth,
    )


def _predict_tree_row(row: torch.Tensor, root: TreeNode) -> int:
    node = root
    while node.feature_index is not None:
        if node.threshold is None or node.left is None or node.right is None:
            raise ValueError("tree contains an incomplete internal node")
        if float(row[node.feature_index].item()) <= node.threshold:
            node = node.left
        else:
            node = node.right
    if node.threshold is not None or node.left is not None or node.right is not None:
        raise ValueError("tree contains an invalid leaf")
    return node.prediction


@torch.no_grad()
def predict_decision_tree(
    x: torch.Tensor,
    model: DecisionTreeModel,
) -> torch.Tensor:
    if not isinstance(model, DecisionTreeModel):
        raise TypeError("model must be a DecisionTreeModel")
    features = _validate_prediction_features(x, model.n_features)
    predictions = [
        _predict_tree_row(features[index], model.root)
        for index in range(features.shape[0])
    ]
    return torch.tensor(
        predictions,
        dtype=torch.long,
        device=features.device,
    )


def fit_random_forest(
    x: torch.Tensor,
    y: torch.Tensor,
    n_trees: int = 25,
    max_depth: int = 4,
    min_samples_split: int = 8,
    min_samples_leaf: int = 4,
    max_features: int = 2,
    seed: int = SEED,
) -> RandomForestModel:
    """Fit shallow CART trees to seeded bootstrap samples and feature subsets."""

    features, labels = _validate_fit_data(
        x,
        y,
        require_both_classes=False,
    )
    n_trees = _validate_integer("n_trees", n_trees, minimum=1)
    max_depth = _validate_integer("max_depth", max_depth, minimum=0)
    min_samples_split = _validate_integer(
        "min_samples_split", min_samples_split, minimum=2
    )
    min_samples_leaf = _validate_integer(
        "min_samples_leaf", min_samples_leaf, minimum=1
    )
    max_features = _validate_integer("max_features", max_features, minimum=1)
    seed = _validate_integer("seed", seed)
    if max_features > features.shape[1]:
        raise ValueError("max_features cannot exceed the feature count")

    trees: list[DecisionTreeModel] = []
    bootstrap_indices: list[torch.Tensor] = []
    aggregate_split_counts = [0] * features.shape[1]
    for tree_index in range(n_trees):
        bootstrap_generator = torch.Generator().manual_seed(
            seed + 1000 + tree_index
        )
        indices = torch.randint(
            0,
            features.shape[0],
            (features.shape[0],),
            generator=bootstrap_generator,
        )
        tree = fit_decision_tree(
            features[indices],
            labels[indices],
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            seed=seed + tree_index + 1,
        )
        trees.append(tree)
        bootstrap_indices.append(indices.to(torch.long).detach())
        for feature_index, count in enumerate(tree.split_counts):
            aggregate_split_counts[feature_index] += count

    return RandomForestModel(
        trees=tuple(trees),
        bootstrap_indices=tuple(bootstrap_indices),
        split_counts=tuple(aggregate_split_counts),
    )


@torch.no_grad()
def predict_random_forest_probabilities(
    x: torch.Tensor,
    model: RandomForestModel,
) -> torch.Tensor:
    if not isinstance(model, RandomForestModel):
        raise TypeError("model must be a RandomForestModel")
    if len(model.trees) == 0:
        raise ValueError("forest must contain at least one tree")
    n_features = len(model.split_counts)
    features = _validate_prediction_features(x, n_features)
    tree_predictions = torch.stack(
        [predict_decision_tree(features, tree) for tree in model.trees],
        dim=0,
    )
    probabilities = tree_predictions.to(features.dtype).mean(dim=0)
    return probabilities.detach()


@torch.no_grad()
def predict_random_forest(
    x: torch.Tensor,
    model: RandomForestModel,
) -> torch.Tensor:
    probabilities = predict_random_forest_probabilities(x, model)
    return (probabilities >= 0.5).to(torch.long)


def _best_regression_stump(
    x: torch.Tensor,
    residuals: torch.Tensor,
    min_samples_leaf: int,
) -> tuple[int, float, float, float] | None:
    n_samples = x.shape[0]
    total_sum = float(residuals.sum().item())
    total_square_sum = float(residuals.square().sum().item())
    best: tuple[int, float, float, float] | None = None
    best_sse = math.inf
    tie_tolerance = 1e-12

    for feature_index in range(x.shape[1]):
        sorted_values, order = torch.sort(x[:, feature_index], stable=True)
        sorted_residuals = residuals[order]
        cumulative_sum = torch.cumsum(sorted_residuals, dim=0)
        cumulative_square_sum = torch.cumsum(sorted_residuals.square(), dim=0)
        for split_position in range(min_samples_leaf, n_samples - min_samples_leaf + 1):
            left_feature_value = float(sorted_values[split_position - 1].item())
            right_feature_value = float(sorted_values[split_position].item())
            if left_feature_value == right_feature_value:
                continue

            left_count = split_position
            right_count = n_samples - split_position
            left_sum = float(cumulative_sum[split_position - 1].item())
            left_square_sum = float(
                cumulative_square_sum[split_position - 1].item()
            )
            right_sum = total_sum - left_sum
            right_square_sum = total_square_sum - left_square_sum
            left_sse = left_square_sum - left_sum * left_sum / left_count
            right_sse = right_square_sum - right_sum * right_sum / right_count
            combined_sse = max(0.0, left_sse) + max(0.0, right_sse)
            if combined_sse < best_sse - tie_tolerance:
                best_sse = combined_sse
                best = (
                    feature_index,
                    _safe_midpoint(left_feature_value, right_feature_value),
                    left_sum / left_count,
                    right_sum / right_count,
                )
    return best


def fit_logistic_gradient_boosting(
    x: torch.Tensor,
    y: torch.Tensor,
    n_estimators: int = 50,
    learning_rate: float = 0.20,
    min_samples_leaf: int = 8,
    loss_tolerance: float = 1e-10,
    max_backtracking_steps: int = 12,
) -> GradientBoostingModel:
    """Fit additive regression stumps to logistic pseudo-residuals."""

    features, labels = _validate_fit_data(
        x,
        y,
        require_both_classes=True,
    )
    n_estimators = _validate_integer("n_estimators", n_estimators, minimum=1)
    learning_rate = _validate_real(
        "learning_rate", learning_rate, minimum=0.0, strictly_greater=True
    )
    min_samples_leaf = _validate_integer(
        "min_samples_leaf", min_samples_leaf, minimum=1
    )
    loss_tolerance = _validate_real(
        "loss_tolerance", loss_tolerance, minimum=0.0
    )
    max_backtracking_steps = _validate_integer(
        "max_backtracking_steps", max_backtracking_steps, minimum=0
    )

    targets = labels.to(features.dtype)
    machine_epsilon = torch.finfo(features.dtype).eps
    positive_probability = float(targets.mean().item())
    positive_probability = min(
        max(positive_probability, machine_epsilon),
        1.0 - machine_epsilon,
    )
    base_logit = math.log(positive_probability / (1.0 - positive_probability))
    logits = torch.full_like(targets, base_logit)
    current_loss = float(
        F.binary_cross_entropy_with_logits(logits, targets).item()
    )
    loss_history = [current_loss]
    stumps: list[RegressionStump] = []
    split_counts = [0] * features.shape[1]

    for _ in range(n_estimators):
        residuals = targets - torch.sigmoid(logits)
        best = _best_regression_stump(
            features,
            residuals,
            min_samples_leaf,
        )
        if best is None:
            break
        feature_index, threshold, left_value, right_value = best
        if not all(
            math.isfinite(value)
            for value in (threshold, left_value, right_value)
        ):
            raise RuntimeError("boosting produced a non-finite stump")
        stump_outputs = torch.where(
            _threshold_mask(features[:, feature_index], threshold),
            torch.as_tensor(left_value, dtype=features.dtype, device=features.device),
            torch.as_tensor(right_value, dtype=features.dtype, device=features.device),
        )

        accepted_step: float | None = None
        accepted_logits: torch.Tensor | None = None
        accepted_loss: float | None = None
        for backtracking_index in range(max_backtracking_steps + 1):
            step = learning_rate / (2**backtracking_index)
            candidate_logits = logits + step * stump_outputs
            candidate_loss = float(
                F.binary_cross_entropy_with_logits(
                    candidate_logits,
                    targets,
                ).item()
            )
            if not math.isfinite(candidate_loss):
                continue
            if candidate_loss < current_loss - loss_tolerance:
                accepted_step = step
                accepted_logits = candidate_logits
                accepted_loss = candidate_loss
                break
        if accepted_step is None or accepted_logits is None or accepted_loss is None:
            break

        stumps.append(
            RegressionStump(
                feature_index=feature_index,
                threshold=threshold,
                left_value=left_value,
                right_value=right_value,
                step_size=accepted_step,
            )
        )
        split_counts[feature_index] += 1
        logits = accepted_logits
        current_loss = accepted_loss
        loss_history.append(current_loss)

    if not all(math.isfinite(value) for value in loss_history):
        raise RuntimeError("boosting produced a non-finite loss")
    return GradientBoostingModel(
        base_logit=base_logit,
        stumps=tuple(stumps),
        loss_history=tuple(loss_history),
        split_counts=tuple(split_counts),
    )


@torch.no_grad()
def decision_function_gradient_boosting(
    x: torch.Tensor,
    model: GradientBoostingModel,
) -> torch.Tensor:
    if not isinstance(model, GradientBoostingModel):
        raise TypeError("model must be a GradientBoostingModel")
    if not math.isfinite(model.base_logit):
        raise ValueError("model contains a non-finite base logit")
    features = _validate_prediction_features(x, len(model.split_counts))
    scores = torch.full(
        (features.shape[0],),
        model.base_logit,
        dtype=features.dtype,
        device=features.device,
    )
    for stump in model.stumps:
        if not isinstance(stump, RegressionStump):
            raise ValueError("model contains an invalid stump")
        if not 0 <= stump.feature_index < features.shape[1]:
            raise ValueError("stump feature index is outside the model feature range")
        if not all(
            math.isfinite(value)
            for value in (
                stump.threshold,
                stump.left_value,
                stump.right_value,
                stump.step_size,
            )
        ):
            raise ValueError("model contains a non-finite stump")
        values = torch.where(
            _threshold_mask(
                features[:, stump.feature_index],
                stump.threshold,
            ),
            torch.as_tensor(
                stump.left_value,
                dtype=features.dtype,
                device=features.device,
            ),
            torch.as_tensor(
                stump.right_value,
                dtype=features.dtype,
                device=features.device,
            ),
        )
        scores += stump.step_size * values
    if not torch.isfinite(scores).all().item():
        raise RuntimeError("boosting prediction produced non-finite scores")
    return scores.detach()


@torch.no_grad()
def predict_gradient_boosting_probabilities(
    x: torch.Tensor,
    model: GradientBoostingModel,
) -> torch.Tensor:
    return torch.sigmoid(decision_function_gradient_boosting(x, model)).detach()


@torch.no_grad()
def predict_gradient_boosting(
    x: torch.Tensor,
    model: GradientBoostingModel,
) -> torch.Tensor:
    probabilities = predict_gradient_boosting_probabilities(x, model)
    return (probabilities >= 0.5).to(torch.long)


def fit_lda(
    x: torch.Tensor,
    y: torch.Tensor,
    covariance_regularization: float = 1e-4,
    max_cholesky_attempts: int = 6,
) -> LDAModel:
    """Fit binary LDA using a regularized pooled covariance and Cholesky."""

    features, labels = _validate_fit_data(
        x,
        y,
        require_both_classes=True,
    )
    covariance_regularization = _validate_real(
        "covariance_regularization",
        covariance_regularization,
        minimum=0.0,
        strictly_greater=True,
    )
    max_cholesky_attempts = _validate_integer(
        "max_cholesky_attempts", max_cholesky_attempts, minimum=1
    )

    class_masks = (labels == 0, labels == 1)
    class_counts = [int(mask.sum().item()) for mask in class_masks]
    if min(class_counts) < 2:
        raise ValueError("LDA requires at least two examples from each class")
    classes = torch.tensor([0, 1], dtype=torch.long, device=features.device)
    class_priors = torch.tensor(
        [count / features.shape[0] for count in class_counts],
        dtype=features.dtype,
        device=features.device,
    )
    class_means = torch.stack(
        [features[mask].mean(dim=0) for mask in class_masks],
        dim=0,
    )
    centered = torch.cat(
        [
            features[mask] - class_means[class_index]
            for class_index, mask in enumerate(class_masks)
        ],
        dim=0,
    )
    pooled_covariance = centered.T @ centered / (features.shape[0] - 2)
    identity = torch.eye(
        features.shape[1],
        dtype=features.dtype,
        device=features.device,
    )

    accepted_covariance: torch.Tensor | None = None
    cholesky_factor: torch.Tensor | None = None
    accepted_regularization: float | None = None
    jitter = covariance_regularization
    for _ in range(max_cholesky_attempts):
        candidate_covariance = pooled_covariance + jitter * identity
        factor, info = torch.linalg.cholesky_ex(candidate_covariance)
        if int(info.max().item()) == 0 and torch.isfinite(factor).all().item():
            accepted_covariance = candidate_covariance
            cholesky_factor = factor
            accepted_regularization = jitter
            break
        jitter *= 10.0
    if (
        accepted_covariance is None
        or cholesky_factor is None
        or accepted_regularization is None
    ):
        raise ValueError("LDA covariance was not positive definite after regularization")
    tensors = (
        classes,
        class_priors,
        class_means,
        accepted_covariance,
        cholesky_factor,
    )
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise RuntimeError("LDA fitting produced non-finite parameters")
    return LDAModel(
        classes=classes.detach(),
        class_priors=class_priors.detach(),
        class_means=class_means.detach(),
        covariance=accepted_covariance.detach(),
        cholesky_factor=cholesky_factor.detach(),
        regularization=accepted_regularization,
    )


def _validate_lda_model(model: object) -> LDAModel:
    if not isinstance(model, LDAModel):
        raise TypeError("model must be an LDAModel")
    if model.class_means.ndim != 2 or model.class_means.shape[0] != 2:
        raise ValueError("model contains invalid LDA class means")
    n_features = model.class_means.shape[1]
    if model.class_priors.shape != (2,) or model.classes.shape != (2,):
        raise ValueError("model contains invalid LDA class metadata")
    if model.covariance.shape != (n_features, n_features):
        raise ValueError("model contains an invalid LDA covariance")
    if model.cholesky_factor.shape != (n_features, n_features):
        raise ValueError("model contains an invalid LDA Cholesky factor")
    tensors = (
        model.class_priors,
        model.class_means,
        model.covariance,
        model.cholesky_factor,
    )
    if not all(torch.isfinite(tensor).all().item() for tensor in tensors):
        raise ValueError("model contains non-finite LDA parameters")
    return model


@torch.no_grad()
def predict_lda_scores(
    x: torch.Tensor,
    model: LDAModel,
) -> torch.Tensor:
    model = _validate_lda_model(model)
    features = _validate_prediction_features(x, model.class_means.shape[1])
    means = model.class_means.to(features)
    factor = model.cholesky_factor.to(features)
    coefficients = torch.cholesky_solve(means.T, factor).T
    constants = (
        -0.5 * (means * coefficients).sum(dim=1)
        + torch.log(model.class_priors.to(features))
    )
    scores = features @ coefficients.T + constants
    if not torch.isfinite(scores).all().item():
        raise RuntimeError("LDA prediction produced non-finite scores")
    return scores.detach()


@torch.no_grad()
def predict_lda_probabilities(
    x: torch.Tensor,
    model: LDAModel,
) -> torch.Tensor:
    return torch.softmax(predict_lda_scores(x, model), dim=1).detach()


@torch.no_grad()
def predict_lda(
    x: torch.Tensor,
    model: LDAModel,
) -> torch.Tensor:
    model = _validate_lda_model(model)
    probability_indices = predict_lda_probabilities(x, model).argmax(dim=1)
    return model.classes.to(probability_indices.device)[probability_indices].detach()


def _accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    return float((y_true == y_pred).to(torch.float32).mean().item())


def _metrics_row(
    *,
    section: str,
    dataset: str,
    model_name: str,
    setting: str,
    splits: ClassificationSplits,
    train_predictions: torch.Tensor,
    validation_predictions: torch.Tensor,
    test_predictions: torch.Tensor,
    initial_loss: float | str = "",
    final_loss: float | str = "",
    training_steps: int | str = "",
    feature_split_counts: tuple[int, ...] | None = None,
) -> dict[str, str | int | float]:
    train_accuracy = _accuracy(splits.y_train, train_predictions)
    validation_accuracy = _accuracy(
        splits.y_validation,
        validation_predictions,
    )
    test_accuracy = _accuracy(splits.y_test, test_predictions)
    matrix = confusion_matrix(splits.y_test, test_predictions, num_classes=2)
    counts = [int(value) for value in matrix.reshape(-1).tolist()]
    return {
        "section": section,
        "dataset": dataset,
        "model": model_name,
        "setting": setting,
        "train_accuracy": train_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_accuracy": test_accuracy,
        "train_validation_gap": train_accuracy - validation_accuracy,
        "test_tn": counts[0],
        "test_fp": counts[1],
        "test_fn": counts[2],
        "test_tp": counts[3],
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "training_steps": training_steps,
        "feature_split_counts": (
            ""
            if feature_split_counts is None
            else "|".join(str(value) for value in feature_split_counts)
        ),
    }


def _serialize_csv_value(field: str, value: str | int | float) -> str:
    if value == "":
        return ""
    if field in _FLOAT_CSV_FIELDS:
        return format(float(value), ".10g")
    if field in _INTEGER_CSV_FIELDS:
        return str(int(value))
    return str(value)


def _write_metrics_csv(
    rows: list[dict[str, str | int | float]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _serialize_csv_value(field, row[field])
                    for field in CSV_FIELDS
                }
            )


def run_day03_demo(
    output_dir: Path | None = None,
) -> list[dict[str, str | int | float]]:
    """Train the fixed Day 3 comparison and write its six-row metrics CSV."""

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "results"
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path or None")
    output_dir.mkdir(parents=True, exist_ok=True)

    margin_splits = make_margin_splits()
    tabular_splits = make_mixed_signal_splits()
    margin_svm = fit_linear_svm(margin_splits.x_train, margin_splits.y_train)
    tabular_svm = fit_linear_svm(tabular_splits.x_train, tabular_splits.y_train)
    tree = fit_decision_tree(tabular_splits.x_train, tabular_splits.y_train)
    forest = fit_random_forest(tabular_splits.x_train, tabular_splits.y_train)
    boosting = fit_logistic_gradient_boosting(
        tabular_splits.x_train,
        tabular_splits.y_train,
    )
    lda = fit_lda(tabular_splits.x_train, tabular_splits.y_train)

    rows = [
        _metrics_row(
            section="margin_check",
            dataset="day01_separable_blobs",
            model_name="linear SVM",
            setting="epochs=300;lr=0.05;l2=0.01",
            splits=margin_splits,
            train_predictions=predict_linear_svm(
                margin_splits.x_train, margin_svm
            ),
            validation_predictions=predict_linear_svm(
                margin_splits.x_validation, margin_svm
            ),
            test_predictions=predict_linear_svm(margin_splits.x_test, margin_svm),
            initial_loss=margin_svm.loss_history[0],
            final_loss=margin_svm.loss_history[-1],
            training_steps=300,
        ),
        _metrics_row(
            section="tabular_comparison",
            dataset="mixed_signal_tabular",
            model_name="linear SVM",
            setting="epochs=300;lr=0.05;l2=0.01",
            splits=tabular_splits,
            train_predictions=predict_linear_svm(
                tabular_splits.x_train, tabular_svm
            ),
            validation_predictions=predict_linear_svm(
                tabular_splits.x_validation, tabular_svm
            ),
            test_predictions=predict_linear_svm(
                tabular_splits.x_test, tabular_svm
            ),
            initial_loss=tabular_svm.loss_history[0],
            final_loss=tabular_svm.loss_history[-1],
            training_steps=300,
        ),
        _metrics_row(
            section="tabular_comparison",
            dataset="mixed_signal_tabular",
            model_name="decision tree",
            setting="depth=3;min_split=12;min_leaf=6",
            splits=tabular_splits,
            train_predictions=predict_decision_tree(
                tabular_splits.x_train, tree
            ),
            validation_predictions=predict_decision_tree(
                tabular_splits.x_validation, tree
            ),
            test_predictions=predict_decision_tree(tabular_splits.x_test, tree),
            feature_split_counts=tree.split_counts,
        ),
        _metrics_row(
            section="tabular_comparison",
            dataset="mixed_signal_tabular",
            model_name="random forest",
            setting="trees=25;depth=4;max_features=2",
            splits=tabular_splits,
            train_predictions=predict_random_forest(
                tabular_splits.x_train, forest
            ),
            validation_predictions=predict_random_forest(
                tabular_splits.x_validation, forest
            ),
            test_predictions=predict_random_forest(
                tabular_splits.x_test, forest
            ),
            feature_split_counts=forest.split_counts,
        ),
        _metrics_row(
            section="tabular_comparison",
            dataset="mixed_signal_tabular",
            model_name="logistic gradient boosting",
            setting="estimators=50;lr=0.20;min_leaf=8",
            splits=tabular_splits,
            train_predictions=predict_gradient_boosting(
                tabular_splits.x_train, boosting
            ),
            validation_predictions=predict_gradient_boosting(
                tabular_splits.x_validation, boosting
            ),
            test_predictions=predict_gradient_boosting(
                tabular_splits.x_test, boosting
            ),
            initial_loss=boosting.loss_history[0],
            final_loss=boosting.loss_history[-1],
            training_steps=len(boosting.stumps),
            feature_split_counts=boosting.split_counts,
        ),
        _metrics_row(
            section="tabular_comparison",
            dataset="mixed_signal_tabular",
            model_name="LDA",
            setting="regularization=0.0001",
            splits=tabular_splits,
            train_predictions=predict_lda(tabular_splits.x_train, lda),
            validation_predictions=predict_lda(
                tabular_splits.x_validation, lda
            ),
            test_predictions=predict_lda(tabular_splits.x_test, lda),
        ),
    ]
    _write_metrics_csv(rows, output_dir / "day03_metrics.csv")
    return rows


if __name__ == "__main__":
    generated_rows = run_day03_demo()
    for generated_row in generated_rows:
        print(
            f"{generated_row['model']}: "
            f"test accuracy={float(generated_row['test_accuracy']):.4f}"
        )
