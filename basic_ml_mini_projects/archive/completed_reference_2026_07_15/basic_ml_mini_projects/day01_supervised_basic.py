"""Day 1 supervised-learning algorithms implemented with PyTorch tensors.

The model logic intentionally avoids scikit-learn so each calculation remains
visible. Running this file trains/evaluates linear regression, ridge
regression, logistic regression, k-nearest neighbors, and Gaussian Naive
Bayes on deterministic toy data.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

matplotlib_cache = Path(tempfile.gettempdir()) / "ml-roadmap-matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


SEED = 42
EPSILON = 1e-8


def _add_bias_column(x: torch.Tensor) -> torch.Tensor:
    """Prepend a column of ones so a bias can join the weight vector."""

    ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    return torch.cat((ones, x), dim=1)


def generate_data(
    n_samples: int = 100,
    noise_std: float = 0.3,
    dimensions: int = 1,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate and randomly split a 1D or 2D linear-regression dataset."""

    if n_samples < 5:
        raise ValueError("n_samples must be at least 5")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if dimensions not in (1, 2):
        raise ValueError("dimensions must be either 1 or 2")

    generator = torch.Generator().manual_seed(seed)
    x = -3.0 + 6.0 * torch.rand(
        (n_samples, dimensions), generator=generator
    )
    true_weights = torch.tensor([[3.0], [1.0]])[:dimensions]
    noise = noise_std * torch.randn(
        (n_samples, 1), generator=generator
    )
    y = x @ true_weights + 2.0 + noise

    indices = torch.randperm(n_samples, generator=generator)
    train_size = int(0.8 * n_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return (
        x[train_indices],
        x[test_indices],
        y[train_indices],
        y[test_indices],
    )


@torch.no_grad()
def predict_linear_regression(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Return predictions for a fitted linear model."""

    return x @ weights + bias


@torch.no_grad()
def regression_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> dict[str, float]:
    """Calculate MSE, MAE, and a compact residual summary."""

    residuals = y_true - y_pred
    return {
        "mse": torch.mean(residuals.square()).item(),
        "mae": torch.mean(residuals.abs()).item(),
        "residual_mean": residuals.mean().item(),
        "residual_std": residuals.std(unbiased=False).item(),
    }


@torch.no_grad()
def fit_linear_regression_closed_form(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ordinary least squares with the normal equation."""

    design_matrix = _add_bias_column(x_train)
    gram_matrix = design_matrix.T @ design_matrix
    target_projection = design_matrix.T @ y_train
    wb = torch.linalg.solve(gram_matrix, target_projection)
    return wb[1:], wb[0]


@torch.no_grad()
def fit_ridge_regression(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    l2_penalty: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit closed-form ridge regression without regularizing the bias."""

    if l2_penalty < 0:
        raise ValueError("l2_penalty must be non-negative")

    design_matrix = _add_bias_column(x_train)
    penalty_matrix = torch.eye(
        design_matrix.shape[1],
        dtype=design_matrix.dtype,
        device=design_matrix.device,
    )
    penalty_matrix[0, 0] = 0.0

    gram_matrix = design_matrix.T @ design_matrix
    target_projection = design_matrix.T @ y_train
    regularized_gram = gram_matrix + l2_penalty * penalty_matrix
    wb = torch.linalg.solve(regularized_gram, target_projection)
    return wb[1:], wb[0]


def train_linear_reg(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 800,
    learning_rate: float = 0.03,
    tolerance: float = 1e-10,
    log_every: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Fit linear regression with manual full-batch gradient descent."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have equal sample counts")

    design_matrix = _add_bias_column(x_train)
    wb = torch.zeros(
        (design_matrix.shape[1], 1),
        dtype=x_train.dtype,
        device=x_train.device,
        requires_grad=True,
    )
    losses: list[float] = []

    for epoch in range(epochs):
        y_pred = design_matrix @ wb
        loss = torch.mean((y_pred - y_train).square()) / 2.0
        current_loss = loss.item()
        losses.append(current_loss)

        loss.backward()
        gradient = wb.grad
        if gradient is None:
            raise RuntimeError("linear-regression gradient was not calculated")
        with torch.no_grad():
            wb -= learning_rate * gradient
        wb.grad = None

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"linear epoch {epoch:4d} | train loss {current_loss:.6f}")

        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < tolerance:
            break

    return wb[1:].detach(), wb[0].detach(), losses


def generate_binary_classification_data(
    n_samples: int = 300,
    test_fraction: float = 0.25,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate two shuffled 2D Gaussian blobs for binary classification."""

    if n_samples < 8:
        raise ValueError("n_samples must be at least 8")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")

    generator = torch.Generator().manual_seed(seed)
    class_zero_count = n_samples // 2
    class_one_count = n_samples - class_zero_count
    class_zero = 0.75 * torch.randn(
        (class_zero_count, 2), generator=generator
    ) + torch.tensor([-1.5, -1.0])
    class_one = 0.75 * torch.randn(
        (class_one_count, 2), generator=generator
    ) + torch.tensor([1.5, 1.0])

    x = torch.cat((class_zero, class_one), dim=0)
    y = torch.cat(
        (
            torch.zeros((class_zero_count, 1)),
            torch.ones((class_one_count, 1)),
        ),
        dim=0,
    )

    indices = torch.randperm(n_samples, generator=generator)
    test_size = max(1, int(test_fraction * n_samples))
    train_size = n_samples - test_size
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    return x[train_indices], x[test_indices], y[train_indices], y[test_indices]


def standardize_from_train(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Standardize both splits using statistics calculated on training data."""

    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(EPSILON)
    return (x_train - mean) / std, (x_test - mean) / std, mean, std


def train_logistic_regression(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 600,
    learning_rate: float = 0.1,
    tolerance: float = 1e-9,
    log_every: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Fit binary logistic regression with manual gradient descent."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train must have equal sample counts")
    if y_train.ndim != 2 or y_train.shape[1] != 1:
        raise ValueError("y_train must have shape (n_samples, 1)")
    if not torch.all((y_train == 0) | (y_train == 1)):
        raise ValueError("logistic-regression targets must be 0 or 1")

    weights = torch.zeros(
        (x_train.shape[1], 1),
        dtype=x_train.dtype,
        device=x_train.device,
        requires_grad=True,
    )
    bias = torch.zeros(
        1,
        dtype=x_train.dtype,
        device=x_train.device,
        requires_grad=True,
    )
    losses: list[float] = []

    for epoch in range(epochs):
        logits = x_train @ weights + bias
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        current_loss = loss.item()
        losses.append(current_loss)

        loss.backward()
        weight_gradient = weights.grad
        bias_gradient = bias.grad
        if weight_gradient is None or bias_gradient is None:
            raise RuntimeError("logistic-regression gradients were not calculated")
        with torch.no_grad():
            weights -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient
        weights.grad = None
        bias.grad = None

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"logistic epoch {epoch:4d} | train loss {current_loss:.6f}")

        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < tolerance:
            break

    return weights.detach(), bias.detach(), losses


@torch.no_grad()
def predict_logistic_probabilities(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Convert logistic-regression logits into class-one probabilities."""

    return torch.sigmoid(x @ weights + bias)


@torch.no_grad()
def predict_logistic_labels(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Threshold logistic probabilities to obtain binary class labels."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    probabilities = predict_logistic_probabilities(x, weights, bias)
    return (probabilities >= threshold).to(torch.long)


@torch.no_grad()
def confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build a confusion matrix whose rows are true and columns predicted."""

    true_labels = y_true.reshape(-1).to(torch.long)
    predicted_labels = y_pred.reshape(-1).to(torch.long)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("y_true and y_pred must contain equal sample counts")
    if torch.any((true_labels < 0) | (true_labels >= num_classes)):
        raise ValueError("y_true contains a label outside the class range")
    if torch.any((predicted_labels < 0) | (predicted_labels >= num_classes)):
        raise ValueError("y_pred contains a label outside the class range")

    flat_indices = true_labels * num_classes + predicted_labels
    counts = torch.bincount(flat_indices, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes)


@torch.no_grad()
def classification_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    probabilities: torch.Tensor | None = None,
) -> dict[str, float]:
    """Calculate accuracy and optional binary log loss."""

    true_labels = y_true.reshape(-1).to(torch.long)
    predicted_labels = y_pred.reshape(-1).to(torch.long)
    if true_labels.shape != predicted_labels.shape:
        raise ValueError("y_true and y_pred must contain equal sample counts")
    metrics = {
        "accuracy": (true_labels == predicted_labels).float().mean().item()
    }
    if probabilities is not None:
        if probabilities.numel() != true_labels.numel():
            raise ValueError("probabilities must contain one value per sample")
        binary_targets = true_labels.to(probabilities.dtype).reshape(-1, 1)
        probability_epsilon = torch.finfo(probabilities.dtype).eps
        safe_probabilities = probabilities.reshape(-1, 1).clamp(
            probability_epsilon, 1.0 - probability_epsilon
        )
        metrics["log_loss"] = F.binary_cross_entropy(
            safe_probabilities, binary_targets
        ).item()
    return metrics


@torch.no_grad()
def predict_knn(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Predict labels by majority vote among PyTorch-computed neighbors."""

    if not 1 <= k <= x_train.shape[0]:
        raise ValueError("k must be between 1 and the number of training samples")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("training and test features must have equal dimensions")
    if x_train.shape[0] != y_train.numel():
        raise ValueError("x_train and y_train must have equal sample counts")

    train_labels = y_train.reshape(-1).to(torch.long)
    distances = torch.cdist(x_test, x_train)
    neighbor_indices = torch.topk(
        distances, k=k, dim=1, largest=False
    ).indices
    neighbor_labels = train_labels[neighbor_indices]
    return torch.mode(neighbor_labels, dim=1).values.reshape(-1, 1)


@dataclass(frozen=True)
class GaussianNBModel:
    """Parameters learned by the Gaussian Naive Bayes classifier."""

    classes: torch.Tensor
    means: torch.Tensor
    variances: torch.Tensor
    log_priors: torch.Tensor


@torch.no_grad()
def fit_gaussian_naive_bayes(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    variance_smoothing: float = 1e-6,
) -> GaussianNBModel:
    """Estimate independent Gaussian feature distributions for each class."""

    if variance_smoothing <= 0:
        raise ValueError("variance_smoothing must be positive")
    if x_train.shape[0] != y_train.numel():
        raise ValueError("x_train and y_train must have equal sample counts")

    labels = y_train.reshape(-1).to(torch.long)
    classes = torch.unique(labels, sorted=True)
    if classes.numel() < 2:
        raise ValueError("Gaussian Naive Bayes requires at least two classes")
    means = []
    variances = []
    log_priors = []

    for class_label in classes:
        class_features = x_train[labels == class_label]
        means.append(class_features.mean(dim=0))
        variances.append(
            class_features.var(dim=0, unbiased=False) + variance_smoothing
        )
        log_priors.append(
            torch.log(
                torch.tensor(
                    class_features.shape[0] / x_train.shape[0],
                    dtype=x_train.dtype,
                    device=x_train.device,
                )
            )
        )

    return GaussianNBModel(
        classes=classes,
        means=torch.stack(means),
        variances=torch.stack(variances),
        log_priors=torch.stack(log_priors),
    )


@torch.no_grad()
def predict_gaussian_nb_log_probabilities(
    x: torch.Tensor,
    model: GaussianNBModel,
) -> torch.Tensor:
    """Return normalized per-class log probabilities."""

    differences = x[:, None, :] - model.means[None, :, :]
    log_likelihoods = -0.5 * (
        torch.log(2.0 * math.pi * model.variances)[None, :, :]
        + differences.square() / model.variances[None, :, :]
    ).sum(dim=2)
    joint_log_probabilities = log_likelihoods + model.log_priors[None, :]
    return joint_log_probabilities - torch.logsumexp(
        joint_log_probabilities, dim=1, keepdim=True
    )


@torch.no_grad()
def predict_gaussian_nb_probabilities(
    x: torch.Tensor,
    model: GaussianNBModel,
) -> torch.Tensor:
    """Return normalized per-class probabilities."""

    return predict_gaussian_nb_log_probabilities(x, model).exp()


@torch.no_grad()
def predict_gaussian_nb_labels(
    x: torch.Tensor,
    model: GaussianNBModel,
) -> torch.Tensor:
    """Choose the class with the greatest Gaussian posterior probability."""

    log_probabilities = predict_gaussian_nb_log_probabilities(x, model)
    best_class_indices = log_probabilities.argmax(dim=1)
    return model.classes[best_class_indices].reshape(-1, 1)


def _write_metrics_table(
    rows: list[dict[str, str | float]],
    output_path: Path,
) -> None:
    """Save one compact table containing regression and classification metrics."""

    fieldnames = [
        "task",
        "model",
        "mse",
        "mae",
        "residual_mean",
        "residual_std",
        "accuracy",
        "log_loss",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _save_regression_plot(
    y_true: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    """Plot predicted against actual regression targets."""

    all_values = [y_true, *predictions.values()]
    lower = min(values.min().item() for values in all_values)
    upper = max(values.max().item() for values in all_values)

    figure, axis = plt.subplots(figsize=(6, 5))
    for model_name, y_pred in predictions.items():
        axis.scatter(
            y_true.reshape(-1).tolist(),
            y_pred.reshape(-1).tolist(),
            alpha=0.7,
            label=model_name,
        )
    axis.plot([lower, upper], [lower, upper], "k--", label="perfect prediction")
    axis.set_xlabel("Actual target")
    axis.set_ylabel("Predicted target")
    axis.set_title("Day 1 regression predictions")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_logistic_loss_plot(losses: list[float], output_path: Path) -> None:
    """Plot binary cross-entropy over logistic-regression training."""

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(range(len(losses)), losses)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary cross-entropy")
    axis.set_title("Logistic-regression training loss")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _save_classification_plot(
    x_test: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    output_path: Path,
) -> None:
    """Compare held-out predictions from the three classifiers."""

    figure, axes = plt.subplots(1, len(predictions), figsize=(13, 4), sharex=True)
    for axis, (model_name, labels) in zip(axes, predictions.items()):
        axis.scatter(
            x_test[:, 0].tolist(),
            x_test[:, 1].tolist(),
            c=labels.reshape(-1).tolist(),
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            alpha=0.8,
        )
        axis.set_title(model_name)
        axis.set_xlabel("standardized feature 1")
        axis.set_ylabel("standardized feature 2")
    figure.suptitle("Held-out classification predictions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _print_metrics(model_name: str, metrics: dict[str, float]) -> None:
    formatted_metrics = ", ".join(
        f"{name}={value:.4f}" for name, value in metrics.items()
    )
    print(f"{model_name}: {formatted_metrics}")


def run_day01_demo(
    output_dir: Path | None = None,
) -> list[dict[str, str | float]]:
    """Run every Day 1 algorithm and save the requested small artifacts."""

    if output_dir is None:
        output_dir = Path(__file__).with_name("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, str | float]] = []

    print("\n=== Regression ===")
    x_train, x_test, y_train, y_test = generate_data(
        n_samples=160, noise_std=0.3, dimensions=2
    )
    gd_weights, gd_bias, _ = train_linear_reg(
        x_train,
        y_train,
        epochs=800,
        learning_rate=0.03,
        log_every=200,
    )
    ols_weights, ols_bias = fit_linear_regression_closed_form(x_train, y_train)
    ridge_weights, ridge_bias = fit_ridge_regression(
        x_train, y_train, l2_penalty=5.0
    )

    regression_predictions = {
        "linear GD": predict_linear_regression(x_test, gd_weights, gd_bias),
        "ordinary least squares": predict_linear_regression(
            x_test, ols_weights, ols_bias
        ),
        "ridge (lambda=5)": predict_linear_regression(
            x_test, ridge_weights, ridge_bias
        ),
    }
    for model_name, predictions in regression_predictions.items():
        metrics = regression_metrics(y_test, predictions)
        _print_metrics(model_name, metrics)
        metric_rows.append({"task": "regression", "model": model_name, **metrics})

    print("linear GD weights:", gd_weights.reshape(-1).tolist())
    print("closed-form weights:", ols_weights.reshape(-1).tolist())
    print("ridge weights:", ridge_weights.reshape(-1).tolist())
    print(
        "GD and closed-form weights close:",
        torch.allclose(gd_weights, ols_weights, atol=0.05),
    )
    print(
        "ridge weight norm smaller than OLS:",
        ridge_weights.norm().item() < ols_weights.norm().item(),
    )

    print("\n=== Classification ===")
    x_train_cls, x_test_cls, y_train_cls, y_test_cls = (
        generate_binary_classification_data()
    )
    x_train_cls, x_test_cls, _, _ = standardize_from_train(
        x_train_cls, x_test_cls
    )

    logistic_weights, logistic_bias, logistic_losses = (
        train_logistic_regression(
            x_train_cls,
            y_train_cls,
            epochs=600,
            learning_rate=0.1,
            log_every=150,
        )
    )
    logistic_probabilities = predict_logistic_probabilities(
        x_test_cls, logistic_weights, logistic_bias
    )
    logistic_labels = predict_logistic_labels(
        x_test_cls, logistic_weights, logistic_bias
    )
    knn_labels = predict_knn(
        x_train_cls, y_train_cls, x_test_cls, k=5
    )
    gaussian_model = fit_gaussian_naive_bayes(x_train_cls, y_train_cls)
    gaussian_probabilities = predict_gaussian_nb_probabilities(
        x_test_cls, gaussian_model
    )
    gaussian_labels = predict_gaussian_nb_labels(x_test_cls, gaussian_model)

    classifier_outputs = {
        "logistic regression": (logistic_labels, logistic_probabilities),
        "k-nearest neighbors": (knn_labels, None),
        "Gaussian Naive Bayes": (
            gaussian_labels,
            gaussian_probabilities[:, 1:2],
        ),
    }
    classification_predictions = {}
    for model_name, (labels, probabilities) in classifier_outputs.items():
        metrics = classification_metrics(y_test_cls, labels, probabilities)
        _print_metrics(model_name, metrics)
        matrix = confusion_matrix(y_test_cls, labels, num_classes=2)
        print(f"{model_name} confusion matrix:\n{matrix}")
        metric_rows.append(
            {"task": "classification", "model": model_name, **metrics}
        )
        classification_predictions[model_name] = labels

    _write_metrics_table(metric_rows, output_dir / "day01_metrics.csv")
    _save_regression_plot(
        y_test,
        regression_predictions,
        output_dir / "day01_regression_predictions.png",
    )
    _save_logistic_loss_plot(
        logistic_losses,
        output_dir / "day01_logistic_loss.png",
    )
    _save_classification_plot(
        x_test_cls,
        classification_predictions,
        output_dir / "day01_classification_predictions.png",
    )

    print(f"\nSaved metrics and plots to: {output_dir}")
    return metric_rows


if __name__ == "__main__":
    run_day01_demo()
