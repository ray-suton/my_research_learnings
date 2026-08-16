import torch

torch.manual_seed(42)


def generate_svm_data(n_per_class=64, dimensions=2, margin=2.0, noise=0.55, seed=42):
    generator = torch.Generator().manual_seed(seed)

    negative = noise * torch.randn(n_per_class, dimensions, generator=generator)
    positive = noise * torch.randn(n_per_class, dimensions, generator=generator)
    negative[:, 0] -= margin
    positive[:, 0] += margin

    x = torch.cat([negative, positive], dim=0)
    y = torch.cat(
        [
            -torch.ones(n_per_class),
            torch.ones(n_per_class),
        ],
        dim=0,
    )

    permutation = torch.randperm(x.shape[0], generator=generator)
    return x[permutation], y[permutation]


def split_data(x, y, train_ratio=0.75):
    permutation = torch.randperm(x.shape[0])
    cutoff = int(x.shape[0] * train_ratio)

    train_indices = permutation[:cutoff]
    test_indices = permutation[cutoff:]

    x_train = x[train_indices]
    y_train = y[train_indices]
    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, y_train, x_test, y_test


def score(x, w, b):
    return x @ w + b


def hinge_loss(scores, y):
    margins = y * scores
    return torch.clamp(1 - margins, min=0).mean()


def svm_objective(x, y, w, b, c=1.0):
    regularization = 0.5 * torch.sum(w.square())
    return regularization + c * hinge_loss(score(x, w, b), y)


def predict(x, w, b):
    scores = score(x, w, b)
    return torch.where(scores >= 0, torch.ones_like(scores), -torch.ones_like(scores))


def accuracy(y_pred, y_real):
    return (y_pred == y_real).float().mean()


def train_linear_svm(x, y, epochs=500, learning_rate=1e-2, c=1.0, epsilon=1e-6):
    w = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
    b = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    losses = []

    for _ in range(epochs):
        scores = score(x, w, b)
        margins = y * scores
        violating = margins < 1

        dw = w.clone()
        db = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        if torch.any(violating):
            violation_count = x.shape[0]
            dw -= c * torch.sum(y[violating, None] * x[violating], dim=0) / violation_count
            db -= c * torch.sum(y[violating]) / violation_count

        w -= learning_rate * dw
        b -= learning_rate * db

        current_loss = svm_objective(x, y, w, b, c)
        if losses and torch.abs(current_loss - losses[-1]) < epsilon:
            losses.append(current_loss)
            break
        losses.append(current_loss)

    return w, b, losses
