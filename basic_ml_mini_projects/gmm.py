import torch
import math 
import torch 
torch.manual_seed(42)

def generate_gmm_data(
    n_samples=240,
    first_component_weight=0.65,
    seed=42,
):
    """Generate two overlapping diagonal-Gaussian components."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    if not 0 < first_component_weight < 1:
        raise ValueError("first_component_weight must be between 0 and 1.")

    generator = torch.Generator().manual_seed(seed)

    means = torch.tensor(
        [
            [-1.5, 0.0],
            [1.5, 0.75],
        ]
    )
    standard_deviations = torch.tensor(
        [
            [1.0, 0.55],
            [0.75, 1.1],
        ]
    )

    n_first = round(n_samples * first_component_weight)
    n_second = n_samples - n_first
    component_sizes = (n_first, n_second)

    component_points = []
    component_labels = []
    for component_id, component_size in enumerate(component_sizes):
        points = means[component_id] + standard_deviations[
            component_id
        ] * torch.randn(
            component_size,
            means.shape[1],
            generator=generator,
        )
        component_points.append(points)
        component_labels.append(
            torch.full(
                (component_size,),
                component_id,
                dtype=torch.long,
            )
        )

    x = torch.cat(component_points, dim=0)
    reference_labels = torch.cat(component_labels, dim=0)
    permutation = torch.randperm(n_samples, generator=generator)

    true_parameters = {
        "means": means,
        "variances": standard_deviations.square(),
        "weights": torch.tensor(component_sizes) / n_samples,
    }

    return x[permutation], reference_labels[permutation], true_parameters



def select_initial_mean(X, k):

    perms = torch.randperm(len(X))
    
    initial_means = X[perms[:k]]
    
    return initial_means


def initial_variance(X, eps):
    m = torch.mean(X, dim = 0)
    sqd = (X - m) ** 2
    variances = torch.clamp(sqd.mean(dim = 0),eps)

    return variances 


def initial_params(X,k,eps):
    means = select_initial_mean(X, k)
    global_variance = initial_variance(X, eps)
    variances = torch.stack(
          [global_variance for _ in range(k)],
          dim=0,
      )
    weights = torch.full(
          (k,),
          1.0 / k,
          dtype=X.dtype,
          device=X.device,
      )

    return means, variances, weights 



def diagonal_gaussian(X, means, variances):
    differences = (X[:, None, :] - means[None, :, :])

    per_feature_terms = math.log(2 * torch.pi) + torch.log(variances) + differences ** 2 / variances 

    log_probs = -0.5 * torch.sum(per_feature_terms, dim = 2)

    return log_probs 



def expectation(X, means, variances, weights):
    log_gaussian = diagonal_gaussian(X, means, variances)

    log_joint = log_gaussian + torch.log(weights)[None, :]

    log_evidence = torch.logsumexp(log_joint, dim = 1, keepdim = True)

    responsibilities = torch.exp(log_joint - log_evidence)

    total_log_likelihood = torch.sum(log_evidence)

    return responsibilities, total_log_likelihood


def maximization(X, responsibilities, eps):
    n_points = X.shape[0]
    soft_counts = torch.sum(responsibilities, dim = 0)

    weights = soft_counts / n_points 

    weighted_sums = responsibilities.T @ X 

    means = weighted_sums / soft_counts[:, None]

    centered = X[:, None, :] - means[None, :, :]

    weighted_squared_distances = responsibilities[:, :, None] * centered.square()

    variances = weighted_squared_distances.sum(dim=0) / soft_counts[:, None]
    variances = torch.clamp(variances, min=eps)

    return means, variances, weights


def fit_gmm(X, k, eps=1e-6, tolerance=1e-4, max_iters=100):
    means, variances, weights = initial_params(X, k, eps)
    previous_log_likelihood = None
    history = []

    for _ in range(max_iters):
        responsibilities, log_likelihood = expectation(
            X,
            means,
            variances,
            weights,
        )
        history.append(log_likelihood.item())

        if previous_log_likelihood is not None:
            improvement = log_likelihood - previous_log_likelihood
            if torch.abs(improvement) < tolerance:
                break

        means, variances, weights = maximization(X, responsibilities, eps)
        previous_log_likelihood = log_likelihood

    responsibilities, log_likelihood = expectation(
        X,
        means,
        variances,
        weights,
    )

    return means, variances, weights, responsibilities, history
