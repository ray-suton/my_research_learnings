import torch

torch.manual_seed(42)


def generate_gnb_data(n_per_class=64, dimensions=2):
    if dimensions != 2:
        raise ValueError("This starter dataset is intentionally 2D.")

    class_0_mean = torch.tensor([-2.0, -1.0])
    class_0_std = torch.tensor([0.7, 1.0])
    class_1_mean = torch.tensor([2.0, 1.5])
    class_1_std = torch.tensor([1.1, 0.8])

    x_0 = class_0_mean + class_0_std * torch.randn(n_per_class, dimensions)
    x_1 = class_1_mean + class_1_std * torch.randn(n_per_class, dimensions)

    x = torch.cat([x_0, x_1], dim=0)
    y = torch.cat(
        [
            torch.zeros(n_per_class, dtype=torch.long),
            torch.ones(n_per_class, dtype=torch.long),
        ],
        dim=0,
    )

    permutation = torch.randperm(x.shape[0])
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


if __name__ == "__main__":
    x, y = generate_gnb_data()
    x_train, y_train, x_test, y_test = split_data(x, y)

    print("x shape:", tuple(x.shape))
    print("y shape:", tuple(y.shape))
    print("train shape:", tuple(x_train.shape), tuple(y_train.shape))
    print("test shape:", tuple(x_test.shape), tuple(y_test.shape))
    print("class counts:", torch.bincount(y))



def get_mean(X, mask):
    return torch.sum(X[mask], dim = 0) / len(X[mask])


def get_var(X, mask, mean_vec, eps = 1e-3):
    
    return ((X[mask] - mean_vec) ** 2).mean(dim=0) + eps

def get_mats(X, y):

    mask_0 = (y == 0)
    mask_1 = (y == 1)
    mean_0 = get_mean(X, mask_0)
    mean_1 = get_mean(X, mask_1)
    prob_0 = (y == 0).sum() / len(y)
    prob_1 = (y == 1).sum() / len(y)

    probs = torch.stack([prob_0, prob_1], dim = 0)

    mean_mat = torch.stack([mean_0, mean_1], dim = 0)
    
    var_mat_0 = get_var(X, mask_0, mean_mat[0])
    var_mat_1 = get_var(X, mask_1, mean_mat[1])

    var_mat = torch.stack([var_mat_0, var_mat_1], dim = 0)

    return {
      "means": mean_mat,
      "vars": var_mat,
      "priors": probs,
    }


def predict_one(x, trained, eps = 1e-3):
    mean_mat, var_mat, probs = trained['means'], trained['vars'], trained['priors']
    log_like_0 = torch.sum( -0.5 * torch.log(2 * torch.pi * var_mat[0]) - ((x - mean_mat[0]) ** 2) / (2 * var_mat[0]))
    log_like_1 = torch.sum( -0.5 * torch.log(2 * torch.pi * var_mat[1]) - ((x - mean_mat[1]) ** 2) / (2 * var_mat[1]))

    
    score_0 = log_like_0 + torch.log(probs[0] + eps)
    score_1 = log_like_1 + torch.log(probs[1] + eps)

    if score_1 > score_0:
        return 1 
    else:
        return 0

def predict_whole(X, trained):
    
    y_pred = torch.tensor([predict_one(sample, trained) for sample in X])

    return y_pred 

def accuracy(y_pred, y_real):
    return (y_pred == y_real).float().mean()