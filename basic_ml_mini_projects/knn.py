import torch 

torch.manual_seed(42)
def generate_knn_data(dimensions = 2, noise_constant = 3):
    noise = noise_constant * torch.randn((128,2))
    
    if dimensions == 2:
        x = torch.randint(-3, 3, (128,2)).float() + noise 
    else:
        raise ValueError("Dimensions must be 2")
    
    y = (x[:,0] * x[:,1] > 0).float()
    return x, y


def split_data(x, y, train_ratio):

    permutation = torch.randperm(x.shape[0])
    cutoff = int(x.shape[0] * train_ratio)

    train_indices = permutation[:cutoff]
    test_indices = permutation[cutoff:]
    x_train = x[train_indices]
    y_train = y[train_indices]

    x_test = x[test_indices]
    y_test = y[test_indices]

    return x_train, y_train, x_test, y_test


def pairwise_squared_distances(x_query, x_train):
    # x_query: (M, D)
    # x_train: (N, D)
    # return: (M, N)

    query_norms = (x_query ** 2).sum(dim=1, keepdim=True)
    train_norms = (x_train ** 2).sum(dim=1).unsqueeze(0)
    dot_products = x_query @ x_train.T

    distances = query_norms + train_norms - 2 * dot_products
    return torch.clamp(distances, min=0.0)

def knn_predict(x_train, y_train, x_test, k = 3):
    distances = pairwise_squared_distances(x_test, x_train)

    _ , neighbor_indices = torch.topk(distances,  k = k, largest = False)

    neighbor_labels = y_train[neighbor_indices]
    votes_for_one = neighbor_labels.sum(dim=1)
    predictions = (votes_for_one > k / 2).float()

    return predictions 

