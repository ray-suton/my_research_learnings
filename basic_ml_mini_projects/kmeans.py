import torch

torch.manual_seed(42)


def generate_kmeans_data(n_per_cluster=50, dimensions=2, cluster_std=0.55):
    if dimensions != 2:
        raise ValueError("This starter dataset is intentionally 2D.")

    centers = torch.tensor(
        [
            [-3.0, -2.0],
            [0.5, 3.0],
            [3.0, -1.0],
        ]
    )

    xs = []
    labels = []
    for cluster_id, center in enumerate(centers):
        points = center + cluster_std * torch.randn(n_per_cluster, dimensions)
        xs.append(points)
        labels.append(torch.full((n_per_cluster,), cluster_id, dtype=torch.long))

    x = torch.cat(xs, dim=0)
    y = torch.cat(labels, dim=0)

    permutation = torch.randperm(x.shape[0])
    return x[permutation], y[permutation], centers


if __name__ == "__main__":
    x, y, true_centers = generate_kmeans_data()

    print("x shape:", tuple(x.shape))
    print("y shape:", tuple(y.shape))
    print("true centers shape:", tuple(true_centers.shape))
    print("cluster counts:", torch.bincount(y))


def select_random_points(X, k):
    per = torch.randperm(len(X))
    selection = per[:k]
    
    return X[selection]

def calculate_dist(centroids, x):
   
    dists = ((x - centroids) ** 2).sum(dim = 1)

    return torch.argmin(dists)

def update_centroid(x_group):
    group = torch.stack(x_group)
    return group.mean(dim = 0)


def train(X, k, eps = 1e-3, max_iters = 100):
    centroids = select_random_points(X, k)
    last_centroids = torch.zeros(centroids.shape)
    for i in range(max_iters): 
        last_centroids = centroids.clone()
        clusters = [[] for _ in range(k)]

        for x in X:
            group = calculate_dist(centroids, x).item()
            clusters[group].append(x)
        
        for i in range(k):
            if len(clusters[i]) == 0:
                centroids[i] = X[torch.randint(len(X), (1,)).item()]
            else:
                centroids[i] = update_centroid(clusters[i])
        
        if ((centroids - last_centroids) ** 2).float().sum() <= eps:
            break
    return centroids, clusters