from collections import defaultdict, deque
import math
import torch

def generate_dbscan_data(
    n_inner=80,
    n_outer=120,
    n_noise=20,
    radial_noise=0.06,
    seed=42,
):
    """Generate two noisy concentric rings plus scattered outliers."""
    if n_inner <= 0 or n_outer <= 0:
        raise ValueError("Each ring must contain at least one point.")
    if n_noise < 0:
        raise ValueError("n_noise must be non-negative.")
    if radial_noise < 0:
        raise ValueError("radial_noise must be non-negative.")

    generator = torch.Generator().manual_seed(seed)

    inner_angles = 2 * math.pi * torch.rand(n_inner, generator=generator)
    outer_angles = 2 * math.pi * torch.rand(n_outer, generator=generator)

    inner_radii = 1.0 + radial_noise * torch.randn(
        n_inner, generator=generator
    )
    outer_radii = 2.5 + radial_noise * torch.randn(
        n_outer, generator=generator
    )

    inner_points = torch.stack(
        (
            inner_radii * torch.cos(inner_angles),
            inner_radii * torch.sin(inner_angles),
        ),
        dim=1,
    )
    outer_points = torch.stack(
        (
            outer_radii * torch.cos(outer_angles),
            outer_radii * torch.sin(outer_angles),
        ),
        dim=1,
    )

    noise_points = 7 * torch.rand(
        (n_noise, 2), generator=generator
    ) - 3.5

    x = torch.cat((inner_points, outer_points, noise_points), dim=0)
    reference_labels = torch.cat(
        (
            torch.zeros(n_inner, dtype=torch.long),
            torch.ones(n_outer, dtype=torch.long),
            torch.full((n_noise,), -1, dtype=torch.long),
        )
    )

    permutation = torch.randperm(x.shape[0], generator=generator)
    return x[permutation], reference_labels[permutation]


if __name__ == "__main__":
    x, reference_labels = generate_dbscan_data()

    print("x shape:", tuple(x.shape))
    print("reference labels shape:", tuple(reference_labels.shape))
    print("inner-ring points:", int((reference_labels == 0).sum()))
    print("outer-ring points:", int((reference_labels == 1).sum()))
    print("generated outliers:", int((reference_labels == -1).sum()))


def determine_point(X, sample, min_samples, eps):
    count = 0
    for point in X:
        if torch.sqrt(torch.sum((point - sample) ** 2)) <= eps:
            count += 1
    
    if 0 < count < min_samples:
        return 'Border'
    elif count >= min_samples:
        return 'Core'
    else:
        return 'Noise'

def find_core_neighbors(X, sample, eps):
    neighbors = []
    for (i,point) in enumerate(X):
        if torch.sqrt(torch.sum((point - sample) ** 2)) <= eps:
            neighbors.append((point, i))
    
    return neighbors
    
    
def dbscan(X, min_sample_size, eps):
    visited = [False] * len(X)
    labels = [-2] * len(X)


    clusters = defaultdict(list)
    count = 0
    noise = []
   
    for (i,sample) in enumerate(X):
        if visited[i]:
            continue
        label = determine_point(X, sample, min_sample_size, eps)
        visited[i] = True
        if label != 'Core':
            labels[i] = -1
            continue
        cores = deque([sample])
        clusters[count].append(sample)
        labels[i] = count
        while cores:
            cur = cores.popleft()
            neighbors = find_core_neighbors(X, cur, eps)
            for neighbor, ind in neighbors:
                label = determine_point(X, neighbor, min_sample_size, eps)
                if label == "Core":
                    if visited[ind]:
                        continue
                    clusters[count].append(neighbor)
                    cores.append(neighbor)
                    visited[ind] = True
                    labels[ind] = count
                elif labels[ind] == -1 or labels[ind] == -2:
                    clusters[count].append(neighbor)
                    visited[ind] = True
                    labels[ind] = count
                    
                
        count += 1
    
    for i in range(len(labels)):
        if labels[i] == -1:
            noise.append(X[i])
    
    return labels, noise

    