
import torch
from dec_tree import (
    candidate_thresholds,
    split_gain,
    majority_label,
    gini,
    internal_node,
    predict_one,
)
def predict_tree(X, tree):
    prediction = [predict_one(sample, tree) for sample in X]
    return torch.stack(prediction).float()


def best_random_split(X, y, max_features):
    n_features = X.shape[1]
    feature_indices = torch.randperm(n_features)[:max_features]
    best_gain = 0
    best_feature = (None, None, None, None, None)
    
    for feature_index in feature_indices:
        feature_values = X[:, feature_index]
        thresholds = candidate_thresholds(feature_values)

        for threshold in thresholds:
            left_mask = (feature_values <= threshold)
            right_mask = ~left_mask

            s_gain = split_gain(y, left_mask, right_mask)

            if s_gain > best_gain:
                best_gain = s_gain
                best_feature = (feature_index, threshold, s_gain, left_mask, right_mask)
    
    return best_feature 

def build_tree_rand(X, y, depth, max_depth, min_sample_split, max_features): 
    prediction = majority_label(y)

    if gini(y) == 0 or depth >= max_depth or len(y) < min_sample_split:
        return {'leaf': majority_label(y)}
    
    feature, threshold, gain, left_mask, right_mask = best_random_split(X, y, max_features)
        
    if feature is None or gain == 0:
        return {'leaf': majority_label(y)}
    
    left_subtree = build_tree_rand(X[left_mask], y[left_mask], depth + 1, max_depth = max_depth, min_sample_split = min_sample_split, max_features = max_features)
    right_subtree = build_tree_rand(X[right_mask], y[right_mask], depth + 1, max_depth = max_depth, min_sample_split = min_sample_split, max_features = max_features)


    return internal_node(feature, threshold, left_subtree, right_subtree)

def build_random_tree(X, y, max_features, max_depth, min_sample_split):
    n_samples = X.shape[0]
    sample_indices = torch.randint(low=0, high=n_samples, size=(n_samples,))
    X_bootstrap = X[sample_indices]
    y_bootstrap = y[sample_indices]
    return build_tree_rand(X_bootstrap, y_bootstrap, depth = 0, max_depth = max_depth, min_sample_split = min_sample_split, max_features = max_features)

def train_forest(X, y, n_trees, N, max_depth, min_sample_split):
    trees = []

    for _ in range(n_trees):
        trees.append(build_random_tree(X, y, N, max_depth = max_depth, min_sample_split = min_sample_split))

    return trees


def predict_forest(x, trees):
    tree_predictions = torch.stack([predict_tree(x, tree) for tree in trees], dim=0,)

    return (tree_predictions.mean(dim=0) > 0.5).float()

