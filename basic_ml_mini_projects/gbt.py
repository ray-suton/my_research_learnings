import torch

def mse(y):
    
    return torch.mean((y - y.mean())**2)

def internal_node(feature, threshold, left_subtree, right_subtree):
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree,
    }

def split_gain(y, left_mask, right_mask):
    parent = mse(y)

    left_fraction = left_mask.float().mean()

    right_fraction = 1 - left_fraction

    weighted_children = left_fraction * mse(y[left_mask]) + right_fraction * mse(y[right_mask])

    return parent - weighted_children

def candidate_thresholds(feature_values):
    values = torch.unique(feature_values, sorted=True)
    return (values[:-1] + values[1:]) / 2

def best_split(X, y):
    best_gain = 0
    best_feature = None
    best_threshold = None
    best_left_mask = None
    best_right_mask = None

    for feature_ind in range(X.shape[1]):
        feature_vals = X[:, feature_ind]
        thresholds = candidate_thresholds(feature_vals)
    
        for threshold in thresholds:
            left_mask = (feature_vals <= threshold)
            right_mask = ~left_mask
            gain = split_gain(y, left_mask, right_mask)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_ind
                best_threshold = threshold
                best_left_mask = left_mask
                best_right_mask = right_mask
    
    return (best_gain, best_feature, best_threshold, best_left_mask, best_right_mask)


def build_regression_tree(X, y, depth, min_sample_split, max_depth):

    leaf_pred = y.mean()

    if depth >= max_depth or len(X) < min_sample_split or mse(y) <= 0:
        return {'leaf': leaf_pred}
    
    gain, feature, thre, left_mask, right_mask  = best_split(X, y)

    if feature is None or gain <= 0:
        return {'leaf': leaf_pred}
    
    left_tree = build_regression_tree(X[left_mask], y[left_mask], depth + 1, min_sample_split, max_depth)
    right_tree = build_regression_tree(X[right_mask], y[right_mask], depth + 1, min_sample_split, max_depth)

    return internal_node(feature, thre, left_tree, right_tree)


def predict_one(sample, node):
    if 'leaf' in node:
        return node['leaf']
    
    if sample[node['feature']] <= node['threshold']:
        return predict_one(sample, node['left'])
    else:
        return predict_one(sample, node['right'])
    
def predict_tree(X, tree):
    
    predictions = [predict_one(sample, tree) for sample in X]

    return torch.stack(predictions)


def train_gbt(X, y, number_of_trees, learning_rate, max_depth, min_sample_split):
    base_prediction = y.mean()

    current_predictions = torch.tensor([base_prediction] * len(y))

    trees = []

    for _ in range(number_of_trees):
        residuals = y - current_predictions
        tree = build_regression_tree(X, residuals, depth = 0, min_sample_split = min_sample_split, max_depth = max_depth)

        correction = predict_tree(X, tree)

        current_predictions = current_predictions + learning_rate * correction

        trees.append(tree)
    
    return (base_prediction, trees, learning_rate)


def predict_gbt(X, base_prediction, trees, learning_rate):
    predictions = torch.tensor([base_prediction] * len(X))

    for tree in trees:
        correction = predict_tree(X, tree)
        predictions = predictions + learning_rate * correction

    return predictions 


def prediction_mse(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)