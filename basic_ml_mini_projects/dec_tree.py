
import torch



def generate_dec_tree_data():
    x = 6 * torch.rand(256, 2) - 3

    y = ((x[:, 0] > 0) & (x[:, 1] > 0)).float()
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



def gini(y):
    p = torch.sum(y) / len(y)

    return 2 * p * (1-p)

def majority_label(y):
    return (torch.sum(y) > len(y)/2)


def candidate_thresholds(feature_values):
    values = torch.unique(feature_values, sorted=True)
    return (values[:-1] + values[1:]) / 2


def split_gain(y, left_mask, right_mask):
    parent = gini(y)

    left_fraction = left_mask.float().mean()

    right_fraction = 1 - left_fraction

    weighted_children = left_fraction * gini(y[left_mask]) + right_fraction * gini(y[right_mask])

    return parent - weighted_children


def best_split(X, y):
    best_gain = 0
    best_feature = (None, None, None, None, None)
    number_of_features = X.shape[1]
    
    for feature_index in range(number_of_features):
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

def internal_node(feature, threshold, left_subtree, right_subtree):
    return {
        "feature": feature,
        "threshold": threshold,
        "left": left_subtree,
        "right": right_subtree,
    }


def build_tree(X, y, depth, max_depth, min_sample_split): 
    prediction = majority_label(y)

    if gini(y) == 0 or depth >= max_depth or len(y) < min_sample_split:
        return {'leaf': majority_label(y)}
    
    feature, threshold, gain, left_mask, right_mask = best_split(X, y)
        
    if feature is None or gain == 0:
        return {'leaf': majority_label(y)}
    
    left_subtree = build_tree(X[left_mask], y[left_mask], depth + 1, max_depth = max_depth, min_sample_split = min_sample_split)
    right_subtree = build_tree(X[right_mask], y[right_mask], depth + 1, max_depth = max_depth, min_sample_split = min_sample_split)


    return internal_node(feature, threshold, left_subtree, right_subtree)


def predict_one(sample, node):
    if "leaf" in node:
        return node['leaf']
    
    if sample[node["feature"]] <= node["threshold"]:
        return predict_one(sample, node['left'])
    else:
        return predict_one(sample, node['right'])
    

def predict_tree(X, tree):
    prediction = [predict_one(sample, tree) for sample in X]
    return torch.stack(prediction).float()
