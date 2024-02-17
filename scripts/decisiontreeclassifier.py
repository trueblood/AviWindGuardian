import numpy as np
#from line_profiler import profile

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        # for leaf node
        self.value = value

class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=2, feature_selection_strategy=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.feature_selection_strategy = feature_selection_strategy

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, current_depth=0):
        num_samples, num_features = X.shape
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self._get_best_split(X, y, num_samples, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self._build_tree(X[best_split["dataset_left"]], y[best_split["dataset_left"]], current_depth+1)
                right_subtree = self._build_tree(X[best_split["dataset_right"]], y[best_split["dataset_right"]], current_depth+1)
                return Node(best_split["feature_index"], best_split["threshold"], left_subtree, right_subtree)
        # compute leaf node
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)

    def _get_best_split(self, X, y, num_samples, num_features):
        best_split = {}
        max_info_gain = -float("inf")
        
        # Determine number of features to consider
        if self.feature_selection_strategy == 'sqrt':
            num_features_considered = int(np.sqrt(num_features))
        elif self.feature_selection_strategy == 'log2':
            num_features_considered = int(np.log2(num_features))
        else:  # None or invalid strategy falls back to using all features
            num_features_considered = num_features

         # Randomly select features to consider for the split
        features_considered = np.random.choice(range(num_features), size=num_features_considered, replace=False)

        for feature_index in features_considered:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                dataset_left, dataset_right = self._split_dataset(X, y, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = y, y[dataset_left], y[dataset_right]
                    current_info_gain = self._information_gain(y, left_y, right_y)
                    if current_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"], best_split["dataset_right"] = dataset_left, dataset_right
                        best_split["info_gain"] = current_info_gain
                        max_info_gain = current_info_gain
        return best_split

    def _split_dataset(self, X, y, feature_index, threshold):
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        return left_indices, right_indices

    def _information_gain(self, parent, left_child, right_child):
        weight_l = len(left_child) / len(parent)
        weight_r = len(right_child) / len(parent)
        gain = self._entropy(parent) - (weight_l*self._entropy(left_child) + weight_r*self._entropy(right_child))
        return gain

    def _entropy(self, y):
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy

    def _calculate_leaf_value(self, y):
        y = list(y)
        return max(y, key=y.count)

    def predict(self, X):
        predictions = [self._make_prediction(x, self.root) for x in X]
        return np.array(predictions)

    def _make_prediction(self, x, tree):
        if tree.value is not None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self._make_prediction(x, tree.left)
        else:
            return self._make_prediction(x, tree.right)
