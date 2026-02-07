import pandas as pd
import numpy as np

class classification_binaryTree():
    def __init__(self, maxDepth = float('inf')):
        self.maxDepth = maxDepth
        self.root = None
    
    def fit(self, X, y):
        data = pd.concat([X, y.rename("target")], axis=1)
        self.root = self._build_tree(data, 0, self.maxDepth)
        print("tree successfully built!")

    def predict(self, X):
        predictions = []
        for _, instance in X.iterrows():
            node = self.root
            while (not node.is_leaf()):
                node = node.get_next_node(instance)

            predictions.append(node.get_threshold())

        return np.array(predictions)
    
    def print_tree(self, node = None, level = 0, isLeft = False, isRight = False):
        if node == None:
            node = self.root
        print(f"level: {level}, is_right: {isRight == True}, is_left: {isLeft == True}, node info: " + node.get_node_info())
        if node.is_leaf():
            return
        
        self.print_tree(node = node.get_left_node(), level = level + 1, isLeft = True, isRight = False)
        self.print_tree(node = node.get_right_node(), level = level + 1, isLeft = False, isRight = True)

    # --- Encapsulated Helper Methods ---

    def _build_tree(self, data, depth, maxDepth):
        if self._stopping_condition(data, depth, maxDepth):
            return self._create_leaf(data)
        
        best_feat, best_threshold, best_gini = self._find_best_split(data)

        left_data, right_data = self._split_data(data, best_feat, best_threshold)

        left_subtree = self._build_tree(left_data, depth + 1, maxDepth)
        right_subtree = self._build_tree(right_data, depth + 1, maxDepth)

        return Node(best_feat, best_threshold, best_gini, left_subtree, right_subtree)

    def _stopping_condition(self, data, depth, maxDepth):
        if depth > maxDepth:
            return True
        if (self._gini_impurity(data) == 0):
            return True
        return False

    def _find_best_split(self, data):
        best_gini = float('inf')
        best_feat = None
        best_threshold = None
        
        target_name = data.iloc[:, -1].name
        
        for name, values in data.items():
            if name != target_name:
                thresholds = self._possible_splits_for(values)
                for threshold in thresholds:
                    left, right = self._split_data(data, name, threshold)
                    gini = self._weighted_gini(left, right)

                    if gini < best_gini :
                        best_gini = gini
                        best_feat = name
                        best_threshold = threshold

        return best_feat, best_threshold, best_gini

    def _possible_splits_for(self, values):
        thresholds = []
        if (values.dtype == 'category'):
            for category in values:
                thresholds.append(category)
        else :
            sorted_values = values.sort_values().unique()
            for i in range(len(sorted_values) - 1):
                thresholds.append(np.mean([sorted_values[i], sorted_values[i+1]]))

        return thresholds

    def _split_data(self, data, name, threshold):
        if type(threshold) == str:
            mask = data[name] == threshold
        else:
            mask = data[name] <= threshold
        
        return data[mask], data[~mask]

    def _weighted_gini(self, left, right):
        left_size, right_size = left.shape[0], right.shape[0]
        total_size = left_size + right_size
        return ((left_size / total_size) * self._gini_impurity(left)) + ((right_size / total_size) * self._gini_impurity(right))

    def _target_proportions(self, target_series):
        _, counts = np.unique(target_series, return_counts=True) 
        return counts / target_series.shape[0]

    def _gini_impurity(self, data, proportions = None):
        if proportions is None:
            proportions = self._target_proportions(data.iloc[:, -1])
        return 1 - np.sum(proportions ** 2)

    def _create_leaf(self, data):
        values, counts = np.unique(data.iloc[:, -1], return_counts=True) 
        proportions = counts / data.shape[0]
        maj_idx = np.argmax(proportions)
        return Node(data.iloc[:, -1].name, values[maj_idx], self._gini_impurity(data, proportions), None, None)


class Node:
    def __init__(self, feature, threshold, impurity, left, right):
        self.feature = feature
        self.threshold = threshold
        self.impurity = impurity
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.left == None and self.right == None
    
    def get_threshold(self):
        return self.threshold

    def get_next_node(self, instance):
        if type(self.threshold) == str :
            if instance[self.feature] == self.threshold :
                return self.left
            else : 
                return self.right
        else :
            if instance[self.feature] <= self.threshold:
                return self.left
            else : 
                return self.right
    def get_right_node(self):
        return self.right
    
    def get_left_node(self):
        return self.left
    
    def get_node_info(self):
        return f"feat_name: {self.feature}, threshold: {self.threshold}, gini_impurity: {self.impurity}, is_leaf: {self.is_leaf()}"