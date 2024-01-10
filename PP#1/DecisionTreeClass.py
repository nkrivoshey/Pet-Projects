import numpy as np
import pandas as pd

class DecisionTree:
    def __init__(self, criterion=None, splitter=None, min_samples_leaf=None,  max_features = None, task=None):
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.task = task
        self.max_features = max_features
        self.tree = None

    def calculate_criterion(self, y):
        if self.task == 'classification':
            if self.criterion == 'gini':
                classes, counts = np.unique(y, return_counts=True)
                probabilities = counts / len(y)
                gini_index = 1 - np.sum(probabilities**2)
                return gini_index
            elif self.criterion == 'entropy':
                classes, counts = np.unique(y, return_counts=True)
                probabilities = counts / len(y)
                entropy = -np.sum(probabilities * np.log2(probabilities))
                return entropy
        elif self.task == 'regression':
            if self.criterion == 'mse':
                return np.mean((y - np.mean(y))**2)

    def split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def find_best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None

        best_feature, best_threshold = None, None
        best_criterion = np.inf  # Инициализация наилучшего критерия

        if self.splitter == 'best':
            features_indices = range(n)
        elif self.splitter == 'random':
            if self.max_features is None:
                features_indices = range(n)
            elif self.max_features == 'auto' or self.max_features == 'sqrt':
                features_indices = np.random.choice(n, size=int(np.sqrt(n)), replace=False)
            elif self.max_features == 'log2':
                features_indices = np.random.choice(n, size=int(np.log2(n)), replace=False)

        for feature_index in features_indices:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                criterion_left = self.calculate_criterion(y_left)
                criterion_right = self.calculate_criterion(y_right)
                weighted_criterion = (len(y_left) / m) * criterion_left + (len(y_right) / m) * criterion_right

                if weighted_criterion < best_criterion:
                    best_criterion = weighted_criterion
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y):
        if self.task == 'classification':
            if len(np.unique(y)) == 1:
                return {'class': y[0]}

            best_feature, best_threshold = self.find_best_split(X, y)

            if best_feature is None:
                return {'class': np.argmax(np.bincount(y))}

            X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)

            return {
                'feature_index': best_feature,
                'threshold': best_threshold,
                'left': self.build_tree(X_left, y_left),
                'right': self.build_tree(X_right, y_right)
            }
        elif self.task == 'regression':
            if len(np.unique(y)) == 1:
                return {'value': np.mean(y)}

            best_feature, best_threshold = self.find_best_split(X, y)

            if best_feature is None:
                return {'value': np.mean(y)}

            X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)

            return {
                'feature_index': best_feature,
                'threshold': best_threshold,
                'left': self.build_tree(X_left, y_left),
                'right': self.build_tree(X_right, y_right)
            }
        
    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_instance(self, x, tree):
        if self.task == 'classification':
            if 'class' in tree:
                return tree['class']
            if x[tree['feature_index']] <= tree['threshold']:
                return self.predict_instance(x, tree['left'])
            else:
                return self.predict_instance(x, tree['right'])
        elif self.task == 'regression':
            if 'value' in tree:
                return tree['value']
            if x[tree['feature_index']] <= tree['threshold']:
                return self.predict_instance(x, tree['left'])
            else:
                return self.predict_instance(x, tree['right'])
       
    def predict(self, X):
        return [self.predict_instance(x, self.tree) for x in X]
