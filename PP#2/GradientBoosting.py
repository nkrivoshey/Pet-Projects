import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoosting:
    def __init__(self, n_estimators, learning_rate,splitter, min_samples_leaf, criterion, max_features = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []

    def compute_gradient(self, y, prediction):
        # Вычисляем градиент как разность между фактическим значением и текущим предсказанием
        return y - prediction

    def fit(self, X, y):
        # Инициализируем предсказания нулями
        predictions = np.zeros_like(y, dtype=float)

        # Обучаем каждое дерево в ансамбле
        for i in range(self.n_estimators):
            # Вычисляем градиент
            gradient = self.compute_gradient(y, predictions)

            # Создаем новое дерево решений
            tree = DecisionTreeRegressor(criterion=self.criterion, splitter=self.splitter, min_samples_leaf=self.min_samples_leaf,  max_features = self.max_features)
            tree.fit(X, gradient)

            # Вычисляем обновление предсказаний с учетом нового дерева
            update = self.learning_rate * np.array(tree.predict(X))

            # Обновляем предсказания
            predictions += update

            # Добавляем дерево в список
            self.trees.append(tree)

    def predict(self, X):
        # Инициализируем предсказания нулями
        predictions = np.zeros(X.shape[0], dtype=float)

        # Предсказываем каждым деревом в ансамбле
        for tree in self.trees:
            predictions += self.learning_rate * np.array(tree.predict(X))

        return predictions