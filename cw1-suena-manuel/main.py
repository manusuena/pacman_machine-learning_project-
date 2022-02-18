

import numpy as np
from collections import Counter



    def euclidean_distance(x1, x2):
     return np.sqrt(np.sum((x1 - x2)**2))




    def Knn(self, k=3):
        self.k = k

    def fit(self, data, target):
        self.data_train = data
        self.target_train = target

    def predict(self, Features):
        y_pred = [self._predict(features) for features in Features]
        return np.array(y_pred)

    def _predict(self, features, k):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(features, data_train) for data_train in self.data_train]
        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:k]
        # Extract the labels of the k nearest neighbor training samples
        k_neighbor_labels = [self.target_train[i] for i in k_idx]
        # return the most common class label
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]