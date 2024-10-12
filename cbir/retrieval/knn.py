from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from cbir import Retrieve


class KNNRetrieval(Retrieve):
    def __init__(self, metric: str | callable | None = "uniform"):
        self.knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        
    def fit(self, features: list, indexes: list) -> None:
        super().fit(features, indexes)
        self.knn.fit(features, indexes)
    
    def predict(self, feature: np.ndarray, k=5) -> tuple[list, list]:
        distances, indices = self.knn.kneighbors(feature.reshape(1, -1), n_neighbors=k)
        
        return distances[0], indices[0]