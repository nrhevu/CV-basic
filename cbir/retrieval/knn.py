from os import PathLike
from cbir import Retrieve
import numpy as np

from cbir.cbir import ImageSearchObject
from cbir.feature_store import FeatureStore
from sklearn.neighbors import KNeighborsClassifier
from typing import Literal

class KNNRetrieval(Retrieve):
    def __init__(self, metric: Literal["uniform", "distance"] | callable | None = "uniform"):
        self.knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
        
    def fit(self, features: list, indexes: list) -> None:
        super().fit(features, indexes)
        self.knn.fit(features, indexes)
    
    def predict(self, feature: np.ndarray, k=5) -> tuple[list, list]:
        distances, indices = self.knn.kneighbors(feature.reshape(1, -1), k=k)
        
        return distances[0], indices[0]