from __future__ import annotations

from typing import Literal

import numpy as np

from cbir import FeatureStore, Retrieve
from cbir.entities.search_objects import ImageSearchObject


class NPArrayStore(FeatureStore):
    def __init__(
        self, retrieve : Retrieve
    ) -> None:
        self.retrieval = retrieve
        self.X = []
        self.y = []
        self.images = []

    def add_index(self, images: np.ndarray | list, features: np.ndarray | list) -> None:
        """
        Add an index of features and corresponding images to the storage using knn.

        Parameters
        ----------
        images : np.ndarray | list
            The images to be indexed. Can be either a single np.ndarray or a list of np.ndarray.
        features : np.ndarray | list
            The feature vectors to be indexed. Can be either a single np.ndarray or a list of np.ndarray.

        Returns
        -------
        None
        """
        images, features = super().check_input_index(images, features)
        
        for image, feature in zip(images, features):
            self.images.append(image)
            self.X.append(feature)
            self.y.append(len(self.y))

        self.retrieval.fit(self.X, self.y)

    def retrieve(self, feature: np.ndarray, k=5) -> list[ImageSearchObject]:
        distances, indices = self.retrieval.predict(feature, k=k)
        
        result = []
        images = [self.images[i] for i in indices]
        for image, distance in zip(images, distances):
            result.append(ImageSearchObject(image, 1/distance))
            
        return result
        
