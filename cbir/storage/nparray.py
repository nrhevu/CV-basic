from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch

from cbir import FeatureStore, Retrieve
from cbir.entities.search_objects import ImageSearchObject
from cbir.utils.distance import d2s_typing, get_d2s_transform


class NPArrayStore(FeatureStore):
    def __init__(self, retrieve: Retrieve) -> None:
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
    
    def retrieve(
        self,
        feature: np.ndarray,
        k=5,
        return_image=False,
        distance_transform: d2s_typing = "exp",
    ) -> list[ImageSearchObject]:
        distances, indices = self.retrieval.predict(feature, k=k)

        d2s_func = get_d2s_transform(distance_transform)
        
        result = []
        for index, distance in zip(indices, distances):
            image = None
            if return_image:
                image = self.image[index]
            result.append(ImageSearchObject(index, d2s_func(distance), image))

        return result
