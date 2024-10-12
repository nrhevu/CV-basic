import os
from abc import ABC, abstractmethod

import numpy as np

from cbir import ImageSearchObject


class FeatureStore(ABC):
    def check_input_index(self, images: np.ndarray | list, features: np.ndarray | list):
        if isinstance(images, list) and isinstance(features, list):
            assert len(images) == len(
                features
            ), "Images and features must have same length"
            assert (
                len(images) > 0 and len(features) > 0
            ), "Images and features cannot be empty"
            assert isinstance(images[0], np.ndarray) and isinstance(
                features[0], np.ndarray
            ), "Images and features must be both np.ndarray or list of np.ndarray"
        elif isinstance(images, np.ndarray) and isinstance(features, np.ndarray):
            images = [images]
            features = [features]
        else:
            raise TypeError(
                f"Invalid type for images {type(images)} and features {type(features)} \
                must be both np.ndarray or list of np.ndarray"
            )

        return images, features
    
    @abstractmethod
    def add_index(self, images: np.ndarray | list, features: np.ndarray | list) -> None:
        """
        Add a feature vector to the feature store.

        Parameters
        ----------
        image : np.ndarray
            The image corresponding to the feature vector.
        feature : np.ndarray
            The feature vector to store.

        Returns
        -------
        None
        """
        pass
    
    @abstractmethod
    def retrieve(self, feature: np.ndarray, k=5) -> list[ImageSearchObject]:
        pass