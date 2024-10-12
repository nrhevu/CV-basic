from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np

from cbir import FeatureExtractor, FeatureStore
from cbir.entities.search_objects import ImageSearchObject


class CBIR():
    def __init__(self, feature_extractor : FeatureExtractor, feature_store : FeatureStore):
        self.feature_extractor = feature_extractor
        self.feature_store = feature_store
    @abstractmethod
    def indexing(self, images: List[np.ndarray | os.PathLike] | np.ndarray | os.PathLike) -> None:
        if isinstance(images, np.ndarray):
            images = [images]
        elif isinstance(images, (str, os.PathLike)):
            if not os.path.exists(images):
                raise FileNotFoundError(f"File not found: {images}")
            images = [cv2.imread(images)]
        elif isinstance(images, list):
            assert len(images) > 0, "Images cannot be empty"
            assert isinstance(images[0], (np.ndarray, str, os.PathLike)), \
                "Images must be a list of np.ndarray or PathLike"    
                
            if isinstance(images[0], (str, os.PathLike)):
                images = [cv2.imread(image) for image in images]
        
        features = []
        for image in images:
            features.append(self.feature_extractor(image))
            
        self.feature_store.add_index(images, features)
        print(f"Index Completed! {len(images)} images indexed.")
    
    @abstractmethod
    def retrieve(self, image: np.ndarray | os.PathLike, k=5) -> list[ImageSearchObject]:
        if isinstance(image, os.PathLike):
            image = cv2.imread(image)
            
        feature = self.feature_extractor(image)
        return self.feature_store.retrieve(feature, k=k)