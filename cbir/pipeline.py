from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from cbir import (BatchFeatureExtractor, FeatureExtractor, FeatureStore,
                  SingleFeatureExtractor)
from cbir.entities.search_objects import ImageSearchObject


class CBIR():
    def __init__(self, feature_extractor : FeatureExtractor, feature_store : FeatureStore):
        self.feature_extractor = feature_extractor
        self.feature_store = feature_store
    @abstractmethod
    def indexing(self, images: List[np.ndarray | os.PathLike] | np.ndarray | os.PathLike) -> None:
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
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
        if isinstance(self.feature_extractor, SingleFeatureExtractor):
            # for image in tqdm(images, desc="Extracting Features"):
            for image in images:
                features.append(self.feature_extractor(image))
        elif isinstance(self.feature_extractor, BatchFeatureExtractor):
            features = self.feature_extractor(images).tolist()
            
        self.feature_store.add_index(images, features)
        # print(f"Index Completed! {len(images)} images indexed.")
    
    @abstractmethod
    def retrieve(self, images: np.ndarray | os.PathLike, k=5) -> list[ImageSearchObject]:
        if isinstance(images, os.PathLike):
            images = cv2.imread(images)
        
        if isinstance(self.feature_extractor, SingleFeatureExtractor):
            feature = self.feature_extractor(images)            
            return self.feature_store.retrieve(feature, k=k)
        
        elif isinstance(self.feature_extractor, BatchFeatureExtractor):
            features = self.feature_extractor(images)
            result = []
            for feature in features:
                result.append(self.feature_store.retrieve(feature, k=k))
                
            return result